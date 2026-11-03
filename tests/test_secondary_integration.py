"""Exercise real secondary refits while proving that test patients stay excluded."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from lidc_experiment.secondary import run_secondary
from lidc_experiment.training import fit_final_models, load_model, tune_models
from test_training import fixture, smoke_config


class SecondaryIntegrationTests(unittest.TestCase):
    def test_secondary_refits_preserve_development_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort, features = fixture(root)
            cohort["rating_median"] = np.where(cohort.label == 0, 1.0, 5.0)
            cohort["reader_count"] = 2
            config = smoke_config()
            config["secondary"] = dict(learning_fractions=[1.0], seeds=[17, 73],
                                       mask_cases_per_cell=0)

            # Mean-indeterminate nodules may join the alternative-label analysis
            # for an existing development patient, but never for a test patient.
            additional = []
            for patient_index, identifier in ((0, "median_added_train"), (42, "median_forbidden_test")):
                observation = cohort.iloc[patient_index].to_dict()
                observation.update(nodule_id=identifier, label=-1, rating_median=5.0,
                                   npz_path=cohort.iloc[1].npz_path)
                additional.append(observation)
                feature = features.iloc[1].to_dict()
                feature["nodule_id"] = identifier
                features = pd.concat([features, pd.DataFrame([feature])], ignore_index=True)
            all_labels = pd.concat([cohort, pd.DataFrame(additional)], ignore_index=True)
            (root / "features").mkdir()
            cohort.to_csv(root / "cohort.csv", index=False)
            all_labels.to_csv(root / "common_all_labels.csv", index=False)
            features.to_csv(root / "features" / "radiomics_features.csv", index=False)
            selection = tune_models(cohort, features, config, root / "training")
            fit_final_models(cohort, features, selection, config, root / "training")

            with patch("lidc_experiment.dataset.prepare_case",
                       side_effect=AssertionError("A zero-case mask sample must never read raw CT")):
                summary = run_secondary(root, config)

            self.assertTrue(summary["completed"])
            self.assertFalse(summary["test_used_for_selection"])
            self.assertEqual(summary["development_fits"], 17)
            self.assertEqual(summary["mask_cases"], 0)
            self.assertEqual(summary["median_newly_eligible"], 1)
            metrics = pd.read_csv(root / "secondary" / "development_metrics.csv")
            self.assertEqual(metrics.groupby("analysis").size().to_dict(), {
                "learning_curve": 4, "class_weight_toggle": 4,
                "no_augmentation": 1, "seed_stability": 4, "median_labels": 4})
            self.assertTrue(np.isfinite(metrics[["roc_auc", "average_precision"]]).all().all())
            self.assertEqual(set(metrics.loc[metrics.analysis == "seed_stability", "seed"]), {17, 73})

            train = cohort[cohort.partition == "train"]
            validation = cohort[cohort.partition == "validation"]
            test = cohort[cohort.partition == "test"]
            predictions = sorted((root / "secondary" / "fits").rglob("validation_predictions.csv"))
            self.assertEqual(len(predictions), 17)
            for prediction_path in predictions:
                with self.subTest(analysis=str(prediction_path.relative_to(root))):
                    prediction = pd.read_csv(prediction_path)
                    self.assertEqual(set(prediction.nodule_id), set(validation.nodule_id))
                    self.assertEqual(set(prediction.patient_id), set(validation.patient_id))
                    self.assertFalse(set(prediction.patient_id) & set(test.patient_id))
                    model_path = prediction_path.parent / (
                        "model.pt" if prediction_path.parent.name == "cnn" else "model.joblib")
                    model = load_model(model_path)
                    self.assertEqual(set(model.metadata["fit_patient_ids"]), set(train.patient_id))
                    fitted_nodules = set(model.metadata["fit_nodule_ids"])
                    self.assertFalse(fitted_nodules & set(test.nodule_id))
                    self.assertNotIn("median_forbidden_test", fitted_nodules)
                    expected = set(train.nodule_id)
                    if "median_labels" in prediction_path.parts:
                        expected.add("median_added_train")
                    self.assertEqual(fitted_nodules, expected)
                    if model.feature_processor is not None:
                        self.assertEqual(set(model.feature_processor.fit_patient_ids_), set(train.patient_id))
                    if model.model_name == "svm":
                        self.assertIsNone(model.calibrator)
                    if "class_weight_toggle" in prediction_path.parts:
                        setting = model.metadata["settings"]
                        if model.model_name == "cnn":
                            self.assertTrue(setting["weighted_loss"])
                        else:
                            self.assertEqual(setting["class_weight"], "balanced")
                    if "no_augmentation" in prediction_path.parts:
                        self.assertFalse(model.metadata["augment"])
                    if "seed_stability" in prediction_path.parts:
                        self.assertIn(model.metadata["seed"], [17, 73])

            changes = pd.read_csv(root / "secondary" / "median_label_changes.csv")
            self.assertIn("median_added_train", set(changes.nodule_id))
            self.assertNotIn("median_forbidden_test", set(changes.nodule_id))
            self.assertFalse(set(changes.patient_id) & set(test.patient_id))
            saved_summary = json.loads((root / "secondary" / "complete.json").read_text())
            self.assertEqual(saved_summary, summary)
            self.assertTrue((root / "secondary" / "learning_curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
