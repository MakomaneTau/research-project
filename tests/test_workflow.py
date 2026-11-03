"""Exercise the freeze boundary using synthetic models, never real test data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lidc_experiment.evaluation import evaluate_predictions
from lidc_experiment.explanations import generate_explanations
from lidc_experiment.reporting import write_report
from lidc_experiment.training import fit_final_models, tune_models
from lidc_experiment.utils import environment_record, write_json
from lidc_experiment.cli import run_stages
from lidc_experiment.workflow import (
    freeze_experiment, load_models, make_predictions, record_geometry_review,
    release_checks, select_thresholds, verify_frozen,
)
from test_training import fixture, smoke_config


class WorkflowTests(unittest.TestCase):
    def test_frozen_resume_never_revisits_model_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIn("tune", run_stages(root))
            (root / "frozen.json").write_text("{}")
            self.assertEqual(run_stages(root), ["evaluate", "benchmark", "report"])
            write_json(root / "stages" / "evaluate.json", {"completed": True})
            self.assertEqual(run_stages(root), ["benchmark", "report"])

    def test_common_missing_predictions_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort, features = fixture(directory)
            from lidc_experiment.training import predict_model
            with patch("lidc_experiment.training.predict_model", return_value=cohort.iloc[:1][["nodule_id"]].assign(score=.5, probability=.5)):
                with self.assertRaisesRegex(ValueError, "Incomplete requested"):
                    make_predictions({"cnn": object()}, cohort, features, smoke_config())

    def test_frozen_pipeline_and_generated_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort, features = fixture(root)
            cohort["source_image_uids"] = cohort.patient_id.map(lambda key: json.dumps([key]))
            cohort["diameter_mm"] = 8.0
            cohort["rating_range"] = 1.0
            cohort["reader_count"] = 2
            (root / "features").mkdir()
            (root / "data").mkdir()
            write_json(root / "environment.json", environment_record())
            (root / "requirements.lock.txt").write_text("Synthetic fixture environment recorded in environment.json\n")
            cohort.to_csv(root / "cohort.csv", index=False)
            features.to_csv(root / "features" / "radiomics_features.csv", index=False)
            config = smoke_config()
            config["evaluation"] = dict(bootstrap_replicates=8, bootstrap_max_attempts=80)
            config["explanations"] = dict(permutation_repeats=1, development_cases_per_class=1)
            selected = tune_models(cohort, features, config, root / "training")
            fit_final_models(cohort, features, selected, config, root / "training")
            thresholds = select_thresholds(root, config)
            with self.assertRaisesRegex(ValueError, "Inspect development"):
                release_checks(root, config)
            (root / "data" / "development_overlays.png").write_bytes(b"synthetic-test-fixture")
            record_geometry_review(root, "Synthetic fixture: release-boundary contract test")
            release_checks(root, config)
            models = load_models(root)
            generate_explanations(models, cohort, features, thresholds, config,
                                  root / "explanations", selected["selected_traditional"])
            write_json(root / "secondary" / "complete.json", {"completed": True, "synthetic_fixture": True})
            freeze_experiment(root, config)
            verify_frozen(root, config)
            original = Path(cohort.iloc[0].npz_path).read_bytes()
            Path(cohort.iloc[0].npz_path).write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "Artifact changed"):
                verify_frozen(root, config)
            Path(cohort.iloc[0].npz_path).write_bytes(original)
            test = cohort[cohort.partition == "test"]
            predictions = make_predictions(models, test, features, config)
            evaluate_predictions(predictions, thresholds, config, root / "results", selected["selected_traditional"])
            write_json(root / "cohort_summary.json", {"binary_nodules": len(cohort), "binary_patients": cohort.patient_id.nunique()})
            report = Path(write_report(root))
            self.assertIn("Frozen-test model comparison", report.read_text())
            feature_file = root / "features" / "radiomics_features.csv"
            feature_file.write_text(feature_file.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Artifact changed"):
                verify_frozen(root, config)


if __name__ == "__main__":
    unittest.main()
