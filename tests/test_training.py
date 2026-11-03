"""Checks against leakage, calibration and checkpoint errors that invalidate results."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lidc_experiment.training import (
    CompactCNN, NoduleDataset, RadiomicsPreprocessor, SigmoidCalibrator,
    _grids, _inner_patients, _training_folds, fit_final_models,
    fit_selected_model, load_model, predict_model, tune_models,
)


def fixture(directory, count=48):
    rng = np.random.default_rng(84)
    rows, features = [], []
    yy, xx = np.ogrid[:64, :64]
    mask = ((xx - 32) ** 2 + (yy - 32) ** 2 <= 7 ** 2).astype(np.uint8)
    for index in range(count):
        label = index % 2
        filename = Path(directory) / f"case_{index}.npz"
        intensity = mask.astype(np.float32) * (0.25 + 0.5 * label)
        np.savez_compressed(filename, image=np.stack([intensity, mask.astype(np.float32)]),
                            mask=mask, hu=intensity * 1400 - 1000)
        partition = "train" if index < 36 else ("validation" if index < 42 else "test")
        rows.append(dict(patient_id=f"patient_{index}", nodule_id=f"nodule_{index}",
                         series_uid=f"series_{index}", label=label, partition=partition,
                         fold=index % 3 if partition == "train" else -1, npz_path=str(filename)))
        value = label * 4 + rng.normal(0, 0.5)
        features.append(dict(nodule_id=f"nodule_{index}", original_signal=value,
                             original_duplicate=2 * value, original_noise=rng.normal(),
                             original_constant=1, mean_malignancy_rating=1 + 4 * label))
    return pd.DataFrame(rows), pd.DataFrame(features)


def smoke_config():
    return dict(profile="smoke", training=dict(seed=42, cnn=dict(max_epochs=1, patience=1)),
                runtime=dict(device="cpu", cpu_threads=1, num_workers=0), grids={
                    "logistic_regression": [dict(C=1.0, class_weight=None)],
                    "svm": [dict(C=1.0, gamma="scale", class_weight=None)],
                    "random_forest": [dict(max_depth=5, min_samples_leaf=1, class_weight=None)],
                    "cnn": [dict(learning_rate=0.001, dropout=0.2, weighted_loss=False)]})


class FeatureProcessingTests(unittest.TestCase):
    def test_training_statistics_and_alphabetical_correlation(self):
        training = pd.DataFrame({"original_z": [2., 4., 6., 8.],
                                 "original_a": [1., 2., 3., 4.],
                                 "original_missing": [np.nan] * 4,
                                 "original_constant": [9., 9., np.nan, 9.],
                                 "original_noise": [0., 1., 1., 0.]})
        processor = RadiomicsPreprocessor().fit(training, patient_ids=["a", "b", "c", "d"])
        self.assertEqual(processor.retained_features_, ["original_a", "original_noise"])
        held_out = training.iloc[:1].copy()
        held_out["original_a"] = np.inf
        held_out["original_noise"] = 10000
        means_before = processor.means_.copy()
        transformed = processor.transform(held_out)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_array_equal(processor.means_, means_before)
        self.assertAlmostEqual(transformed[0, 0], 0.0)
        self.assertEqual(processor.fit_patient_ids_, ["a", "b", "c", "d"])

    def test_metadata_cannot_enter_a_transform(self):
        with self.assertRaisesRegex(ValueError, "original_"):
            RadiomicsPreprocessor().fit(pd.DataFrame({"mean_malignancy_rating": [1., 5.]}))

    def test_protocol_grids_cannot_be_silently_reduced(self):
        self.assertEqual({k: len(v) for k, v in _grids({}).items()},
                         dict(logistic_regression=12, svm=12, random_forest=12, cnn=12))
        with self.assertRaisesRegex(ValueError, "smoke"):
            _grids(dict(grids={"cnn": [dict(learning_rate=1)]}))

    def test_platt_calibration_is_probability_not_native_score(self):
        scores = np.array([-3., -2., -1., 1., 2., 3.])
        calibrator = SigmoidCalibrator().fit(scores, [0, 0, 0, 1, 1, 1])
        probabilities = calibrator.predict(scores)
        self.assertTrue(np.all(np.diff(probabilities) > 0))
        self.assertTrue(np.all((probabilities > 0) & (probabilities < 1)))
        self.assertAlmostEqual(float(calibrator.predict([0])[0]), 0.5, places=6)

    def test_calibration_converges_on_nearly_separable_training_scores(self):
        rng = np.random.default_rng(42)
        scores = np.concatenate([rng.normal(-.8, .6, 229), rng.normal(.9, .6, 150)])
        labels = np.concatenate([np.zeros(229), np.ones(150)])
        calibrator = SigmoidCalibrator().fit(scores, labels)
        probability = calibrator.predict(scores)
        targets = np.where(labels == 1, 151 / 152, 1 / 231)
        residual = probability - targets
        gradient = [np.dot(residual, scores / max(1., np.abs(scores).max())), residual.sum()]
        self.assertLess(np.max(np.abs(gradient)), 2e-6)
        self.assertTrue(np.all((probability > 0) & (probability < 1)))


class PipelineTests(unittest.TestCase):
    def test_training_partition_and_source_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort, features = fixture(directory)
            with self.assertRaisesRegex(ValueError, "training observations only"):
                fit_selected_model("logistic_regression", cohort, features,
                                   dict(C=1, class_weight=None), smoke_config(), None)
            contaminated = cohort.copy()
            contaminated.loc[42, "patient_id"] = contaminated.loc[0, "patient_id"]
            with self.assertRaisesRegex(ValueError, "Leakage"):
                tune_models(contaminated, features, smoke_config(), Path(directory) / "bad")
            contaminated = cohort.copy()
            contaminated.loc[42, "series_uid"] = contaminated.loc[0, "series_uid"]
            with self.assertRaisesRegex(ValueError, "Leakage"):
                tune_models(contaminated, features, smoke_config(), Path(directory) / "bad")

    def test_fold_and_inner_split_preserve_patient_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort, _ = fixture(directory)
            train = cohort[cohort.partition == "train"].reset_index(drop=True)
            for fit, held in _training_folds(train, 42):
                outer = train.iloc[fit]
                inner_fit, inner_held = _inner_patients(outer, 42)
                self.assertFalse(set(inner_fit.patient_id) & set(inner_held.patient_id))
                self.assertFalse(set(inner_held.patient_id) & set(train.iloc[held].patient_id))
            bad = train.copy()
            bad.loc[1, "patient_id"] = bad.loc[0, "patient_id"]
            with self.assertRaisesRegex(ValueError, "multiple training folds"):
                _training_folds(bad, 42)

    def test_four_model_development_save_reload_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort, features = fixture(directory)
            output = Path(directory) / "experiment"
            config = smoke_config()
            selection = tune_models(cohort, features, config, output)
            self.assertEqual(set(selection["models"]), {"logistic_regression", "svm", "random_forest", "cnn"})
            self.assertNotEqual(selection["selected_traditional"], "cnn")
            rows = pd.read_csv(output / "cv_results.csv")
            self.assertEqual(len(rows), 12)
            self.assertEqual(set(rows.status), {"success"})
            paths = fit_final_models(cohort, features, selection, config, output)
            validation = cohort[cohort.partition == "validation"]
            train_ids = set(cohort[cohort.partition == "train"].patient_id)
            for name, filename in paths.items():
                model = load_model(filename)
                predictions = predict_model(model, validation, features, config)
                self.assertEqual(list(predictions.nodule_id), list(validation.nodule_id))
                self.assertTrue(np.isfinite(predictions[["score", "probability"]]).all().all())
                self.assertEqual(set(model.metadata["fit_patient_ids"]), train_ids)
                np.testing.assert_allclose(predictions["score"],
                                           predict_model(load_model(filename), validation, features, config)["score"],
                                           rtol=1e-6, atol=1e-7)
                if name == "svm":
                    folds = model.metadata["calibration_folds"]
                    self.assertEqual(len(folds), 3)
                    for fold in folds:
                        self.assertFalse(set(fold["fit_patient_ids"]) & set(fold["held_out_patient_ids"]))
                        self.assertTrue(set(fold["fit_patient_ids"]) <= train_ids)
                if name != "cnn":
                    self.assertTrue(all(c.startswith("original_") for c in model.retained_features))
            again = tune_models(cohort, features, config, output)
            self.assertEqual(again["models"], selection["models"])
            changed = copy.deepcopy(config)
            changed["grids"]["logistic_regression"][0]["C"] = 10.0
            with self.assertRaisesRegex(ValueError, "incompatible CV resume"):
                tune_models(cohort, features, changed, output)

    def test_masked_cnn_contract_and_joint_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort, _ = fixture(directory)
            dataset = NoduleDataset(cohort.iloc[:1], augment=True)
            image, _ = dataset[0]
            self.assertEqual(tuple(image.shape), (2, 64, 64))
            self.assertTrue(torch.all(image[0][image[1] == 0] == 0))
            network = CompactCNN(0.2)
            self.assertEqual(tuple(network(image.unsqueeze(0)).shape), (1,))
            bad_image = np.ones((2, 64, 64), dtype=np.float32)
            bad_image[1, 0, 0] = 0
            np.savez(cohort.iloc[0].npz_path, image=bad_image)
            with self.assertRaisesRegex(ValueError, "masked"):
                NoduleDataset(cohort.iloc[:1])


if __name__ == "__main__":
    unittest.main()
