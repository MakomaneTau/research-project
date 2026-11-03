"""Scientific checks for thresholds, pairing and patient-cluster resampling."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from lidc_experiment.evaluation import (
    choose_threshold, classification_metrics, cluster_bootstrap_indices,
    evaluate_predictions, metrics_table, paired_patient_bootstrap, validate_predictions,
)
from lidc_experiment.explanations import grad_cam, select_error_panel


def predictions():
    records = []
    for model in ("logistic_regression", "svm", "random_forest", "cnn"):
        for index, (patient, label, probability) in enumerate([
            ("p0", 0, 0.1), ("p0", 1, 0.8), ("p1", 1, 0.9),
            ("p2", 0, 0.4), ("p3", 1, 0.3), ("p4", 0, 0.2),
        ]):
            records.append({"patient_id": patient, "nodule_id": f"n{index}", "label": label,
                            "model": model, "score": probability, "probability": probability,
                            "diameter_mm": [4, 8, 12, 6, 11, 9][index], "rating_range": index % 3,
                            "partition": "test"})
    return pd.DataFrame(records)


class EvaluationTests(unittest.TestCase):
    def test_threshold_ties_and_boundaries(self):
        self.assertEqual(choose_threshold([1, 1, 0, 0], [.9, .4, .4, .1], .9), .4)
        self.assertEqual(choose_threshold([1, 1, 0], [.9, .4, .8], .5), .9)
        with self.assertRaises(ValueError):
            choose_threshold([0, 0], [.1, .2])
        with self.assertRaises(ValueError):
            choose_threshold([0, 1], [.1, np.nan])

    def test_hand_calculated_metrics(self):
        result = classification_metrics([0, 0, 1, 1], [.1, .8, .9, .4], [.1, .8, .9, .4], .5)
        for key in ("tp", "tn", "fp", "fn"):
            self.assertEqual(result[key], 1)
        for key in ("accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy"):
            self.assertAlmostEqual(result[key], .5)
        self.assertAlmostEqual(result["roc_auc"], .75)
        self.assertAlmostEqual(result["brier_score"], (.01 + .64 + .01 + .36) / 4)
        undefined = classification_metrics([0, 1], [.1, .2], [.1, .2], .5)
        self.assertTrue(np.isnan(undefined["precision"]))
        self.assertEqual(undefined["f1"], 0)

    def test_patient_draw_retains_multiplicity(self):
        class FixedDraw:
            def choice(self, patients, count, replace):
                return np.array(["a", "a"])
        indices = cluster_bootstrap_indices(["a", "a", "b"], FixedDraw())
        np.testing.assert_array_equal(indices, [0, 1, 0, 1])

    def test_missing_predictions_and_label_conflicts_fail(self):
        data = predictions()
        validate_predictions(data)
        with self.assertRaisesRegex(ValueError, "common prediction cohort"):
            validate_predictions(data.loc[~((data.model == "cnn") & (data.nodule_id == "n0"))])
        broken = data.copy()
        broken.loc[(broken.model == "cnn") & (broken.nodule_id == "n0"), "label"] = 1
        with self.assertRaises(ValueError):
            validate_predictions(broken)
        duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_predictions(duplicated)

    def test_identical_models_have_zero_paired_auc_difference(self):
        data = predictions()
        thresholds = {name: .5 for name in data.model.unique()}
        intervals, primary = paired_patient_bootstrap(data, thresholds, "logistic_regression", replicates=30, max_attempts=100)
        self.assertEqual(primary["valid_replicates"], 30)
        self.assertTrue(primary["reliable"])
        self.assertEqual(primary["interpretation"], "inconclusive")
        self.assertEqual(primary["differences"]["roc_auc"]["lower_95"], 0)
        self.assertEqual(primary["differences"]["roc_auc"]["upper_95"], 0)
        self.assertEqual(len(intervals), 36)

    def test_patient_weighting_gives_each_patient_total_weight_one(self):
        data = predictions()
        result = metrics_table(data, {name: .5 for name in data.model.unique()}, patient_weighted=True)
        row = result.iloc[0]
        self.assertEqual(row.tp + row.tn + row.fp + row.fn, 5)
        self.assertEqual(row.n_nodules, 6)

    def test_saved_report_is_prediction_derived(self):
        data = predictions()
        thresholds = {name: .5 for name in data.model.unique()}
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_predictions(data, thresholds, {"evaluation": {"bootstrap_replicates": 10, "bootstrap_max_attempts": 100}}, directory, "logistic_regression")
            self.assertEqual(len(result["metrics"]), 4)
            expected = ["metrics.csv", "metrics_probability_050.csv", "metrics_patient_weighted.csv",
                        "metric_intervals.csv", "subgroup_metrics.csv", "calibration_bins.csv",
                        "figures/roc.png", "figures/precision_recall.png", "figures/calibration.png"]
            for file in expected:
                self.assertTrue((Path(directory) / file).is_file(), file)
            primary = json.loads((Path(directory) / "primary_comparison.json").read_text())
            self.assertEqual(primary["valid_replicates"], 10)

    def test_error_panel_is_shared_and_deterministic(self):
        data = predictions()
        panel = select_error_panel(data, {name: .5 for name in data.model.unique()}, "logistic_regression", limit=1)
        self.assertEqual(set(panel.nodule_id), {"n0", "n1", "n4"})
        self.assertEqual(set(panel.groupby("nodule_id").size()), {4})
        pd.testing.assert_frame_equal(panel, select_error_panel(data.sample(frac=1, random_state=3),
                                                               {name: .5 for name in data.model.unique()}, "logistic_regression", limit=1))

    def test_grad_cam_targets_output_and_restores_network_mode(self):
        import torch
        network = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.ReLU(),
                                      torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 1))
        network.train()
        cam, logit = grad_cam(network, np.random.default_rng(1).random((2, 16, 16), dtype=np.float32))
        self.assertEqual(cam.shape, (16, 16))
        self.assertTrue(np.isfinite(cam).all())
        self.assertTrue(np.isfinite(logit))
        self.assertTrue(network.training)
        self.assertGreaterEqual(cam.min(), 0)
        self.assertLessEqual(cam.max(), 1)


if __name__ == "__main__":
    unittest.main()
