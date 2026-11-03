"""Prevent an incomplete extraction run from becoming the primary cohort."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lidc_experiment.utils import write_json
from lidc_experiment.workflow import freeze_cohort


class CohortCompletionTests(unittest.TestCase):
    def fixture(self, root):
        (root / "data").mkdir()
        (root / "features").mkdir()
        rows = pd.DataFrame([dict(nodule_id=f"n{i:03}", patient_id=f"p{i:03}", series_uid=f"s{i:03}",
                                  label=i % 2, diameter_mm=8.0, source_fingerprint=f"hash{i}") for i in range(100)])
        rows.to_csv(root / "data" / "all_nodules.csv", index=False)
        rows.to_csv(root / "data" / "audit_nodules.csv", index=False)
        pd.DataFrame(columns=["stage", "reason", "nodule_id"]).to_csv(root / "data" / "exclusions.csv", index=False)
        pd.DataFrame(columns=["nodule_id", "reason"]).to_csv(root / "features" / "feature_errors.csv", index=False)
        features = pd.DataFrame({"nodule_id": rows.nodule_id, "original_firstorder_Mean": range(100)})
        features.to_csv(root / "features" / "radiomics_features.csv", index=False)
        write_json(root / "data" / "preparation_summary.json", {"prepared_nodules": 100})
        write_json(root / "data" / "audit_summary.json", {"source_patients": 100, "source_scans": 100})
        return features

    def test_partial_features_cannot_silently_reduce_primary_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = self.fixture(root)
            features.iloc[:-1].to_csv(root / "features" / "radiomics_features.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Feature extraction is incomplete"):
                freeze_cohort(root)

    def test_completed_features_produce_reconciled_patient_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            cohort = freeze_cohort(root)
            self.assertEqual(len(cohort), 100)
            self.assertTrue((root / "cohort_flow.csv").exists())
            self.assertEqual(cohort.groupby("patient_id").partition.nunique().max(), 1)


if __name__ == "__main__":
    unittest.main()
