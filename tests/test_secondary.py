import unittest
import pandas as pd
from lidc_experiment.secondary import median_development_cohort


class AlternativeLabelTests(unittest.TestCase):
    def test_new_nodules_can_enter_only_existing_development_patients(self):
        primary = pd.DataFrame([
            dict(patient_id="train", nodule_id="a", label=0, partition="train", fold=0),
            dict(patient_id="validation", nodule_id="b", label=1, partition="validation", fold=-1),
            dict(patient_id="test", nodule_id="c", label=1, partition="test", fold=-1)])
        all_labels = pd.DataFrame([
            dict(patient_id="train", nodule_id="a", label=0, rating_median=1),
            dict(patient_id="train", nodule_id="d", label=-1, rating_median=4),
            dict(patient_id="validation", nodule_id="b", label=1, rating_median=4),
            dict(patient_id="test", nodule_id="c", label=1, rating_median=5),
            dict(patient_id="unallocated", nodule_id="e", label=-1, rating_median=4)])
        result = median_development_cohort(all_labels, primary)
        self.assertEqual(set(result.nodule_id), {"a", "b", "d"})
        self.assertEqual(result.set_index("nodule_id").loc["d", "fold"], 0)
        self.assertEqual(result.set_index("nodule_id").loc["d", "label"], 1)


if __name__ == "__main__":
    unittest.main()
