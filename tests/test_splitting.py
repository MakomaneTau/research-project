import unittest

import pandas as pd

from lidc_experiment.splitting import assign_partitions, assert_independent, nested_patient_subsets


class PartitionTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([dict(patient_id=f"p{i:03}", nodule_id=f"p{i:03}_n{j}",
                                  series_uid=f"s{i}", label=i % 2)
                             for i in range(100) for j in range(1 + i % 3)])

    def test_reproducible_grouped_allocation(self):
        a, b = assign_partitions(self.fixture()), assign_partitions(self.fixture())
        pd.testing.assert_frame_equal(a, b)
        assert_independent(a)
        self.assertEqual(a.groupby("patient_id").partition.nunique().max(), 1)

    def test_overlap_is_hard_failure(self):
        data = assign_partitions(self.fixture())
        row = data.iloc[0].copy()
        row.nodule_id = "other"
        row.partition = "test" if row.partition != "test" else "train"
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            assert_independent(pd.concat([data, row.to_frame().T], ignore_index=True))

    def test_learning_subsets_are_nested(self):
        data = assign_partitions(self.fixture())
        previous = set()
        for subset in nested_patient_subsets(data[data.partition == "train"]).values():
            patients = set(subset.patient_id)
            self.assertTrue(previous <= patients)
            previous = patients

    def test_shared_source_instance_is_rejected(self):
        data = assign_partitions(self.fixture())
        data["source_image_uids"] = '["shared-sop-instance"]'
        with self.assertRaisesRegex(ValueError, "Source image overlap"):
            assert_independent(data)


if __name__ == "__main__":
    unittest.main()
