import unittest
from unittest.mock import Mock

import numpy as np

from lidc_experiment.dataset import DataQualityError
from lidc_experiment.features import extract_arrays


class FeatureFailureTests(unittest.TestCase):
    def fixture(self):
        hu = np.zeros((64, 64), dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30:34, 30:34] = 1
        return hu, mask

    def test_known_degenerate_roi_is_recordable_data_failure(self):
        extractor = Mock()
        extractor.execute.side_effect = ValueError("mask only contains 1 segmented voxel! Cannot extract features for a single voxel.")
        with self.assertRaises(DataQualityError):
            extract_arrays(*self.fixture(), extractor)

    def test_software_failure_cannot_be_relabelled_as_case_exclusion(self):
        extractor = Mock()
        extractor.execute.side_effect = ValueError("Unexpected parameter configuration")
        with self.assertRaises(ValueError) as raised:
            extract_arrays(*self.fixture(), extractor)
        self.assertNotIsInstance(raised.exception, DataQualityError)

    def test_unplanned_feature_family_is_a_hard_failure(self):
        extractor = Mock()
        extractor.execute.return_value = {"original_shape2D_Area": 16.0, "original_gldm_Unplanned": 1.0}
        with self.assertRaisesRegex(RuntimeError, "Unexpected feature family"):
            extract_arrays(*self.fixture(), extractor)


if __name__ == "__main__":
    unittest.main()
