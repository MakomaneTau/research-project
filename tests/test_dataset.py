"""Scientific invariants for matched image preparation (no real data needed)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lidc_experiment.dataset import (DataQualityError, SourceUnavailableError,
                                     _prepare_record, _scan_dir, consensus_mask,
                                     crop_slice, select_slice, suspicion_label)


class LabelTests(unittest.TestCase):
    def test_boundaries_and_indeterminate(self):
        self.assertEqual(suspicion_label([1, 3]), 0)
        self.assertEqual(suspicion_label([3, 5]), 1)
        self.assertEqual(suspicion_label([2, 3]), -1)
        self.assertEqual(suspicion_label([1, 1, 5], "median"), 0)
        with self.assertRaises(DataQualityError):
            suspicion_label([float("nan"), 2])

    def test_source_access_failure_is_not_a_scientific_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            scan = SimpleNamespace(patient_id="synthetic", get_path_to_dicom_files=lambda: (_ for _ in ()).throw(RuntimeError("offline")))
            with self.assertRaises(SourceUnavailableError):
                _scan_dir(scan, {"data_root": directory})


class ImageTests(unittest.TestCase):
    def test_half_and_strict_consensus(self):
        masks = np.zeros((2, 1, 3, 3), dtype=bool)
        masks[0, 0, 1, 1] = True
        self.assertTrue(consensus_mask(masks)[0, 1, 1])
        self.assertFalse(consensus_mask(masks, "strict").any())

    def test_max_area_tie_uses_centroid_then_index(self):
        mask = np.zeros((7, 5, 5), dtype=bool)
        mask[1, :2, :2] = True
        mask[5, :2, :2] = True
        self.assertEqual(select_slice(mask), 1)
        mask[4, 0, 0] = True
        self.assertEqual(select_slice(mask), 5)

    def test_crop_padding_scaling_and_mask_information(self):
        hu = np.full((10, 10), -300, dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)
        mask[:2, :2] = True
        crop, roi, channels, _ = crop_slice(hu, mask, 8)
        self.assertEqual(crop.shape, (8, 8))
        self.assertEqual(roi.sum(), 4)
        self.assertTrue(np.all(channels[0, roi == 0] == 0))
        np.testing.assert_array_equal(channels[1], roi)
        self.assertTrue(np.all(channels[0, roi != 0] == 0.5))
        self.assertIn(-1000, crop)

    def test_truncated_lesion_is_rejected(self):
        with self.assertRaises(DataQualityError):
            crop_slice(np.zeros((20, 20)), np.ones((20, 20)), 8)

    def test_empty_mask_is_rejected(self):
        with self.assertRaises(DataQualityError):
            select_slice(np.zeros((2, 3, 3)))

    @unittest.skipUnless(importlib.util.find_spec("SimpleITK"), "SimpleITK not installed")
    def test_local_resampling_matches_full_physical_reference(self):
        import SimpleITK as sitk
        # Nonzero origin, unequal spacing, and a rotated coordinate frame catch
        # the zero-origin/identity-direction bugs common in array-only pipelines.
        z, y, x = np.indices((5, 11, 13))
        raw = sitk.GetImageFromArray((20 * z + 3 * y + x).astype(np.float32))
        raw.SetSpacing((0.5, 0.7, 2.0))
        raw.SetOrigin((20.0, 30.0, 40.0))
        raw.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        values = np.zeros((5, 11, 13), dtype=np.uint8)
        values[2:4, 4:7, 4:9] = 1
        reader = sitk.GetImageFromArray(values)
        reader.CopyInformation(raw)
        config = {"preprocessing": {"crop_size": 8}}
        record = {"annotation_ids": "[1, 2]", "nodule_id": "synthetic", "source_fingerprint": "synthetic"}
        scan = SimpleNamespace(annotations=[SimpleNamespace(id=1, contours=[]), SimpleNamespace(id=2, contours=[])], slice_zvals=np.arange(5) * 2)
        geometry = {"source_image_uids": "[]"}
        with tempfile.TemporaryDirectory() as temp, patch("lidc_experiment.dataset._annotation_masks", return_value=[reader, reader]):
            path = Path(temp) / "test.npz"
            prepared = _prepare_record(record, config, path, scan, raw, geometry)
            full_reference = sitk.Image([7, 8, 9], sitk.sitkFloat32)
            full_reference.SetSpacing((1.0, 1.0, 1.0))
            full_reference.SetOrigin(raw.GetOrigin())
            full_reference.SetDirection(raw.GetDirection())
            full_hu = sitk.GetArrayFromImage(sitk.Resample(raw, full_reference, sitk.Transform(), sitk.sitkLinear, -1000, sitk.sitkFloat32))
            full_mask = sitk.GetArrayFromImage(sitk.Resample(reader, full_reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
            selected = select_slice(full_mask)
            expected_hu, expected_mask, expected_image, _ = crop_slice(full_hu[selected], full_mask[selected], 8)
            self.assertEqual(prepared["selected_slice_index"], selected)
            with np.load(path) as result:
                np.testing.assert_allclose(result["hu"], expected_hu)
                np.testing.assert_array_equal(result["mask"], expected_mask)
                np.testing.assert_allclose(result["image"], expected_image)
                metadata = json.loads(str(result["metadata"].item()))
                self.assertEqual(metadata["spacing_mm"], 1.0)


if __name__ == "__main__":
    unittest.main()
