import unittest

import numpy as np

from lidc_experiment.compat import _LegacyNumpy


class CompatibilityTests(unittest.TestCase):
    def test_builtin_aliases_preserve_dtype_semantics_without_global_changes(self):
        adapter = _LegacyNumpy()
        np.testing.assert_array_equal(adapter.asarray([1, 2], dtype=adapter.int), np.array([1, 2], dtype=int))
        self.assertIs(adapter.mean, np.mean)
        self.assertNotIn("int", np.__dict__)


if __name__ == "__main__":
    unittest.main()
