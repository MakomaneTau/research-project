"""Failure-path checks for the separate audit tools; no experiment data needed."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def load_script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditExitStatusTests(unittest.TestCase):
    def run_audit(self, passed):
        module = load_script('verify_results_evidence')
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            module,
            OUT=Path(directory),
            preservation=lambda: None,
            cohort_audit=lambda: (None, None),
            training_audit=lambda *_: ({}, {}),
            predictive_audit=lambda *_: None,
            secondary_audit=lambda *_: None,
            resource_audit=lambda: None,
            prototype_audit=lambda *_: module.check('prediction parity', passed),
            finish=lambda: None,
        ):
            module.main()

    def test_a_completed_audit_with_a_mismatch_fails_the_command(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_audit(False)
        self.assertEqual(caught.exception.code, 1)

    def test_a_completed_passing_audit_returns_normally(self):
        self.run_audit(True)


class PreservationFailureTests(unittest.TestCase):
    def test_a_missing_original_directory_cannot_produce_an_archive(self):
        module = load_script('preserve_results_evidence')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.multiple(module, SOURCE=root/'missing', OUT=root/'review'):
                with self.assertRaisesRegex(SystemExit, 'unavailable'):
                    module.main()
            self.assertFalse((root/'review').exists())

    def test_an_empty_original_directory_cannot_produce_an_archive(self):
        module = load_script('preserve_results_evidence')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root/'original'
            original.mkdir()
            with patch.multiple(module, SOURCE=original, OUT=root/'review'):
                with self.assertRaisesRegex(SystemExit, 'empty'):
                    module.main()
            self.assertFalse((root/'review/preservation/chapter3_original.zip').exists())


if __name__ == '__main__':
    unittest.main()
