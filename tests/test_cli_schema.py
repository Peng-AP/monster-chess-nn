import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run_help(script_name):
    result = subprocess.run(
        [sys.executable, str(SRC / script_name), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


class CliSchemaSmoke(unittest.TestCase):
    def test_iterate_help_contains_expected_flags(self):
        code, out = _run_help("iterate.py")
        self.assertEqual(code, 0)
        self.assertIn("--black-focus-gate-threshold", out)
        self.assertIn("--human-seed-dir", out)
        self.assertIn("--value-head", out)
        self.assertIn("--wdl-loss-weight", out)

    def test_train_help_contains_wdl_flags(self):
        code, out = _run_help("train.py")
        self.assertEqual(code, 0)
        self.assertIn("--value-head", out)
        self.assertIn("--wdl-loss-weight", out)
        self.assertIn("--wdl-draw-epsilon", out)

    def test_data_processor_help_contains_expected_flags(self):
        code, out = _run_help("data_processor.py")
        self.assertEqual(code, 0)
        self.assertIn("--use-source-quotas", out)
        self.assertIn("--human-target-mcts-lambda", out)

    def test_gate_sweep_help_contains_new_calibration_flags(self):
        code, out = _run_help("gate_sweep.py")
        self.assertEqual(code, 0)
        self.assertIn("--recommend-k", out)
        self.assertIn("--target-accept-rate", out)
        self.assertIn("--prefer-strict", out)

    def test_iterate_presets_help_and_dry_run(self):
        code, out = _run_help("iterate_presets.py")
        self.assertEqual(code, 0)
        self.assertIn("--preset", out)
        self.assertIn("--show-presets", out)

        result = subprocess.run(
            [sys.executable, str(SRC / "iterate_presets.py"), "--preset", "smoke", "--dry-run"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        combined = (result.stdout or "") + (result.stderr or "")
        self.assertIn("iterate.py", combined)
        self.assertIn("Preset: smoke", combined)


if __name__ == "__main__":
    unittest.main()
