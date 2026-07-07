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
        self.assertIn("--generations", out)
        self.assertIn("--gate-threshold", out)
        self.assertIn("--arena-games", out)
        self.assertIn("--anchor-epsilon", out)
        self.assertIn("--blackfocus-games", out)
        self.assertIn("--max-generation-age", out)

    def test_train_help_contains_wdl_flags(self):
        code, out = _run_help("train.py")
        self.assertEqual(code, 0)
        self.assertIn("--value-head", out)
        self.assertIn("--wdl-loss-weight", out)
        self.assertIn("--wdl-draw-epsilon", out)
        self.assertIn("--target", out)

    def test_data_processor_help_contains_expected_flags(self):
        code, out = _run_help("data_processor.py")
        self.assertEqual(code, 0)
        self.assertIn("--max-generation-age", out)
        self.assertIn("--min-nonhuman-plies", out)
        self.assertIn("--exclude-human-games", out)

    def test_data_generation_help_contains_temperature_flags(self):
        code, out = _run_help("data_generation.py")
        self.assertEqual(code, 0)
        self.assertIn("--temperature-high", out)
        self.assertIn("--temperature-low", out)
        self.assertIn("--temperature-moves", out)

if __name__ == "__main__":
    unittest.main()
