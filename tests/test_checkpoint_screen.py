import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import checkpoint_screen as screen  # noqa: E402


class CheckpointScreenTests(unittest.TestCase):
    def test_ranking_maximizes_worst_color_before_aggregate(self):
        balanced = {
            "minimum_color_delta": 0.01,
            "deltas": {"white": 0.01, "black": 0.02, "aggregate": 0.015},
        }
        white_only = {
            "minimum_color_delta": -0.02,
            "deltas": {"white": 0.30, "black": -0.02, "aggregate": 0.14},
        }
        self.assertGreater(screen.rank_key(balanced), screen.rank_key(white_only))

    def test_calibration_is_per_color(self):
        calibration = {
            "a_score": 0.50,
            "a_as_white": {"score": 0.70},
            "a_as_black": {"score": 0.30},
        }
        candidate = {
            "a_score": 0.55,
            "a_as_white": {"score": 0.75},
            "a_as_black": {"score": 0.35},
        }
        result = screen.calibrated_result(candidate, calibration)
        self.assertAlmostEqual(result["minimum_color_delta"], 0.05)
        self.assertTrue(result["passes_both_colors"])

    def test_discovery_deduplicates_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            payload = b"same checkpoint"
            (model_dir / "selected_epoch_002.pt").write_bytes(payload)
            (model_dir / "best_value_net.pt").write_bytes(payload)
            checkpoints = screen.discover_checkpoints(model_dir)
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(checkpoints[0]["name"], "selected_epoch_002")


if __name__ == "__main__":
    unittest.main()
