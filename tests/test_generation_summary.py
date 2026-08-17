import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_generation import _result_summary


class TestGenerationResultSummary(unittest.TestCase):
    def test_move_limit_labels_are_draws_not_capture_wins(self):
        out = _result_summary({1: 7, -1: 5, 0: 2, 0.5: 3, -0.5: 4})
        self.assertEqual(out["white_wins"], 7)
        self.assertEqual(out["black_wins"], 5)
        self.assertEqual(out["draws"], 9)
        self.assertEqual(out["neutral_draws"], 2)
        self.assertEqual(out["time_leaning_white"], 3)
        self.assertEqual(out["time_leaning_black"], 4)


if __name__ == "__main__":
    unittest.main()
