import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "tools"))

import confirm_candidates as confirm  # noqa: E402


def match(black, white, aggregate):
    return {
        "a_as_black": {"score": black},
        "a_as_white": {"score": white},
        "a_score": aggregate,
    }


class ConfirmCandidatesTests(unittest.TestCase):
    def test_requires_positive_delta_for_each_color(self):
        baseline = match(0.2, 0.8, 0.5)
        self.assertTrue(confirm.calibrated_result(
            match(0.25, 0.85, 0.55), baseline)["passes_both_colors"])
        self.assertFalse(confirm.calibrated_result(
            match(0.25, 0.8, 0.525), baseline)["passes_both_colors"])
        self.assertFalse(confirm.calibrated_result(
            match(0.15, 0.85, 0.5), baseline)["passes_both_colors"])

    def test_minimum_color_delta_is_the_robust_ranking_key(self):
        result = confirm.calibrated_result(
            match(0.25, 0.82, 0.535), match(0.2, 0.8, 0.5))
        self.assertAlmostEqual(result["minimum_color_delta"], 0.02)

    def test_candidate_spec_uses_repo_relative_paths(self):
        name, path = confirm.parse_candidate("short=models/example.pt")
        self.assertEqual(name, "short")
        self.assertEqual(path, ROOT / "models/example.pt")


if __name__ == "__main__":
    unittest.main()
