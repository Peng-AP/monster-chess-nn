import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from search_sweep import (arm_search_kwargs, parse_arm, rank_arms,
                          summarize_arm)


class TestBlackFirstSearchSweep(unittest.TestCase):
    def test_arm_changes_only_candidate_search(self):
        kwargs = arm_search_kwargs({"policy_temperature": 1.25})
        self.assertEqual(kwargs["policy_temperature_a"], 1.25)
        self.assertEqual(kwargs["policy_temperature_b"], 1.0)
        self.assertEqual(kwargs["c_puct_a"], kwargs["c_puct_b"])
        self.assertEqual(kwargs["fpu_reduction_a"],
                         kwargs["fpu_reduction_b"])

    def test_ranking_is_black_first(self):
        arms = [
            {"id": "better-all", "black_score": 0.5,
             "aggregate_score": 0.9, "white_score": 1.0},
            {"id": "better-black", "black_score": 0.6,
             "aggregate_score": 0.6, "white_score": 0.6},
        ]
        self.assertEqual(rank_arms(arms)[0]["id"], "better-black")

    def test_screen_keeps_white_and_aggregate_safeguards(self):
        match = {
            "a_as_black": {"score": 0.7},
            "a_as_white": {"score": 0.3},
            "a_score": 0.6,
        }
        self.assertFalse(summarize_arm(
            "c_puct", 1.2, match)["screen_pass"])

    def test_black_uses_the_existing_inclusive_per_side_floor(self):
        match = {
            "a_as_black": {"score": 0.4},
            "a_as_white": {"score": 0.7},
            "a_score": 0.55,
        }
        self.assertTrue(summarize_arm(
            "c_puct", 1.2, match)["screen_pass"])

    def test_parser_rejects_unknown_or_nonpositive_temperature(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arm("unknown=1")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arm("policy_temperature=0")


if __name__ == "__main__":
    unittest.main()
