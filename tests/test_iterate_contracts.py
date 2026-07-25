import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import iterate as it


class GateTests(unittest.TestCase):
    def test_gate_requires_arena_threshold(self):
        ok, reason = it.gate_passes(
            arena_score=0.54, candidate_anchor_score=0.9,
            incumbent_anchor_score=0.5, threshold=0.55, anchor_epsilon=0.05)
        self.assertFalse(ok)
        self.assertIn("arena", reason)

    def test_gate_requires_no_anchor_regression(self):
        ok, reason = it.gate_passes(
            arena_score=0.60, candidate_anchor_score=0.40,
            incumbent_anchor_score=0.70, threshold=0.55, anchor_epsilon=0.05)
        self.assertFalse(ok)
        self.assertIn("anchor", reason)

    def test_gate_allows_small_anchor_dip(self):
        ok, _reason = it.gate_passes(
            arena_score=0.60, candidate_anchor_score=0.67,
            incumbent_anchor_score=0.70, threshold=0.55, anchor_epsilon=0.05)
        self.assertTrue(ok)

    def test_gate_skips_anchor_test_without_baseline(self):
        ok, _reason = it.gate_passes(
            arena_score=0.60, candidate_anchor_score=0.10,
            incumbent_anchor_score=None, threshold=0.55, anchor_epsilon=0.05)
        self.assertTrue(ok)

    def test_gate_rejects_a_one_sided_collapse_the_aggregate_hides(self):
        """The exact shape that reached a playtest twice: a passing mean over
        a collapsed color (router run 2 was White 0.35 / Black 0.80)."""
        ok, reason = it.gate_passes(
            arena_score=0.575, candidate_anchor_score=0.9,
            incumbent_anchor_score=0.5, threshold=0.55, anchor_epsilon=0.05,
            arena_side_scores={"white": 0.35, "black": 0.80})
        self.assertFalse(ok)
        self.assertIn("white", reason)
        self.assertIn("per-side floor", reason)

    def test_gate_allows_balanced_sides_above_the_floor(self):
        ok, _reason = it.gate_passes(
            arena_score=0.60, candidate_anchor_score=0.9,
            incumbent_anchor_score=0.5, threshold=0.55, anchor_epsilon=0.05,
            arena_side_scores={"white": 0.55, "black": 0.65})
        self.assertTrue(ok)

    def test_side_floor_is_not_weakenable_below_the_documented_bar(self):
        """Gates are never relaxed to let a recipe through (repo rule)."""
        self.assertGreaterEqual(it.ARENA_SIDE_FLOOR, 0.40)

    def test_run_arena_reports_per_side_scores(self):
        import inspect
        doc = inspect.getdoc(it.run_arena)
        self.assertIn("white", doc)
        self.assertIn("black", doc)


class HelperTests(unittest.TestCase):
    def test_next_generation_counts_nn_gen_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(it._next_generation(tmp), 1)
            os.makedirs(os.path.join(tmp, "nn_gen3"))
            os.makedirs(os.path.join(tmp, "nn_gen7_blackfocus"))
            os.makedirs(os.path.join(tmp, "heuristic_v4"))
            self.assertEqual(it._next_generation(tmp), 8)

    def test_resolve_project_path_handles_relative_existing_path(self):
        rel = os.path.join("data", "raw", "human_games")
        resolved = it._resolve_project_path(rel)
        self.assertTrue(os.path.isabs(resolved))

    def test_latest_incumbent_anchor_score_uses_last_promotion(self):
        history = {"generations": [
            {"promoted": True, "candidate_anchor_score": 0.5},
            {"promoted": False, "candidate_anchor_score": 0.9},
            {"promoted": True, "candidate_anchor_score": 0.7},
            {"promoted": False, "candidate_anchor_score": 0.2},
        ]}
        self.assertEqual(it._latest_incumbent_anchor_score(history), 0.7)


if __name__ == "__main__":
    unittest.main()
