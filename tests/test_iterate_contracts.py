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
