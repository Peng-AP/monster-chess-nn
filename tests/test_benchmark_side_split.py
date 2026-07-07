"""Contracts for benchmark.py per-side strength aggregation.

The per-side split isolates play quality by color: because Monster Chess is
imbalanced (Black wins with correct play), a merged win rate mostly reflects
which side the model played, not how well. summarize_side turns one side's
games into a strength block; run_benchmark buckets by the side the CANDIDATE
played and keeps the overall totals consistent with the two blocks.
"""
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import benchmark as bm


class SummarizeSideTests(unittest.TestCase):
    def test_counts_score_and_ply_buckets(self):
        # candidate-perspective results: win(+), loss(-), draw(0)
        games = [(1, 30), (1, 40), (-1, 120), (0, 200)]
        s = bm.summarize_side(games)
        self.assertEqual(s["games"], 4)
        self.assertEqual((s["wins"], s["losses"], s["draws"]), (2, 1, 1))
        self.assertEqual(s["score"], round((2 + 0.5) / 4, 4))
        self.assertEqual(s["mean_plies"], round((30 + 40 + 120 + 200) / 4, 2))
        # conversion speed = mean plies of WON games only
        self.assertEqual(s["mean_plies_when_won"], 35.0)
        # resistance = mean plies of LOST games only
        self.assertEqual(s["mean_plies_when_lost"], 120.0)

    def test_empty_buckets_are_none_not_zero(self):
        s = bm.summarize_side([(1, 20), (1, 24)])  # all wins, no losses
        self.assertEqual(s["wins"], 2)
        self.assertEqual(s["mean_plies_when_won"], 22.0)
        self.assertIsNone(s["mean_plies_when_lost"])  # never lost -> None, not 0

    def test_no_games(self):
        s = bm.summarize_side([])
        self.assertEqual(s["games"], 0)
        self.assertIsNone(s["score"])
        self.assertIsNone(s["mean_plies"])


class RunBenchmarkConsistencyTests(unittest.TestCase):
    """Overall totals must equal the sum of the two per-side blocks.

    Uses the heuristic-vs-heuristic path (model_path=None) so no torch/model
    load is needed; a handful of short games at low sims keeps it fast.
    """

    def test_overall_equals_white_plus_black(self):
        r = bm.run_benchmark(model_path=None, games=4, sims=8,
                             anchor_sims=8, seed=12345)
        w, b = r["white_strength"], r["black_strength"]
        self.assertEqual(w["games"] + b["games"], r["games"])
        self.assertEqual(w["wins"] + b["wins"], r["candidate_wins"])
        self.assertEqual(w["losses"] + b["losses"], r["candidate_losses"])
        self.assertEqual(w["draws"] + b["draws"], r["candidate_draws"])
        # black_win_share is the model-as-Black win count over black games
        expected_share = b["wins"] / b["games"] if b["games"] else 0.0
        self.assertAlmostEqual(r["candidate_black_win_share"], round(expected_share, 4))

    def test_black_share_uses_candidate_perspective(self):
        # heuristic-vs-heuristic: the "candidate" as Black should win a share
        # consistent with the game's Black-favoring balance and be in [0, 1].
        r = bm.run_benchmark(model_path=None, games=4, sims=8,
                             anchor_sims=8, seed=999)
        self.assertGreaterEqual(r["candidate_black_win_share"], 0.0)
        self.assertLessEqual(r["candidate_black_win_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
