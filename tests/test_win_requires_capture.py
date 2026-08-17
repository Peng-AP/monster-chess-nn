"""Only a king capture is a win. A win by time is a draw.

Owner, 2026-08-03: *"A win by time shouldn't be counted the same as win by
capturing the king."*

The old rule scored `result > 0` as a win, and a game reaching MAX_GAME_TURNS
is relabelled +-0.5 by heuristic sign — so "ahead when the clock ran out" earned
identical gate credit to actually finishing. That is not a scoring nicety: it
rewarded precisely the failure the owner reported at the board, and it inflated
a real candidate. `v20w`'s post-promotion conversion against v19 read 0.28 ->
0.46 while its true king-capture rate went 0.15 -> 0.13; the whole apparent gain
was +-0.5 relabels (13 -> 33).

The +-0.5 *training label* is untouched — it carries gradient and is a
deliberate design (CONTEXT law 3). This is only about what the gate calls a win.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark import summarize_side  # noqa: E402


def games(*results):
    return [(r, 100) for r in results]


class TestOnlyCapturesWin(unittest.TestCase):
    def test_king_capture_is_a_win(self):
        s = summarize_side(games(1, 1))
        self.assertEqual((s["wins"], s["draws"], s["losses"]), (2, 0, 0))
        self.assertEqual(s["score"], 1.0)

    def test_ahead_at_the_move_limit_is_a_draw(self):
        s = summarize_side(games(0.5, 0.5))
        self.assertEqual((s["wins"], s["draws"], s["losses"]), (0, 2, 0))
        self.assertEqual(s["score"], 0.5)

    def test_behind_at_the_move_limit_is_also_a_draw(self):
        # Symmetric: the rule cannot favour one side.
        s = summarize_side(games(-0.5, -0.5))
        self.assertEqual((s["wins"], s["draws"], s["losses"]), (0, 2, 0))
        self.assertEqual(s["score"], 0.5)

    def test_true_draw_still_a_draw(self):
        s = summarize_side(games(0, 0))
        self.assertEqual(s["draws"], 2)
        self.assertEqual(s["score"], 0.5)

    def test_losing_by_capture_is_a_loss(self):
        s = summarize_side(games(-1, -1))
        self.assertEqual((s["wins"], s["draws"], s["losses"]), (0, 0, 2))
        self.assertEqual(s["score"], 0.0)


class TestShufflingCannotScoreLikeFinishing(unittest.TestCase):
    def test_the_v20w_pattern_no_longer_scores_as_a_win(self):
        # A model that reaches winning positions and never ends them.
        shuffler = summarize_side(games(*([0.5] * 10)))
        finisher = summarize_side(games(*([1] * 10)))
        self.assertEqual(shuffler["score"], 0.5)
        self.assertEqual(finisher["score"], 1.0)
        self.assertLess(shuffler["score"], finisher["score"])

    def test_time_leaning_games_stay_visible_in_the_summary(self):
        # Folding them into "draws" would hide the very thing being measured.
        s = summarize_side(games(0.5, 0.5, -0.5, 0))
        self.assertEqual(s["draws"], 4)
        self.assertEqual(s["time_leaning_wins"], 2)
        self.assertEqual(s["time_leaning_losses"], 1)


class TestScoreArithmetic(unittest.TestCase):
    def test_mixed_bag(self):
        # 2 captures, 1 capture-loss, 2 ahead-at-cap, 1 true draw.
        s = summarize_side(games(1, 1, -1, 0.5, 0.5, 0))
        self.assertEqual((s["wins"], s["draws"], s["losses"]), (2, 3, 1))
        self.assertEqual(s["score"], round((2 + 0.5 * 3) / 6, 4))

    def test_plies_when_won_counts_only_real_wins(self):
        s = summarize_side([(1, 40), (0.5, 225)])
        self.assertEqual(s["mean_plies_when_won"], 40)


if __name__ == "__main__":
    unittest.main()
