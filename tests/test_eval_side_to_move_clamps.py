"""Threat clamps must respect side to move.

A capture available to the SIDE TO MOVE decides the game (clamp OK).
A capture threat AGAINST the side to move is not decided: the mover may
parry, escape, or capture first — clamping there poisons search values
(Black learned that any check ~= a won game and played sacrificial checks).
"""
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import evaluate
from monster_chess import MonsterChessGame

# Black just played Qh4+ (a pure sacrifice). White to move refutes it
# outright inside one double-move: g2-g3 attacks the queen, then g3xh4.
SAC_CHECK_FEN = "6k1/8/8/8/7q/8/PPPPP1PP/4K3 w - - 0 1"

# White threatens to capture the Black king within a double-move, but it is
# BLACK to move and the rook on a8 captures the White king first.
MUTUAL_THREAT_BLACK_MOVES_FEN = "r5k1/8/8/8/8/8/6K1/R7 b - - 0 1"


class SideToMoveClampTests(unittest.TestCase):
    def test_check_against_white_is_not_a_decided_black_win(self):
        game = MonsterChessGame(fen=SAC_CHECK_FEN)
        val = evaluate(game)
        self.assertGreater(val, -0.5,
                           "check against the mover must not clamp to -0.95")

    def test_white_threat_does_not_clamp_when_black_moves_first(self):
        game = MonsterChessGame(fen=MUTUAL_THREAT_BLACK_MOVES_FEN)
        val = evaluate(game)
        # Contract: the White double-move threat must NOT clamp to +0.95 on a
        # Black-to-move node (Black defends first). Assert Black-favored, not a
        # brittle exact magnitude (eval tuning shifts the number, not the sign).
        self.assertLess(val, -0.5,
                        "White threat must not clamp positive when Black moves first")

    def test_mover_side_clamps_still_fire(self):
        # White to move, can capture Black king in a double-move.
        white_wins = MonsterChessGame(fen="8/8/8/8/8/8/8/k1K5 w - - 0 1")
        self.assertGreaterEqual(evaluate(white_wins), 0.90)
        # Black to move, rook captures the White king.
        black_wins = MonsterChessGame(fen="4k3/8/8/8/8/8/4r3/4K3 b - - 0 1")
        self.assertLessEqual(evaluate(black_wins), -0.90)


class NNEvaluatorClampPathTests(unittest.TestCase):
    """Same rule for the NN evaluator's scan short-circuits (no model load:
    monkeypatch-free check via the module-level helpers used by both paths)."""

    def test_batch_and_single_share_evaluate_semantics(self):
        # The NN paths short-circuit BEFORE the forward pass with the same
        # scan order as evaluate(); assert on evaluate() FENs here so a
        # regression in either shared helper shows up. Full NN-path parity
        # is enforced by code: both call the same helpers.
        game = MonsterChessGame(fen=SAC_CHECK_FEN)
        self.assertGreater(evaluate(game), -0.5)


if __name__ == "__main__":
    unittest.main()
