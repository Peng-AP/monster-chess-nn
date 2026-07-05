"""play.py human-input safety probe: _winning_or_safe_pair_exists.

The old _safe_pairs_gen yielded the first legal pair WITHOUT testing safety,
so "safe pair exists" was always True — in forced-blunder positions every
human move was rejected forever (softlock).
"""
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monster_chess import MonsterChessGame
from play import _winning_or_safe_pair_exists


class WinningOrSafePairExistsTests(unittest.TestCase):
    def test_true_in_quiet_position(self):
        game = MonsterChessGame()  # opening: plenty of safe pairs
        self.assertTrue(_winning_or_safe_pair_exists(game))

    def test_true_when_only_option_is_the_winning_capture(self):
        # Kings adjacent: Kb1xa2-ish — White captures the Black king; that
        # counts as a "safe" option even if every quiet pair hangs the king.
        game = MonsterChessGame(fen="8/8/8/8/8/8/k7/1K6 w - - 0 1")
        self.assertTrue(_winning_or_safe_pair_exists(game))

    def test_false_when_every_pair_hangs_the_king(self):
        # WK a1 boxed by queens b3+c3 (BK far away): every reachable final
        # square is attacked and no Black king capture exists -> forced
        # blunder, so NO winning-or-safe pair exists.
        game = MonsterChessGame(fen="7k/8/8/8/8/1qq5/8/K7 w - - 0 1")
        self.assertFalse(_winning_or_safe_pair_exists(game))


if __name__ == "__main__":
    unittest.main()
