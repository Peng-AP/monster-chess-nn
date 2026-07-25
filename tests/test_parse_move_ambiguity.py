"""play.parse_move must refuse ambiguous SAN instead of guessing.

The manual-SAN fallback drops file/rank disambiguators, so "Qa5" matched both
Qc7-a5 and Qa1-a5 and the loop returned whichever the unordered legal set
yielded first.  Replaying playstrategy game D3p2sAyu it picked the wrong queen;
the game then diverged silently and only died two plies later on a move that
had become impossible.  Two queens bearing on one square is a promotion-endgame
position, i.e. exactly what this project plays.
"""
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monster_chess import MonsterChessGame
from play import parse_move, parse_move_candidates, _ambiguity_note

# D3p2sAyu after 17. c6,Kc5: Black queens on c7 and a1 both reach a5.
TWO_QUEENS = "2bkr3/1pq2p2/r1P5/2K1p3/8/8/8/q7 b - - 1 17"


class ParseMoveAmbiguityTests(unittest.TestCase):
    def setUp(self):
        self.game = MonsterChessGame(fen=TWO_QUEENS)
        self.legal = self.game.get_search_actions()

    def _parse(self, text):
        return parse_move(text, self.game.board, legal_set=self.legal)

    def test_ambiguous_san_returns_none(self):
        self.assertEqual(
            [m.uci() for m in parse_move_candidates(
                "Qa5+", self.game.board, legal_set=self.legal)],
            ["a1a5", "c7a5"])
        self.assertIsNone(self._parse("Qa5+"))

    def test_disambiguated_san_still_resolves(self):
        self.assertEqual(self._parse("Qca5+").uci(), "c7a5")
        self.assertEqual(self._parse("Q1a5+").uci(), "a1a5")

    def test_uci_still_resolves(self):
        self.assertEqual(self._parse("a1a5").uci(), "a1a5")

    def test_ambiguity_note_names_both_moves(self):
        note = _ambiguity_note("Qa5+", self.game.board, self.legal)
        self.assertIn("Qaa5", note)   # board.san() disambiguates by file
        self.assertIn("Qca5", note)
        self.assertIsNone(_ambiguity_note("Qca5+", self.game.board, self.legal))

    def test_unambiguous_monster_only_san_still_resolves(self):
        # White's first half-move may step into check, which parse_san rejects;
        # the fallback must keep handling it (only one king, never ambiguous).
        game = MonsterChessGame(fen="4k3/8/8/8/8/8/2b5/4K3 w - - 0 1")
        legal = game.get_search_actions()
        self.assertEqual(parse_move("Kd2", game.board, legal_set=legal).uci(),
                         "e1d2")


if __name__ == "__main__":
    unittest.main()
