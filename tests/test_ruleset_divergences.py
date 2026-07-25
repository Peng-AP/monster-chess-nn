"""Two rules our engine settles differently from playstrategy.org.

Both surfaced replaying their 2,971-game human corpus (2026-07-25); both were
put to the owner and both were decided in favour of the behavior already
implemented.  These tests exist so the questions stop being re-litigated: the
divergence is a decision, not an oversight.  Anyone reversing one of these is
changing the game the engine plays and should say so out loud.

1. En passant is conferred only by the LAST move of White's turn.  python-chess
   holds one ep square and each push recomputes it, so `e4` then `f4` leaves
   only `f3` capturable.  playstrategy allows either.  A faithful copy of their
   rule needs up to two ep squares at once, which no FEN can carry, so it means
   hand-rolled Black move generation.  Their game `BDarxSwF` is the only place
   in 2,971 games where the difference is visible.

2. White may not END its turn with its own king attacked (the forced-blunder
   exception, where every option hangs, is unchanged and lives in
   _get_white_actions).  playstrategy accepted `12. Ke6,f5#` in game
   `gaBbbLhv` with White's king on the Bc8-d7-e6 diagonal and scored it 1-0;
   under our rules Black replies Bxe6, captures the king, and wins outright.
"""
import sys
import unittest
from pathlib import Path

import chess

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monster_chess import MonsterChessGame


class EnPassantComesFromWhitesLastMoveOnly(unittest.TestCase):
    # Black pawn d4; White to move with pawns on e2 and f2.  Whichever order
    # White plays e2e4 and f2f3 in, the final position is the same.
    FEN = "rnbqkbnr/ppp1pppp/8/8/3p4/8/2P1PP2/4K3 w kq - 0 1"

    def _black_ep_moves(self, first, second):
        game = MonsterChessGame(fen=self.FEN)
        game.apply_search_action(chess.Move.from_uci(first))
        game.apply_search_action(chess.Move.from_uci(second))
        self.assertFalse(game.is_white_turn)
        return sorted(m.uci() for m in game.get_search_actions()
                      if m.from_square == chess.D4
                      and m.to_square in (chess.C3, chess.E3))

    def test_double_push_second_confers_en_passant(self):
        self.assertEqual(self._black_ep_moves("f2f3", "e2e4"), ["d4e3"])

    def test_double_push_first_does_not(self):
        self.assertEqual(self._black_ep_moves("e2e4", "f2f3"), [])

    def test_the_two_orders_reach_the_same_position(self):
        # The divergence is real and known: identical placement, different
        # legality.  Pinned so a "harmless" refactor cannot quietly flip it.
        boards = []
        for first, second in (("e2e4", "f2f3"), ("f2f3", "e2e4")):
            game = MonsterChessGame(fen=self.FEN)
            game.apply_search_action(chess.Move.from_uci(first))
            game.apply_search_action(chess.Move.from_uci(second))
            boards.append(game.board.board_fen())
        self.assertEqual(boards[0], boards[1])


class WhiteMayNotEndItsTurnInCheck(unittest.TestCase):
    # playstrategy game gaBbbLhv after 12. Ke6: White's king stepped onto the
    # Bc8-d7-e6 diagonal (legal for us as a FIRST half-move) and they then
    # allowed f4-f5.  We require the second half to unpin the king.
    FEN = "r1bq1knr/1p2pPb1/4Knp1/8/2P2P2/7p/8/8 w - - 2 12"

    def setUp(self):
        self.game = MonsterChessGame(fen=self.FEN)
        self.game.white_half_pending = True

    def test_second_half_must_escape_the_attack(self):
        moves = sorted(m.uci() for m in self.game.get_search_actions())
        self.assertEqual(moves, ["e6e5"])
        self.assertNotIn("f4f5", moves)

    def test_black_would_capture_the_king_if_it_were_allowed(self):
        # Why the restriction only prunes losing moves: play f4f5 anyway and
        # Black's generator answers with the king capture, winning at once.
        self.game.apply_search_action(chess.Move.from_uci("f4f5"))
        reply = self.game.get_search_actions()
        self.assertEqual([m.uci() for m in reply], ["c8e6"])
        self.game.apply_search_action(reply[0])
        self.assertTrue(self.game.is_terminal())
        self.assertEqual(self.game.get_result(), -1)   # Black wins


if __name__ == "__main__":
    unittest.main()
