"""Double-move-check rule contracts (confirmed 2026-07-04).

King capture wins UNCONDITIONALLY: a side may capture the enemy king even if
its own king is left attacked — the game ends before the reply.  Consequences:
- White in check may still spend its two moves capturing the Black king.
- A "defended" Black king within double-move range is NOT safe.
- Threat scans must report these threats without capturer-safety conditions.

These pin the fix for the v5-era engine blunder: Black's search modeled White
as unable to capture while in check, and lost to a human who could.
"""
import sys
import unittest
from pathlib import Path

import chess


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monster_chess import MonsterChessGame
from evaluation import (evaluate, _white_can_capture_king,
                        _white_can_capture_king_single,
                        _black_can_capture_king)


class WhiteCapturesWhileInCheck(unittest.TestCase):
    # White Ke1 is IN CHECK from the rook on e8; Black king on f2 is adjacent
    # and "defended" by that same rook (e-file? no - via rank 2? choose:
    # rook e8 checks Ke1 down the e-file; Black Kf2 sits one step away from
    # White's king. White must be allowed to play Kxf2 and win, remaining
    # nominally in check the whole time.
    FEN = "4r3/8/8/8/8/8/5k2/4K3 w - - 0 1"

    def test_atomic_pair_generator_offers_the_capture(self):
        game = MonsterChessGame(fen=self.FEN)
        pairs = game.get_legal_actions()
        capture_first = [p for p in pairs
                         if p[0].to_square == chess.F2]
        self.assertTrue(capture_first,
                        "White in check must be offered Kxf2 (king capture "
                        "wins unconditionally)")

    def test_halfmove_generator_offers_the_capture(self):
        game = MonsterChessGame(fen=self.FEN)
        moves = game.get_search_actions()
        self.assertIn(chess.Move.from_uci("e1f2"), moves)

    def test_second_half_capture_unconditional(self):
        # After a wasted first half (Ke1-d1), the Black king at c2 is within
        # one move; capturing must be offered even though the White king
        # remains attacked by the e8 rook... construct directly:
        fen = "4r3/8/8/8/8/8/2k5/4K3 w - - 0 1"
        game = MonsterChessGame(fen=fen)
        game.apply_search_action(chess.Move.from_uci("e1d1"))
        self.assertTrue(game.white_half_pending)
        moves = game.get_search_actions()
        self.assertEqual(moves, [chess.Move.from_uci("d1c2")],
                         "the winning king capture must dominate the "
                         "second-half move list")

    def test_threat_scans_ignore_capturer_safety(self):
        game = MonsterChessGame(fen=self.FEN)
        self.assertTrue(_white_can_capture_king(game.board))
        self.assertEqual(evaluate(game), 0.95)

    def test_single_move_scan_unconditional(self):
        fen = "4r3/8/8/8/8/8/2k5/4K3 w - - 0 1"
        game = MonsterChessGame(fen=fen)
        game.apply_search_action(chess.Move.from_uci("e1d1"))
        self.assertTrue(_white_can_capture_king_single(game.board))


class BlackCapturesWhileAttacked(unittest.TestCase):
    def test_black_capture_offered_even_if_black_king_left_attacked(self):
        # Black Kb8 next to White Ka8 (White king attacks b8); Black must be
        # offered Kxa8 even though python-chess would consider b8->a8 moving
        # while "attacked".  Add a White rook attacking a8 so Black's king
        # would be "unsafe" after the capture — irrelevant, game over.
        fen = "Kk6/8/8/8/8/8/8/R7 b - - 0 1"
        game = MonsterChessGame(fen=fen)
        moves = game.get_legal_actions()
        self.assertEqual(moves, [chess.Move.from_uci("b8a8")],
                         "the winning king capture must dominate Black's list")
        self.assertTrue(_black_can_capture_king(game.board))


if __name__ == "__main__":
    unittest.main()
