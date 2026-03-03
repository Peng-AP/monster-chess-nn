import sys
import unittest
from pathlib import Path

import chess

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import evaluate, _black_can_capture_king, _white_can_capture_king
from monster_chess import MonsterChessGame


class BlackCanCaptureKingTests(unittest.TestCase):
    def test_rook_captures_king_on_same_file(self):
        # Black rook on e2 can capture White king on e1
        board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 b - - 0 1")
        self.assertTrue(_black_can_capture_king(board))

    def test_queen_captures_king_diagonally(self):
        # Black queen on d2 can capture White king on e1
        board = chess.Board("4k3/8/8/8/8/8/3q4/4K3 b - - 0 1")
        self.assertTrue(_black_can_capture_king(board))

    def test_no_capture_when_distant(self):
        # Black rook on h8, White king on a1 — no single-move capture
        board = chess.Board("4k2r/8/8/8/8/8/8/K7 b - - 0 1")
        self.assertFalse(_black_can_capture_king(board))

    def test_no_capture_when_blocked(self):
        # Black rook on e8 blocked by own pawn on e2, can't reach e1
        board = chess.Board("4kr2/8/8/8/8/8/4p3/4K3 b - - 0 1")
        self.assertFalse(_black_can_capture_king(board))

    def test_self_preservation_blocks_capture(self):
        # Black rook on e2 could take White king on e1, but Black king on
        # d1 would be left attacked by White rook on a1 after the capture.
        board = chess.Board("8/8/8/8/8/8/4r3/Rk1K4 b - - 0 1")
        # After Rxe1, Black king on b1 is attacked by Ra1? No — rook was
        # on a1, and after Re2xe1 the rook moves to e1, a1 is empty.
        # Actually Black king is on b1, White rook on a1 attacks b1.
        # After Re2xe1 (captures White king), Black king is still on b1
        # and White rook on a1 still attacks b1. So self-preservation blocks.
        self.assertFalse(_black_can_capture_king(board))

    def test_returns_false_when_no_white_king(self):
        # Degenerate: no White king on board
        board = chess.Board("4k3/8/8/8/8/8/8/8 b - - 0 1")
        self.assertFalse(_black_can_capture_king(board))

    def test_board_turn_restored(self):
        board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        original_turn = board.turn
        _black_can_capture_king(board)
        self.assertEqual(board.turn, original_turn)


class WhiteCanCaptureSymmetryTests(unittest.TestCase):
    """Verify existing _white_can_capture_king still works alongside new code."""

    def test_white_double_move_capture(self):
        # White king on c1, Black king on a1 — Kc1-b1-a1 captures
        board = chess.Board("8/8/8/8/8/8/8/k1K5 w - - 0 1")
        self.assertTrue(_white_can_capture_king(board))


class EvaluateWithCaptureThreatsTests(unittest.TestCase):
    def test_black_capture_threat_very_negative(self):
        # Black rook can capture White king
        game = MonsterChessGame(fen="4k3/8/8/8/8/8/4r3/4K3 b - - 0 1")
        val = evaluate(game)
        self.assertLessEqual(val, -0.90)

    def test_white_capture_threat_very_positive(self):
        # White king on c1 can reach Black king on a1 in double-move
        game = MonsterChessGame(fen="8/8/8/8/8/8/8/k1K5 w - - 0 1")
        val = evaluate(game)
        self.assertGreaterEqual(val, 0.90)

    def test_starting_position_near_zero(self):
        game = MonsterChessGame()
        val = evaluate(game)
        # Starting position should be mildly White-favored, not extreme
        self.assertGreater(val, -0.5)
        self.assertLess(val, 0.5)


class PieceSafetyTests(unittest.TestCase):
    def test_piece_safety_bonus_applied(self):
        # Verify the piece safety bonus is active by checking config value
        from config import PIECE_SAFETY_BONUS
        self.assertGreater(PIECE_SAFETY_BONUS, 0.0)

    def test_piece_safety_contributes_to_score(self):
        # A position with 2 rooks at safe distance (>= 3) from White king
        # should have piece safety contributing to Black's advantage.
        # Rooks on a7, h7; White king on d1 — distance >= 3 for both.
        game = MonsterChessGame(fen="4k3/r6r/8/8/8/8/4P3/3K4 b - - 0 1")
        val = evaluate(game)
        # Score should be negative (Black has rooks + safe distance bonus)
        self.assertLess(val, 0.0)


class BlackKingExposureTests(unittest.TestCase):
    def test_exposed_king_worse_for_black(self):
        # Black king near White king (exposed) vs far away (safe)
        exposed = MonsterChessGame(fen="8/8/8/8/3Kk3/8/8/8 b - - 0 1")
        safe = MonsterChessGame(fen="4k3/8/8/8/8/8/8/K7 b - - 0 1")
        exposed_val = evaluate(exposed)
        safe_val = evaluate(safe)
        # Exposed king should give worse (more positive / less negative) score for Black
        self.assertGreater(exposed_val, safe_val)


if __name__ == "__main__":
    unittest.main()
