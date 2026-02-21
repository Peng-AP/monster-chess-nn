import sys
import unittest
from pathlib import Path

import chess


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monster_chess import MonsterChessGame


class MonsterChessContracts(unittest.TestCase):
    def test_black_actions_include_white_king_capture(self):
        # Black rook on e2 can capture White king on e1 in one move.
        fen = "4k3/8/8/8/8/8/4r3/4K3 b - - 0 1"
        game = MonsterChessGame(fen=fen)
        self.assertFalse(game.is_white_turn)
        moves = game.get_legal_actions()
        self.assertIn(chess.Move.from_uci("e2e1"), moves)

    def test_black_capture_king_sets_black_win_result(self):
        fen = "4k3/8/8/8/8/8/4r3/4K3 b - - 0 1"
        game = MonsterChessGame(fen=fen)
        game.apply_action(chess.Move.from_uci("e2e1"))
        self.assertTrue(game.is_terminal())
        self.assertEqual(game.get_result(), -1)


if __name__ == "__main__":
    unittest.main()
