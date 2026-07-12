import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chess
from monster_chess import MonsterChessGame
from mcts import _hangs_king, _king_safety_override, _turn_completing


def _info(*ucis):
    return [(SimpleNamespace(action=chess.Move.from_uci(u)), u, 10 - i)
            for i, u in enumerate(ucis)]


class KingSafetyOverrideTests(unittest.TestCase):
    def test_hanging_move_detected(self):
        # Black king b2 adjacent to White king a1: quiet moves hang to the
        # double-move king; capturing a1 ends the game (not a hang).
        g = MonsterChessGame(fen="8/8/8/8/8/8/1k6/K7 b - - 0 1")
        self.assertTrue(_hangs_king(g, chess.Move.from_uci("b2b3")))
        self.assertFalse(_hangs_king(g, chess.Move.from_uci("b2a1")))

    def test_override_swaps_to_surviving_action(self):
        g = MonsterChessGame(fen="8/8/8/8/8/8/1k6/K7 b - - 0 1")
        chosen = chess.Move.from_uci("b2b3")          # hangs
        info = _info("b2b3", "b2a1")                  # a1 capture survives (wins)
        out = _king_safety_override(g, chosen, info)
        self.assertEqual(out, chess.Move.from_uci("b2a1"))

    def test_override_keeps_action_when_all_hang(self):
        # Strip the winning capture from the candidates: everything hangs,
        # the chosen action must be kept (forced loss, no infinite search).
        g = MonsterChessGame(fen="8/8/8/8/8/8/1k6/K7 b - - 0 1")
        chosen = chess.Move.from_uci("b2b3")
        info = _info("b2b3", "b2c3")
        self.assertEqual(_king_safety_override(g, chosen, info), chosen)

    def test_white_first_half_exempt(self):
        # White's first half-move may pass through check: not turn-completing.
        g = MonsterChessGame()
        self.assertTrue(g.is_white_turn)
        self.assertFalse(_turn_completing(g))


if __name__ == "__main__":
    unittest.main()
