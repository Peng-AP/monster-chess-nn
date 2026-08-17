"""The forced-capture solver must be exact — it is sizing a build decision.

A solver that over-claims makes the finisher look valuable when it is not
(DIRECTIVE E0(b) would green-light work on nothing); one that under-claims
buries the only lever aimed at the conversion pathology. So the tests below
pin both directions, plus the two rules-layer traps from the module docstring:
the move-limit clock, and unconditional pseudo-legal king capture.
"""
import sys
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forced_capture import (  # noqa: E402
    black_can_capture_now,
    forced_capture_depth,
    try_forced_capture_depth,
)
from monster_chess import MonsterChessGame  # noqa: E402


def game(fen):
    return MonsterChessGame(fen)


class TestImmediateCapture(unittest.TestCase):
    def test_rook_on_the_king_file_is_depth_1(self):
        # Black rook a8, White king a1, empty file between.
        g = game("r3k3/8/8/8/8/8/8/K7 b - - 0 1")
        self.assertTrue(black_can_capture_now(g))
        self.assertEqual(forced_capture_depth(g, max_black_moves=2), 1)

    def test_a_pin_does_not_prevent_capture(self):
        # Black king d8, black rook d5 pinned down the d-file by the White rook
        # d1, White king h5 on the rook's rank. Rxh5 is illegal in ordinary
        # chess and winning here: the capture ends the game before the pin can
        # ever be cashed. This is the divergence `attackers` gets right and
        # `legal_moves` gets wrong, so the test asserts both halves.
        g = game("3k4/8/8/3r3K/8/8/8/3R4 b - - 0 1")
        self.assertTrue(black_can_capture_now(g))
        self.assertEqual(forced_capture_depth(g, max_black_moves=1), 1)
        legal = {m.uci() for m in g.board.legal_moves}
        self.assertNotIn("d5h5", legal, "fixture stopped testing the pin case")

    def test_no_contact_is_not_a_capture(self):
        g = game("4k3/8/8/8/8/8/8/K6r b - - 0 1")
        g.board.remove_piece_at(chess.H1)
        g.board.set_piece_at(chess.H8, chess.Piece(chess.ROOK, chess.BLACK))
        self.assertFalse(black_can_capture_now(g))


class TestNoForcedWin(unittest.TestCase):
    def test_open_board_lone_rook_cannot_force(self):
        # A rook alone cannot force a capture against a double-moving king.
        g = game("4k2r/8/8/8/8/8/8/4K3 b - - 0 1")
        self.assertIsNone(forced_capture_depth(g, max_black_moves=2))

    def test_white_escapes_with_its_second_move(self):
        # The double move is the whole defence: any single-move net leaks.
        g = game("3qk3/8/8/8/8/8/8/4K3 b - - 0 1")
        self.assertIsNone(forced_capture_depth(g, max_black_moves=2))


class TestWhiteRefutes(unittest.TestCase):
    def test_white_capturing_the_black_king_refutes(self):
        # If Black's only try lets White capture first, it is not a forced win.
        g = game("4k3/8/8/8/8/8/4q3/4K3 b - - 0 1")
        depth = forced_capture_depth(g, max_black_moves=2)
        if depth is not None:
            # If it claims a win it must be the immediate one, never a depth-2
            # line that ignores White's counter-capture.
            self.assertEqual(depth, 1)
            self.assertTrue(black_can_capture_now(g))


class TestMoveLimitIsNeutralised(unittest.TestCase):
    def test_a_position_at_the_cap_is_still_solved(self):
        # The trap: is_terminal() fires at turn_count >= MAX_GAME_TURNS and
        # relabels by heuristic sign. A solver that inherited the clock would
        # see a finished game and report nothing -- which is precisely the
        # class of position this spike exists to search.
        g = game("r3k3/8/8/8/8/8/8/K7 b - - 0 1")
        g.turn_count = 149
        self.assertEqual(forced_capture_depth(g, max_black_moves=2), 1)

    def test_the_caller_state_is_not_mutated(self):
        g = game("r3k3/8/8/8/8/8/8/K7 b - - 0 1")
        g.turn_count = 149
        before = g.fen()
        forced_capture_depth(g, max_black_moves=2)
        self.assertEqual(g.fen(), before)
        self.assertEqual(g.turn_count, 149)


class TestBudget(unittest.TestCase):
    def test_budget_exhaustion_is_reported_not_silently_a_miss(self):
        g = game("r2qk2r/8/8/8/8/8/8/4K3 b - - 0 1")
        depth, nodes, exhausted = try_forced_capture_depth(
            g, max_black_moves=2, node_budget=5)
        self.assertTrue(exhausted or depth is not None)
        if exhausted:
            self.assertIsNone(depth)
            self.assertGreater(nodes, 0)

    def test_white_to_move_is_rejected(self):
        g = game("r3k3/8/8/8/8/8/8/K7 w - - 0 1")
        with self.assertRaises(ValueError):
            forced_capture_depth(g)


class TestClaimsAreRealisable(unittest.TestCase):
    def test_a_depth_1_claim_can_actually_be_played(self):
        # Self-consistency: if the solver says depth 1, some legal Black action
        # really does remove the White king.
        g = game("r3k3/8/8/8/8/8/8/K7 b - - 0 1")
        self.assertEqual(forced_capture_depth(g, max_black_moves=1), 1)
        removed = False
        for move in g.get_legal_actions():
            child = g.clone()
            child.apply_action(move)
            if child.board.king(chess.WHITE) is None:
                removed = True
                break
        self.assertTrue(removed)


if __name__ == "__main__":
    unittest.main()
