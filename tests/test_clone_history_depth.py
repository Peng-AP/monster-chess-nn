"""clone() keeps a bounded move history -- enough for oscillation detection.

MonsterChessGame.clone used to copy the entire move stack, which made every
MCTS node expansion cost O(game length) and put 82% of a late-game decision
inside python-chess's Board.copy (measured 2026-08-01: 18.6 s of 22.6 s).

The bound is only safe because exactly one consumer reads history --
mcts._own_previous_moves, at offsets -1/-3/-4.  These tests fail if someone
lowers CLONE_HISTORY_PLIES below what those offsets need, or widens the
offsets past what clone retains: either way the oscillation override would
silently stop firing on searched-from-clone positions (which includes the
owner's play path, src/play.py:128).
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chess  # noqa: E402
import mcts as mcts_mod  # noqa: E402
from monster_chess import CLONE_HISTORY_PLIES, MonsterChessGame  # noqa: E402


def play_plies(n):
    """A game advanced n half-move decisions with legal, arbitrary moves."""
    g = MonsterChessGame()
    for _ in range(n):
        actions = g.get_search_actions()
        if not actions or g.is_terminal():
            break
        g.apply_search_action(actions[0])
    return g


class TestDepthCoversItsOnlyConsumer(unittest.TestCase):
    def test_retention_covers_the_oscillation_offsets(self):
        # _own_previous_moves indexes stack[-1], [-3], [-4] for White and
        # [-3] for Black. The deepest is 4; the constant must cover it.
        source = (ROOT / "src" / "mcts.py").read_text(encoding="utf-8")
        start = source.index("def _own_previous_moves")
        body = source[start:start + source[start:].index("\ndef ")]
        depths = [int(tok.strip(" ()"))
                  for line in body.splitlines() if "offsets = " in line
                  for tok in line.split("=")[1].split(",") if tok.strip(" ()")]
        self.assertTrue(depths, "no offsets found -- did the function change?")
        self.assertLessEqual(max(abs(d) for d in depths), CLONE_HISTORY_PLIES)

    def test_clone_preserves_the_moves_oscillation_reads(self):
        g = play_plies(30)
        self.assertGreater(len(g.board.move_stack), CLONE_HISTORY_PLIES)
        before = mcts_mod._own_previous_moves(g)
        after = mcts_mod._own_previous_moves(g.clone())
        self.assertEqual(before, after)
        self.assertTrue(before, "position should have history to compare")


class TestBoundIsActuallyBounded(unittest.TestCase):
    def test_clone_history_does_not_grow_with_game_length(self):
        short, long = play_plies(10), play_plies(60)
        self.assertGreater(len(long.board.move_stack),
                           len(short.board.move_stack))
        self.assertEqual(len(long.clone().board.move_stack),
                         CLONE_HISTORY_PLIES)
        self.assertEqual(len(long.clone().board.move_stack),
                         len(short.clone().board.move_stack))

    def test_repeated_cloning_does_not_erode_history(self):
        # Search clones a clone at every level; retention must be a floor, not
        # a budget that drains as the tree deepens.
        g = play_plies(30)
        for _ in range(10):
            g = g.clone()
            g.apply_search_action(g.get_search_actions()[0])
        self.assertEqual(len(g.board.move_stack), CLONE_HISTORY_PLIES + 1)


class TestPositionIsUnaffected(unittest.TestCase):
    def test_clone_is_position_identical(self):
        g = play_plies(40)
        c = g.clone()
        self.assertEqual(c.board.fen(), g.board.fen())
        self.assertEqual(c.is_white_turn, g.is_white_turn)
        self.assertEqual(c.turn_count, g.turn_count)
        self.assertEqual(c.white_half_pending, g.white_half_pending)
        self.assertEqual(sorted(m.uci() for m in c.board.legal_moves),
                         sorted(m.uci() for m in g.board.legal_moves))

    def test_clone_can_still_push_and_pop(self):
        # monster_chess probes legality with push/pop pairs on the clone; a
        # truncated history must not break the pop path.
        c = play_plies(40).clone()
        fen = c.board.fen()
        move = next(iter(c.board.legal_moves))
        c.board.push(move)
        c.board.pop()
        self.assertEqual(c.board.fen(), fen)

    def test_en_passant_survives_the_clone(self):
        # ep is board state, not stack state -- but it is the one thing a
        # truncated history could plausibly have dropped.
        g = MonsterChessGame(fen="rnbqkbnr/pppppppp/8/8/4P3/8/2PP1P2/4K3 b kq e3 0 1")
        self.assertEqual(g.clone().board.ep_square, chess.E3)
        self.assertEqual(g.clone().board.fen(), g.board.fen())


if __name__ == "__main__":
    unittest.main()
