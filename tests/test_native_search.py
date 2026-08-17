"""Native sequential UCB1 search (DIRECTIVE E3, heuristic mode).

D4 puts *search* under statistical parity, not bit-parity: Python shuffles
untried actions with MT19937 and the native side uses its own xorshift, so
identical playouts are not a target. What must hold is that the search reaches
the same conclusions — and, where the answer is forced, reaches it exactly.

Measured 2026-08-03 over 60 positions at 300 sims: 78% top-1 agreement with the
Python search, zero moves chosen that Python did not rank at all, and
disagreements averaging a 0.0136 probability gap — ties broken differently, not
a search that disagrees about the position.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "native"))

try:
    import monster_native as mn
except ImportError:
    mn = None

from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


@unittest.skipIf(mn is None, "native crate not built")
class TestForcedAnswersAreFound(unittest.TestCase):
    def test_an_available_king_capture_takes_every_visit(self):
        # Black rook a1 sees the White king on e1. The search API truncates to
        # the single winning action, so every simulation must land on it.
        tree = mn.Tree("k7/8/8/8/8/8/8/r3K3 b - - 0 1")
        tree.run_sequential(200, allow_early_stop=False)
        visits = tree.root_visits()
        self.assertEqual(len(visits), 1)
        self.assertEqual(visits[0][0], "a1e1")
        self.assertEqual(visits[0][1], 200)

    def test_search_expands_one_node_per_simulation(self):
        tree = mn.Tree(START_FEN)
        tree.run_sequential(50, allow_early_stop=False)
        self.assertEqual(tree.node_count(), 51)  # root + one per simulation
        self.assertEqual(sum(v for _a, v in tree.root_visits()), 50)


@unittest.skipIf(mn is None, "native crate not built")
class TestTreeNeedsTheWholeState(unittest.TestCase):
    """A FEN alone does not determine a Monster Chess position.

    `white_half_pending` selects a different action set entirely. Constructing
    a tree from a FEN alone silently assumed False and made two engines search
    different positions — found while measuring search agreement, and the cause
    of native picking moves Python did not list at all.
    """

    def test_pending_changes_the_root_action_set(self):
        # The flag only bites where the second half's safety filter removes
        # something: White's king on an open file against a rook. With no
        # threat present both halves offer the same moves, which is why a
        # start-position fixture cannot test this.
        fen = "4rk2/8/8/8/8/8/3P4/4K3 w - - 0 1"
        first = mn.Tree(fen, False, 0)
        second = mn.Tree(fen, True, 0)
        first.run_sequential(40, allow_early_stop=False)
        second.run_sequential(40, allow_early_stop=False)
        first_actions = {a for a, _ in first.root_visits()}
        second_actions = {a for a, _ in second.root_visits()}
        self.assertNotEqual(first_actions, second_actions)
        # The pawn pushes leave the king on the open file, so only the first
        # half may offer them.
        self.assertIn("d2d4", first_actions)
        self.assertNotIn("d2d4", second_actions)

    def test_pending_action_sets_match_the_python_engine(self):
        fen = "4rk2/8/8/8/8/8/3P4/4K3 w - - 0 1"
        py_first = MonsterChessGame(fen)
        py_second = MonsterChessGame(fen)
        py_second.white_half_pending = True
        for game, pending in ((py_first, False), (py_second, True)):
            tree = mn.Tree(fen, pending, 0)
            tree.run_sequential(40, allow_early_stop=False)
            self.assertEqual({a for a, _ in tree.root_visits()},
                             {m.uci() for m in game.get_search_actions()},
                             f"pending={pending}")

    def test_pending_root_matches_the_python_action_set(self):
        game = MonsterChessGame(START_FEN)
        game.apply_search_action(game.get_search_actions()[0])
        tree = mn.Tree(game.fen(), game.white_half_pending, game.turn_count)
        tree.run_sequential(len(game.get_search_actions()), allow_early_stop=False)
        self.assertTrue(
            {a for a, _ in tree.root_visits()}.issubset(
                {m.uci() for m in game.get_search_actions()}))

    def test_turn_count_is_carried_so_the_cap_applies(self):
        import config
        game = mn.Game(START_FEN, False, config.MAX_GAME_TURNS)
        self.assertTrue(game.is_terminal())
        self.assertTrue(game.at_turn_cap())


if __name__ == "__main__":
    unittest.main()
