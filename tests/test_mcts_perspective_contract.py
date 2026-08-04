"""The PUCT/backprop perspective contract, on hand-built trees with known Q.

`CONTEXT.md` §1.2 calls this the subtlest logic in the engine, and the
directive's risk table records that M2's metric bugs came from misreading it
**twice**. So before the native port exists, the rule is written down here as
executable arithmetic computed by hand — not as a differential.

A differential between two implementations can only say they disagree. These
tests say what the answer *must* be, so when the port disagrees there is no
question about which side is wrong.

The rule, in two parts:

1. **Backprop perspective.** A node accumulates value in the perspective of the
   side that moved *into* it — that is, its parent's side to move — so the
   parent reads Q consistently during selection. The root, having no parent,
   uses its own side.

2. **FPU perspective flip.** An unvisited child starts at parent-Q minus 0.30,
   expressed in the *selector's* perspective. `parent.q_value` is stored in the
   grandparent's perspective, so it matches the selector only when grandparent
   and parent share a side to move. Under strict alternation the flip happens
   every non-root ply — but **White's two half-moves do not flip side**, which
   is exactly why the general grandparent test is required and why a port
   written from chess intuition gets it wrong.

The tree used throughout is the one that makes the asymmetry visible:

    root (White, first half)
      -> a (White, second half pending)   -- same side as root
        -> b (Black to move)              -- side changes
          -> c (White to move)            -- side changes again
"""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import C_PUCT, FPU_REDUCTION  # noqa: E402
from mcts import MCTSNode, MCTS  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def build_chain():
    """root(W) -> a(W, pending) -> b(Black) -> c(White), via real states."""
    root_state = MonsterChessGame(START_FEN)
    root = MCTSNode(root_state)

    a_state = root_state.clone()
    a_state.apply_search_action(a_state.get_search_actions()[0])
    a = MCTSNode(a_state, parent=root, prior=0.5)
    root.children.append(a)

    b_state = a_state.clone()
    b_state.apply_search_action(b_state.get_search_actions()[0])
    b = MCTSNode(b_state, parent=a, prior=0.5)
    a.children.append(b)

    c_state = b_state.clone()
    c_state.apply_search_action(c_state.get_search_actions()[0])
    c = MCTSNode(c_state, parent=b, prior=0.5)
    b.children.append(c)
    return root, a, b, c


class TestTheChainHasTheShapeTheTestsAssume(unittest.TestCase):
    def test_white_keeps_the_move_across_its_two_halves(self):
        root, a, b, c = build_chain()
        self.assertTrue(root.state.is_white_turn)
        self.assertFalse(root.state.white_half_pending)
        # The whole point: a is still White to move.
        self.assertTrue(a.state.is_white_turn)
        self.assertTrue(a.state.white_half_pending)
        self.assertFalse(b.state.is_white_turn)
        self.assertTrue(c.state.is_white_turn)


class TestBackpropagationPerspective(unittest.TestCase):
    def test_value_is_stored_in_the_parents_perspective(self):
        root, a, b, c = build_chain()
        value = 0.8  # from White's perspective
        MCTS._backpropagate(None, c, value)

        # c's parent is b (Black to move) -> c stores the negated value.
        self.assertAlmostEqual(c.total_value, -0.8)
        # b's parent is a (White to move) -> b stores it unchanged.
        self.assertAlmostEqual(b.total_value, 0.8)
        # a's parent is root (White to move) -> unchanged.
        self.assertAlmostEqual(a.total_value, 0.8)
        # root has no parent -> uses its own side, which is White.
        self.assertAlmostEqual(root.total_value, 0.8)
        for node in (root, a, b, c):
            self.assertEqual(node.visit_count, 1)

    def test_a_black_rooted_tree_negates_at_the_root(self):
        state = MonsterChessGame(START_FEN)
        state.apply_search_action(state.get_search_actions()[0])
        state.apply_search_action(state.get_search_actions()[0])
        self.assertFalse(state.is_white_turn)
        root = MCTSNode(state)
        MCTS._backpropagate(None, root, 0.6)
        self.assertAlmostEqual(root.total_value, -0.6)

    def test_q_value_is_zero_while_unvisited(self):
        root, a, _b, _c = build_chain()
        self.assertEqual(a.q_value, 0.0)
        MCTS._backpropagate(None, a, 1.0)
        self.assertAlmostEqual(a.q_value, 1.0)


class TestFpuPerspectiveFlip(unittest.TestCase):
    def test_no_flip_when_grandparent_and_parent_share_a_side(self):
        # Selecting among a's children: parent = a (White), grandparent = root
        # (White). Same side, so parent-Q is already in the selector's frame.
        root, a, b, _c = build_chain()
        a.visit_count, a.total_value = 4, 2.0  # q = 0.5
        b.visit_count = 0
        b.prior = 0.25

        expected_fpu = 0.5 - FPU_REDUCTION
        expected = expected_fpu + C_PUCT * 0.25 * math.sqrt(max(1, a.visit_count))
        self.assertAlmostEqual(b.puct_score(), expected)

    def test_flip_when_the_side_changes(self):
        # Selecting among b's children: parent = b (Black), grandparent = a
        # (White). Different sides, so parent-Q must be negated.
        root, a, b, c = build_chain()
        b.visit_count, b.total_value = 4, 2.0  # q = 0.5, in a's (White) frame
        c.visit_count = 0
        c.prior = 0.25

        expected_fpu = -0.5 - FPU_REDUCTION  # negated first, then reduced
        expected = expected_fpu + C_PUCT * 0.25 * math.sqrt(max(1, b.visit_count))
        self.assertAlmostEqual(c.puct_score(), expected)

    def test_the_flip_actually_changes_the_answer(self):
        # Guards against a port that omits the flip and still passes by luck.
        root, a, b, c = build_chain()
        a.visit_count, a.total_value = 4, 2.0
        b.visit_count, b.total_value = 4, 2.0
        b.prior = c.prior = 0.25
        b.visit_count = 0
        unflipped_case = b.puct_score()
        b.visit_count, b.total_value = 4, 2.0
        c.visit_count = 0
        flipped_case = c.puct_score()
        self.assertNotAlmostEqual(unflipped_case, flipped_case)

    def test_fpu_is_clamped_into_range(self):
        root, a, b, _c = build_chain()
        a.visit_count, a.total_value = 1, 5.0  # q = 5.0, out of range
        b.visit_count = 0
        b.prior = 0.0
        # clamp(5.0 - 0.30) -> 1.0
        self.assertAlmostEqual(b.puct_score(), 1.0)

        a.total_value = -5.0
        self.assertAlmostEqual(b.puct_score(), -1.0)

    def test_root_children_use_zero_when_there_is_no_grandparent(self):
        root, a, _b, _c = build_chain()
        root.visit_count = 9
        a.visit_count = 0
        a.prior = 0.5
        # root is a's parent and has no parent itself -> parent_q path still
        # applies (a.parent is root, root.parent is None => no flip).
        expected = (0.0 - FPU_REDUCTION) + C_PUCT * 0.5 * math.sqrt(9)
        self.assertAlmostEqual(a.puct_score(), expected)


class TestVisitedChildScore(unittest.TestCase):
    def test_visited_children_use_q_plus_scaled_prior(self):
        root, a, _b, _c = build_chain()
        root.visit_count = 16
        a.visit_count, a.total_value = 3, 1.5  # q = 0.5
        a.prior = 0.4
        expected = 0.5 + C_PUCT * 0.4 * math.sqrt(16) / (1 + 3)
        self.assertAlmostEqual(a.puct_score(), expected)

    def test_unvisited_and_visited_agree_at_the_boundary(self):
        # The unvisited branch omits the /(1+N) divisor; with N = 0 that is 1,
        # so the two expressions must coincide apart from the FPU term.
        root, a, _b, _c = build_chain()
        root.visit_count = 25
        a.visit_count = 0
        a.prior = 0.3
        fpu_part = 0.0 - FPU_REDUCTION
        self.assertAlmostEqual(
            a.puct_score(), fpu_part + C_PUCT * 0.3 * math.sqrt(25) / (1 + 0))



# ---------------------------------------------------------------------------
# The same contract, against the native arena. Same hand-computed numbers --
# not a differential, so a disagreement names which side is wrong.
# ---------------------------------------------------------------------------

sys.path.insert(0, str(ROOT / "native"))
try:
    import monster_native as mn
except ImportError:
    mn = None


def build_native_chain():
    """Mirror of build_chain() in the native arena, via the same actions."""
    py_state = MonsterChessGame(START_FEN)
    tree = mn.Tree(START_FEN)
    indices = [0]
    parent = 0
    for _ in range(3):
        action = py_state.get_search_actions()[0]
        parent = tree.add_child(parent, action.uci(), 0.5)
        indices.append(parent)
        py_state.apply_search_action(action)
    return tree, indices  # [root, a, b, c]


@unittest.skipIf(mn is None, "native crate not built")
class TestNativeChainShape(unittest.TestCase):
    def test_white_keeps_the_move_across_its_two_halves(self):
        tree, (root, a, b, c) = build_native_chain()
        self.assertTrue(tree.is_white_turn(root))
        self.assertFalse(tree.white_half_pending(root))
        self.assertTrue(tree.is_white_turn(a))
        self.assertTrue(tree.white_half_pending(a))
        self.assertFalse(tree.is_white_turn(b))
        self.assertTrue(tree.is_white_turn(c))

    def test_states_match_the_python_chain(self):
        _py_root, py_a, py_b, py_c = build_chain()
        tree, (_root, a, b, c) = build_native_chain()
        self.assertEqual(tree.fen(a), py_a.state.fen())
        self.assertEqual(tree.fen(b), py_b.state.fen())
        self.assertEqual(tree.fen(c), py_c.state.fen())


@unittest.skipIf(mn is None, "native crate not built")
class TestNativeBackpropagation(unittest.TestCase):
    def test_value_is_stored_in_the_parents_perspective(self):
        tree, (root, a, b, c) = build_native_chain()
        tree.backpropagate(c, 0.8)
        self.assertAlmostEqual(tree.total_value(c), -0.8)
        self.assertAlmostEqual(tree.total_value(b), 0.8)
        self.assertAlmostEqual(tree.total_value(a), 0.8)
        self.assertAlmostEqual(tree.total_value(root), 0.8)
        for idx in (root, a, b, c):
            self.assertEqual(tree.visit_count(idx), 1)

    def test_q_value_is_zero_while_unvisited(self):
        tree, (_root, a, _b, _c) = build_native_chain()
        self.assertEqual(tree.q_value(a), 0.0)
        tree.backpropagate(a, 1.0)
        self.assertAlmostEqual(tree.q_value(a), 1.0)


@unittest.skipIf(mn is None, "native crate not built")
class TestNativeFpuFlip(unittest.TestCase):
    def test_no_flip_when_grandparent_and_parent_share_a_side(self):
        tree, (_root, a, b, _c) = build_native_chain()
        tree.set_stats(a, 4, 2.0)   # q = 0.5
        tree.set_prior(b, 0.25)
        expected = (0.5 - FPU_REDUCTION) + C_PUCT * 0.25 * math.sqrt(4)
        self.assertAlmostEqual(tree.puct_score(b), expected)

    def test_flip_when_the_side_changes(self):
        tree, (_root, _a, b, c) = build_native_chain()
        tree.set_stats(b, 4, 2.0)
        tree.set_prior(c, 0.25)
        expected = (-0.5 - FPU_REDUCTION) + C_PUCT * 0.25 * math.sqrt(4)
        self.assertAlmostEqual(tree.puct_score(c), expected)

    def test_fpu_is_clamped_into_range(self):
        tree, (_root, a, b, _c) = build_native_chain()
        tree.set_stats(a, 1, 5.0)
        tree.set_prior(b, 0.0)
        self.assertAlmostEqual(tree.puct_score(b), 1.0)
        tree.set_stats(a, 1, -5.0)
        self.assertAlmostEqual(tree.puct_score(b), -1.0)


@unittest.skipIf(mn is None, "native crate not built")
class TestNativeMatchesPythonNumerically(unittest.TestCase):
    def test_puct_agrees_across_a_grid_of_stats(self):
        # Sweep both perspective cases against the Python implementation.
        for visits, value, prior in [(4, 2.0, 0.25), (1, -0.9, 0.5),
                                     (7, 3.5, 0.1), (2, 0.0, 0.9)]:
            py_root, py_a, py_b, py_c = build_chain()
            tree, (_root, a, b, c) = build_native_chain()

            py_a.visit_count, py_a.total_value = visits, value
            py_b.prior = prior
            tree.set_stats(a, visits, value)
            tree.set_prior(b, prior)
            self.assertAlmostEqual(tree.puct_score(b), py_b.puct_score(),
                                   msg=f"no-flip case {visits},{value},{prior}")

            py_b.visit_count, py_b.total_value = visits, value
            py_c.prior = prior
            tree.set_stats(b, visits, value)
            tree.set_prior(c, prior)
            self.assertAlmostEqual(tree.puct_score(c), py_c.puct_score(),
                                   msg=f"flip case {visits},{value},{prior}")

    def test_constants_match_config(self):
        self.assertEqual(mn.C_PUCT, C_PUCT)
        self.assertEqual(mn.FPU_REDUCTION, FPU_REDUCTION)

if __name__ == "__main__":
    unittest.main()
