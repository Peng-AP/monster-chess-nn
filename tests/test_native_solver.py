"""Certainty propagation (MCTS-Solver), DIRECTIVE E6 / LC0 reference.

Plain MCTS *dilutes* a forced win: the winning line is one path among
thousands and its +1 is averaged into a mean, so the visit distribution never
concentrates hard enough to play it. Measured on this engine: 29% of
dominant-unfinished games held a forced king capture within 3 Black moves that
1600-sim search walked past, and endgame PV depth stays at 4 plies even at
51,200 sims.

The solver fixes the dilution rather than the depth — a proven result backs up
as an exact, unaveraged fact:

* any child proven winning for the side to move proves the node;
* all children proven losing proves the node lost;
* proofs are held in **White's** perspective, so White maximises and Black
  minimises — and because White's two half-moves do not change the side to
  move, that framing survives the half-pair where ply parity would not.

**Only a king capture proves anything.** The move-limit relabel (+-0.5 by
heuristic sign) is an opinion about an unfinished game; treating it as a proof
would let the search "prove" wins that were merely positions it liked when the
clock ran out.

OFF by default: it changes what the engine plays, so under section 0.1 it is
flagged and measured on its own.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "native"))

try:
    import monster_native as mn
except ImportError:
    mn = None

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"
MATE_IN_2 = "4k3/8/8/8/8/2q5/1r5r/K7 b - - 0 1"
MATE_IN_1 = "k7/8/8/8/8/8/8/r3K3 b - - 0 1"


def stub(buf, n, channels):
    """A flat evaluator: any proof found is the search's, not the network's."""
    return (np.zeros(n, dtype=np.float32).tobytes(),
            np.zeros(n * 4096, dtype=np.float32).tobytes())


def search(fen, sims, solver, batch=16):
    tree = mn.Tree(fen)
    tree.run_batched_puct(sims, stub, batch_size=batch, channels=17,
                          allow_early_stop=False, solver=solver)
    return tree


@unittest.skipIf(mn is None, "native crate not built")
class TestProofsAreFound(unittest.TestCase):
    def test_an_immediate_capture_is_proven(self):
        tree = search(MATE_IN_1, 60, solver=True, batch=8)
        self.assertEqual(tree.root_proof(), -1.0)  # Black wins, White's frame

    def test_a_forced_capture_in_two_is_proven(self):
        tree = search(MATE_IN_2, 800, solver=True)
        self.assertEqual(tree.root_proof(), -1.0)
        proven = [a for a, p in tree.root_proofs() if p is not None]
        self.assertTrue(proven)

    def test_without_the_solver_nothing_is_proven(self):
        tree = search(MATE_IN_2, 800, solver=False)
        self.assertIsNone(tree.root_proof())
        self.assertTrue(all(p is None for _a, p in tree.root_proofs()))

    def test_a_proven_line_reports_an_exact_value(self):
        # The owner's complaint, 2026-07-17: a proven mate reading ~+0.7
        # because the root average includes refuted siblings.
        proven = search(MATE_IN_2, 800, solver=True).best_action(temperature=0.0)
        diluted = search(MATE_IN_2, 800, solver=False).best_action(temperature=0.0)
        self.assertEqual(proven[2], 1.0)
        self.assertLessEqual(diluted[2], 1.0)

    def test_the_reported_value_is_in_the_movers_frame(self):
        # A root child's Q accumulates in the root's side-to-move perspective,
        # so "+1" means the MOVER wins. Returning White's frame here would flip
        # the sign for Black and disagree with the non-solver path.
        _a, _p, value = search(MATE_IN_1, 60, solver=True, batch=8).best_action(
            temperature=0.0)
        self.assertEqual(value, 1.0)  # Black to move, Black wins -> +1


@unittest.skipIf(mn is None, "native crate not built")
class TestTheCapIsNeverAProof(unittest.TestCase):
    def test_a_move_limit_position_is_not_proven(self):
        # At the cap the engine relabels by heuristic sign. That is an opinion,
        # not a result, and must never harden into a proof.
        import config
        tree = mn.Tree("3k4/8/8/8/8/2q5/1r5r/K7 b - - 0 1", False,
                       config.MAX_GAME_TURNS)
        tree.run_batched_puct(32, stub, batch_size=8, channels=17,
                              allow_early_stop=False, solver=True)
        self.assertIsNone(tree.root_proof())


@unittest.skipIf(mn is None, "native crate not built")
class TestItIsOffByDefault(unittest.TestCase):
    def test_default_leaves_the_search_unproven(self):
        tree = mn.Tree(MATE_IN_2)
        tree.run_batched_puct(400, stub, batch_size=16, channels=17,
                              allow_early_stop=False)
        self.assertIsNone(tree.root_proof())

    def test_the_flag_changes_nothing_where_there_is_no_proof(self):
        # Opening position, no forced result: both paths must agree, so the
        # flag is not quietly altering ordinary play.
        off = search(START_FEN, 200, solver=False).best_action(temperature=0.0)
        on = search(START_FEN, 200, solver=True).best_action(temperature=0.0)
        self.assertEqual(off[0], on[0])


if __name__ == "__main__":
    unittest.main()
