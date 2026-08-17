import sys
import unittest
from pathlib import Path

import chess


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import evaluate
from mcts import MCTS, MCTSNode, _softmax_masked
from monster_chess import MonsterChessGame


class _DummyState:
    def __init__(self, is_white_turn):
        self.is_white_turn = bool(is_white_turn)


class MCTSContracts(unittest.TestCase):
    def test_policy_temperature_flattens_priors(self):
        logits = [0.0, 2.0]
        baseline = _softmax_masked(logits, [0, 1], temperature=1.0)
        flatter = _softmax_masked(logits, [0, 1], temperature=2.0)
        self.assertGreater(flatter[0], baseline[0])
        self.assertLess(flatter[1], baseline[1])
        self.assertAlmostEqual(sum(flatter.values()), 1.0)

    def test_policy_temperature_must_be_positive(self):
        with self.assertRaises(ValueError):
            _softmax_masked([0.0], [0], temperature=0.0)
        with self.assertRaises(ValueError):
            MCTS(policy_temperature=-1.0)

    def test_search_parameters_are_instance_scoped(self):
        engine = MCTS(c_puct=1.25, fpu_reduction=0.2,
                      policy_temperature=1.4)
        self.assertEqual(engine.c_puct, 1.25)
        self.assertEqual(engine.fpu_reduction, 0.2)
        self.assertEqual(engine.policy_temperature, 1.4)

    def test_puct_fpu_uses_root_parent_perspective(self):
        root = MCTSNode(_DummyState(is_white_turn=True))
        root.visit_count = 10
        root.total_value = 4.0  # q = +0.4 in root perspective
        child = MCTSNode(_DummyState(is_white_turn=False), parent=root, prior=0.0)
        score = child.puct_score(c_puct=0.0, fpu_reduction=0.1)
        self.assertAlmostEqual(score, 0.3, places=6)

    def test_puct_fpu_flips_non_root_parent_q_for_selector_perspective(self):
        root = MCTSNode(_DummyState(is_white_turn=True))
        mid = MCTSNode(_DummyState(is_white_turn=False), parent=root, prior=1.0)
        mid.visit_count = 10
        mid.total_value = 6.0  # q = +0.6 from root perspective
        leaf = MCTSNode(_DummyState(is_white_turn=True), parent=mid, prior=0.0)
        # Selector at `mid` needs mid-perspective FPU; expected -0.6 - 0.1
        score = leaf.puct_score(c_puct=0.0, fpu_reduction=0.1)
        self.assertAlmostEqual(score, -0.7, places=6)

    def test_black_immediate_king_capture_is_selected(self):
        fen = "4k3/8/8/8/8/8/4r3/4K3 b - - 0 1"
        game = MonsterChessGame(fen=fen)
        engine = MCTS(num_simulations=24, eval_fn=evaluate)
        action, _, _ = engine.get_best_action(game, temperature=0.0)
        self.assertEqual(action, chess.Move.from_uci("e2e1"))


if __name__ == "__main__":
    unittest.main()
