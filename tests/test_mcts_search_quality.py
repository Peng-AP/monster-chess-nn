"""Phase 0 search-quality contract (REWORK_PLAN.md §3.1).

Pins the fix for defect D1: batched PUCT must actually visit root children at the
simulation budgets the pipeline uses.  Before the Phase 1 fix these assertions FAIL
at N=80/200 (zero children visited — move selection is random.choice).  After the
fix they pass at every budget and stay green forever.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import chess


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import POLICY_SIZE
from evaluation import evaluate
from mcts import MCTS, MCTSNode
from monster_chess import MonsterChessGame


class _StubPolicyValueEvaluator:
    """Heuristic value + flat (uniform) policy logits.

    Exposes both ``batch_evaluate`` and ``evaluate_with_policy`` so MCTS dispatches
    to the batched-PUCT path (the one D1 broke).  Flat logits make the search rely
    entirely on PUCT exploration, so a healthy visit distribution proves the search
    machinery works rather than the policy prior doing the work.
    """

    def __call__(self, game_state):
        return evaluate(game_state)

    def evaluate_with_policy(self, game_state):
        return evaluate(game_state), np.zeros(POLICY_SIZE, dtype=np.float32)

    def batch_evaluate(self, game_states):
        return [evaluate(gs) for gs in game_states]

    def batch_evaluate_with_policy(self, game_states):
        vals = [evaluate(gs) for gs in game_states]
        pols = [np.zeros(POLICY_SIZE, dtype=np.float32) for _ in game_states]
        return vals, pols


def _run_root(sims):
    """Run batched PUCT from the standard opening; return the searched root."""
    game = MonsterChessGame()
    engine = MCTS(num_simulations=sims, eval_fn=_StubPolicyValueEvaluator())
    root = MCTSNode(game.clone())
    engine._run_batched_puct(root)
    return root


class SearchQualityContract(unittest.TestCase):
    def test_children_are_actually_visited(self):
        for sims in (80, 200, 800):
            root = _run_root(sims)
            self.assertTrue(root.children, f"no children expanded at sims={sims}")
            total_child_visits = sum(c.visit_count for c in root.children)
            self.assertGreaterEqual(
                total_child_visits, 0.8 * sims,
                f"sims={sims}: only {total_child_visits} child visits "
                f"(expected >= {0.8 * sims:.0f}); search is wasting simulations",
            )

    def test_batched_puct_finds_a_forced_win(self):
        # A position with a genuinely dominant move: Black rook e2 captures the White
        # king on e1.  Through the batched-PUCT path (flat policy), the search must
        # both select the winning move at temperature 0 and read the root as clearly
        # winning for the side to move.  (On the near-symmetric opening a flat policy
        # correctly spreads visits, so this position tests concentration, not that.)
        fen = "4k3/8/8/8/8/8/4r3/4K3 b - - 0 1"
        game = MonsterChessGame(fen=fen)
        engine = MCTS(num_simulations=200, eval_fn=_StubPolicyValueEvaluator(),
                      root_noise=False)
        action, _probs, root_value = engine.get_best_action(game, temperature=0.0)
        self.assertEqual(action, chess.Move.from_uci("e2e1"),
                         "batched PUCT did not select the king-capturing move")
        self.assertGreater(root_value, 0.5,
                           "batched PUCT did not read the forced win as winning")

    def test_action_probs_not_uniform_at_high_sims(self):
        game = MonsterChessGame()
        engine = MCTS(num_simulations=800, eval_fn=_StubPolicyValueEvaluator(),
                      root_noise=False)
        _action, action_probs, _value = engine.get_best_action(game, temperature=1.0)
        self.assertTrue(action_probs)
        probs = sorted(action_probs.values(), reverse=True)
        self.assertGreater(
            probs[0], 2.0 * probs[-1],
            "action_probs are effectively uniform; visit distribution carries no signal",
        )


if __name__ == "__main__":
    unittest.main()
