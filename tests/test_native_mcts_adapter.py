"""The native drop-in must satisfy the same contract as `mcts.MCTS` (D5/E4).

Every driver in the project calls `get_best_action(state, temperature)` and
applies the result with `apply_search_action`. If the adapter returns the wrong
*type*, or a move the state does not list, the failure surfaces deep inside
generation as corrupt recorded games — so it is pinned here instead.

The engine choice itself is pinned too. D5 keeps the default on `python` until
the E5 re-baseline; a silent flip would change every measurement at once.
"""
import os
import sys
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "native"))

try:
    import monster_native  # noqa: F401
    HAVE_NATIVE = True
except ImportError:
    HAVE_NATIVE = False

from benchmark import _build_engine, _engine_choice  # noqa: E402
from evaluation import evaluate  # noqa: E402
from mcts import MCTS  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


class TestEngineChoice(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("MONSTER_ENGINE", None)

    def tearDown(self):
        os.environ.pop("MONSTER_ENGINE", None)
        if self._saved is not None:
            os.environ["MONSTER_ENGINE"] = self._saved

    def test_default_is_python(self):
        # D5: the default stays python until the E5 re-baseline.
        self.assertEqual(_engine_choice(), "python")
        self.assertEqual(_engine_choice(None), "python")

    def test_explicit_argument_wins_over_environment(self):
        os.environ["MONSTER_ENGINE"] = "native"
        self.assertEqual(_engine_choice("python"), "python")
        self.assertEqual(_engine_choice(), "native")

    def test_unknown_engine_is_rejected(self):
        with self.assertRaises(ValueError):
            _engine_choice("rust")

    def test_python_factory_returns_the_python_search(self):
        engine, label = _build_engine(None, 32, engine="python")
        self.assertIsInstance(engine, MCTS)
        self.assertNotIn("native", label)


@unittest.skipIf(not HAVE_NATIVE, "native crate not built")
class TestAdapterContract(unittest.TestCase):
    def engine(self, sims=48):
        from native_mcts import NativeMCTS
        return NativeMCTS(num_simulations=sims, eval_fn=evaluate,
                          allow_early_stop=False)

    def test_returns_a_move_object_the_state_lists(self):
        state = MonsterChessGame(START_FEN)
        action, probs, value = self.engine().get_best_action(state, temperature=0.0)
        self.assertIsInstance(action, chess.Move)
        self.assertIn(action.uci(), {m.uci() for m in state.get_search_actions()})
        self.assertIsInstance(probs, dict)
        self.assertIsInstance(value, float)

    def test_the_returned_move_is_applicable(self):
        state = MonsterChessGame(START_FEN)
        action, _p, _v = self.engine().get_best_action(state, temperature=0.0)
        state.apply_search_action(action)   # the driver contract
        self.assertTrue(state.white_half_pending)

    def test_probabilities_are_a_distribution_over_legal_moves(self):
        state = MonsterChessGame(START_FEN)
        _a, probs, _v = self.engine().get_best_action(state, temperature=0.0)
        legal = {m.uci() for m in state.get_search_actions()}
        self.assertTrue(set(probs).issubset(legal))
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_it_can_drive_a_whole_game(self):
        engine = self.engine(32)
        state = MonsterChessGame(START_FEN)
        for _ in range(24):
            if state.is_terminal():
                break
            action, _p, _v = engine.get_best_action(state, temperature=0.0)
            self.assertIn(action.uci(), {m.uci() for m in state.get_search_actions()})
            state.apply_search_action(action)
        self.assertGreater(state.turn_count, 0)

    def test_factory_builds_the_native_search(self):
        from native_mcts import NativeMCTS
        engine, label = _build_engine(None, 32, engine="native")
        self.assertIsInstance(engine, NativeMCTS)
        self.assertIn("native", label)

    def test_tree_is_reused_across_whites_two_halves_then_dropped(self):
        engine = self.engine(32)
        state = MonsterChessGame(START_FEN)
        first, _p, _v = engine.get_best_action(state, temperature=0.0)
        state.apply_search_action(first)
        self.assertIsNotNone(engine._reuse_tree)   # White still owes m2
        second, _p, _v = engine.get_best_action(state, temperature=0.0)
        state.apply_search_action(second)
        self.assertIsNone(engine._reuse_tree)      # turn complete: dropped

    def test_history_is_taken_from_the_board_not_the_caller(self):
        # Callers never track history; the adapter reads move_stack, which is
        # what the oscillation penalty needs.
        from native_mcts import NativeMCTS
        state = MonsterChessGame(START_FEN)
        for _ in range(4):
            state.apply_search_action(state.get_search_actions()[0])
        history = NativeMCTS._history(state)
        self.assertEqual(history, [m.uci() for m in state.board.move_stack[-8:]])
        self.assertEqual(len(history), 4)


if __name__ == "__main__":
    unittest.main()
