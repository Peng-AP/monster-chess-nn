"""Phase-router contracts: root routing by White pawn count, turn stickiness,
spec loading, and _build_engine dispatch."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from monster_chess import MonsterChessGame
from router import RouterMCTS, load_router

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"
TWO_PAWN_FEN = "rnbqkbnr/pppppppp/8/8/8/8/3PP3/4K3 w kq - 0 1"
NO_PAWN_FEN = "4k3/8/8/8/8/8/8/4K3 b - - 0 1"


class _FakeEngine:
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.eval_fn = lambda state: name

    def get_best_action(self, root_state, temperature=1.0):
        self.calls.append(root_state.fen())
        return None, {}, 0.0


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.opening = _FakeEngine("opening")
        self.late = _FakeEngine("late")
        self.router = RouterMCTS(self.opening, self.late, min_white_pawns=3)

    def test_full_pawns_route_to_opening_model(self):
        self.router.get_best_action(MonsterChessGame(START_FEN))
        self.assertEqual(len(self.opening.calls), 1)
        self.assertEqual(len(self.late.calls), 0)

    def test_low_pawns_route_to_late_model(self):
        self.router.get_best_action(MonsterChessGame(TWO_PAWN_FEN))
        self.assertEqual(len(self.opening.calls), 0)
        self.assertEqual(len(self.late.calls), 1)

    def test_black_roots_route_by_the_same_rule(self):
        self.router.get_best_action(MonsterChessGame(NO_PAWN_FEN))
        self.assertEqual(len(self.late.calls), 1)

    def test_pending_second_half_sticks_with_the_turn_engine(self):
        # m1 routed to the opening engine; even if the m2 root's pawn count
        # were to cross the threshold, the same engine finishes the turn.
        self.router.get_best_action(MonsterChessGame(START_FEN))
        pending = MonsterChessGame(TWO_PAWN_FEN)
        pending.white_half_pending = True
        self.router.get_best_action(pending)
        self.assertEqual(len(self.opening.calls), 2)
        self.assertEqual(len(self.late.calls), 0)

    def test_pending_root_without_history_routes_fresh(self):
        pending = MonsterChessGame(TWO_PAWN_FEN)
        pending.white_half_pending = True
        self.router.get_best_action(pending)
        self.assertEqual(len(self.late.calls), 1)

    def test_eval_for_routes_without_mutating_stickiness(self):
        self.assertEqual(self.router.eval_for(MonsterChessGame(TWO_PAWN_FEN)),
                         "late")
        self.assertIsNone(self.router._turn_engine)

    def test_load_router_resolves_relative_paths_from_repo_root(self):
        with tempfile.TemporaryDirectory() as td:
            spec = os.path.join(td, "router.json")
            with open(spec, "w", encoding="utf-8") as f:
                json.dump({"opening_model": "models/a.pt",
                           "late_model": os.path.join(td, "b.pt"),
                           "min_white_pawns": 2}, f)
            seen = []

            def fake_evaluator(path):
                seen.append(path)
                return lambda state: 0.0

            with mock.patch("evaluation.NNEvaluator", side_effect=fake_evaluator):
                router = load_router(spec, sims=8)
        import router as router_mod
        self.assertEqual(os.path.normpath(seen[0]),
                         os.path.join(router_mod.ROOT, "models", "a.pt"))
        self.assertEqual(seen[1], os.path.join(td, "b.pt"))
        self.assertEqual(router.min_white_pawns, 2)
        self.assertEqual(router.opening_engine.num_simulations, 8)
        self.assertFalse(router.opening_engine.root_noise)

    def test_build_engine_dispatches_router_specs(self):
        import benchmark
        sentinel = object()
        with mock.patch("router.load_router", return_value=sentinel) as loader:
            engine, label = benchmark._build_engine(
                os.path.join("models", "experiments", "router_v17_ramp",
                             "router.json"), 400)
        self.assertIs(engine, sentinel)
        self.assertEqual(label, "router:router_v17_ramp")
        loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
