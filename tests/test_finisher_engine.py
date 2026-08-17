"""Contract tests for the forced-capture finisher.

The finisher exists because of a measurement: of 24 capped games at 1600
simulations, every one ended with Black ahead on material (mean +26.4, White on
a bare king in 21) while cycling through about twelve positions until the turn
limit -- and an exact search found forced king captures within four Black moves
in 6 of them. Those draws were unconverted wins.

What must hold: it is OFF by default so no existing result moves; it returns
only legal, directly playable moves; it never fires on White or mid-half; and
an exhausted search falls through to the network rather than being read as
"no win".
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import benchmark  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402


class _Inner:
    """Stand-in network engine: records that it was consulted."""

    def __init__(self):
        self.calls = 0

    def get_best_action(self, state, temperature=0.0):
        self.calls += 1
        return ("NETWORK", None, None)


class TestFinisherIsOptIn(unittest.TestCase):
    def test_absent_env_leaves_the_engine_untouched(self):
        # A behaviour switch that defaults on would silently move every
        # historical number.
        self.assertNotIn(benchmark.FINISHER_ENV, ("", None))
        saved = os.environ.pop(benchmark.FINISHER_ENV, None)
        try:
            self.assertFalse(benchmark._env_flag(benchmark.FINISHER_ENV))
        finally:
            if saved is not None:
                os.environ[benchmark.FINISHER_ENV] = saved

    def test_label_records_that_it_was_active(self):
        source = (ROOT / "src" / "benchmark.py").read_text(encoding="utf-8")
        self.assertIn('label += "+finisher"', source)


class TestFinisherGate(unittest.TestCase):
    def setUp(self):
        self.inner = _Inner()
        self.engine = benchmark._FinisherEngine(self.inner)

    def test_never_fires_on_white_to_move(self):
        game = MonsterChessGame()
        self.assertTrue(game.is_white_turn)
        self.engine.get_best_action(game)
        self.assertEqual(self.engine.finisher_calls, 0)
        self.assertEqual(self.inner.calls, 1)

    def test_never_fires_mid_white_half(self):
        game = MonsterChessGame()
        game.apply_search_action(game.get_search_actions()[0])
        self.assertTrue(game.white_half_pending)
        self.engine.get_best_action(game)
        self.assertEqual(self.engine.finisher_calls, 0)

    def test_does_not_fire_while_white_still_has_material(self):
        # The opening has four White pawns; running an exact search there would
        # cost time in exactly the phase it cannot help.
        game = MonsterChessGame()
        game.apply_search_action(game.get_search_actions()[0])
        game.apply_search_action(game.get_search_actions()[0])
        self.assertFalse(game.is_white_turn)
        self.engine.get_best_action(game)
        self.assertEqual(self.engine.finisher_calls, 0)
        self.assertEqual(self.inner.calls, 1)

    def test_fires_against_a_bare_king(self):
        game = MonsterChessGame("r5kr/8/8/8/8/8/8/4K3 b - - 0 1")
        self.assertFalse(game.is_white_turn)
        self.engine.get_best_action(game)
        self.assertEqual(self.engine.finisher_calls, 1)

    def test_white_material_count_excludes_the_king(self):
        bare = MonsterChessGame("r5kr/8/8/8/8/8/8/4K3 b - - 0 1")
        self.assertEqual(benchmark._FinisherEngine._white_material(bare), 0)
        with_pawn = MonsterChessGame("r5kr/8/8/8/8/8/4P3/4K3 b - - 0 1")
        self.assertEqual(benchmark._FinisherEngine._white_material(with_pawn), 1)


class TestFinisherMovesArePlayable(unittest.TestCase):
    def test_an_immediate_capture_is_returned_and_is_playable(self):
        # Black rook on e8, White king on e1, open file: the capture is
        # available now and must come back directly playable through the
        # search-action path the game loop uses.
        game = MonsterChessGame("4r1k1/8/8/8/8/8/8/4K3 b - - 0 1")
        engine = benchmark._FinisherEngine(_Inner())
        action, _probs, _value = engine.get_best_action(game)
        self.assertNotEqual(action, "NETWORK")
        self.assertIn(action, game.get_legal_actions())
        game.apply_search_action(action)       # must not raise
        self.assertIsNone(game.board.king(__import__("chess").WHITE))

    def test_exhaustion_falls_through_to_the_network(self):
        # node_budget=1 guarantees the search cannot finish. "No answer" must
        # never be treated as "no win" -- it must defer to the network.
        inner = _Inner()
        engine = benchmark._FinisherEngine(inner, max_black_moves=4,
                                           node_budget=1)
        game = MonsterChessGame("r5kr/8/8/8/8/8/8/4K3 b - - 0 1")
        action, _p, _v = engine.get_best_action(game)
        self.assertEqual(action, "NETWORK")
        self.assertEqual(inner.calls, 1)


class TestFinisherDelegates(unittest.TestCase):
    def test_unknown_attributes_reach_the_wrapped_engine(self):
        inner = _Inner()
        inner.num_simulations = 1600
        engine = benchmark._FinisherEngine(inner)
        self.assertEqual(engine.num_simulations, 1600)


if __name__ == "__main__":
    unittest.main()
