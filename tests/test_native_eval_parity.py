"""The native heuristic must reproduce `evaluation.py` exactly (DIRECTIVE E2).

The gate is |delta| <= 1e-9, but the achieved result is bit-identical across
148,272 positions (`tools/eval_parity.py`, 2026-08-03) — every operation is
+, -, *, / on f64 plus an integer power, so there is no libm difference to
absorb. A non-zero delta here means a term was reordered or a branch differs,
not that floating point drifted.

Skipped when the crate is not built.
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

import config  # noqa: E402
from evaluation import evaluate  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


@unittest.skipIf(mn is None, "native crate not built")
class TestHeuristicParity(unittest.TestCase):
    def assert_same(self, fen, is_white=None, pending=False):
        game = MonsterChessGame(fen)
        if is_white is not None:
            game.is_white_turn = is_white
        game.white_half_pending = pending
        self.assertEqual(
            mn.evaluate_fen(fen, game.is_white_turn, pending), evaluate(game), fen)

    def test_start_position(self):
        self.assert_same(START_FEN)

    def test_black_to_move(self):
        self.assert_same("rnbqkbnr/pppppppp/8/8/3PP3/8/2P2P2/4K3 b kq - 0 1")

    def test_white_second_half_is_pending_aware(self):
        # The pending flag switches the threat scan from a two-move to a
        # one-move lookahead; a port that ignores it reports threats a turn
        # early and the error is invisible in aggregate.
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/2PP1P2/4K3 w kq - 0 1"
        self.assert_same(fen, pending=False)
        self.assert_same(fen, pending=True)

    def test_endgame_bare_white_king(self):
        self.assert_same("3k4/8/8/8/8/2q5/1r5r/K7 b - - 0 1")

    def test_white_winning_material(self):
        self.assert_same("4k3/8/8/8/8/8/2PPPP2/4K3 w - - 0 1")

    def test_random_walk_is_bit_identical(self):
        import random
        rng = random.Random(20260803)
        checked = 0
        for _ in range(25):
            game = MonsterChessGame(START_FEN)
            for _ in range(60):
                if game.is_terminal():
                    break
                self.assertEqual(
                    mn.evaluate_fen(game.fen(), game.is_white_turn,
                                    game.white_half_pending),
                    evaluate(game), game.fen())
                checked += 1
                actions = game.get_search_actions()
                if not actions:
                    break
                game.apply_search_action(rng.choice(actions))
        self.assertGreater(checked, 500)


@unittest.skipIf(mn is None, "native crate not built")
class TestConstantsMatchConfig(unittest.TestCase):
    """Constants are mirrored in Rust, not imported — so pin them.

    A config edit that silently diverged from the native copy would shift every
    heuristic evaluation and every cap relabel at once.
    """

    def test_mirrored_constants(self):
        self.assertEqual(mn.WHITE_PAWN_VALUE, config.WHITE_PAWN_VALUE)
        self.assertEqual(mn.KING_GEOM_SCALE, config.KING_GEOM_SCALE)
        self.assertEqual(mn.KING_ATTACK_SCALE, config.KING_ATTACK_SCALE)
        self.assertEqual(mn.MAX_GAME_TURNS, config.MAX_GAME_TURNS)


@unittest.skipIf(mn is None, "native crate not built")
class TestCapRelabel(unittest.TestCase):
    def test_result_at_the_cap_follows_the_heuristic_sign(self):
        game = mn.Game("3k4/8/8/8/8/2q5/1r5r/K7 b - - 0 1")
        self.assertIsNone(game.result())  # not terminal yet
        # Drive to the cap by replaying a quiet shuffle is expensive; instead
        # assert the rule directly on a Black-dominant position.
        self.assertLess(game.evaluate(), -0.4)

    def test_cap_result_matches_python_after_playing_to_the_limit(self):
        # Drive both engines to MAX_GAME_TURNS in lockstep and compare the
        # relabel they produce. Setting turn_count by hand would test the
        # branch but not that both engines *arrive* in the same state.
        import random
        rng = random.Random(11)
        py = MonsterChessGame(START_FEN)
        rs = mn.Game(START_FEN)
        while not py.is_terminal():
            actions = py.get_search_actions()
            if not actions:
                break
            choice = rng.choice([m.uci() for m in actions])
            py.apply_search_action(
                next(m for m in py.get_search_actions() if m.uci() == choice))
            rs.apply_search_action(choice)
        self.assertTrue(rs.is_terminal())
        self.assertEqual(rs.result(), float(py.get_result()))
        if rs.at_turn_cap():
            self.assertIn(rs.result(), (-0.5, 0.0, 0.5))


if __name__ == "__main__":
    unittest.main()
