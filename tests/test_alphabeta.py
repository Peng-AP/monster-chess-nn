"""Opt-in alpha-beta contract; incumbent MCTS is not changed."""
from pathlib import Path
import sys
import unittest
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'native'))
sys.path.insert(0, str(ROOT / 'src'))
import monster_native as native
from monster_chess import MonsterChessGame

START = 'rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1'


class AlphaBetaTests(unittest.TestCase):
    def test_optimized_fixed_depth_matches_uncached_search(self):
        rng = random.Random(5193)
        game = MonsterChessGame()
        for i in range(60):
            if game.is_terminal():
                game = MonsterChessGame()
            if i % 3 == 0:
                kwargs = dict(pending=game.white_half_pending, turn_count=game.turn_count,
                              max_depth=2, seconds=10, node_limit=1_000_000)
                a = native.alphabeta_search(game.board.fen(en_passant='fen'), optimizations=False, **kwargs)
                b = native.alphabeta_search(game.board.fen(en_passant='fen'), optimizations=True, **kwargs)
                self.assertFalse(a.interrupted)
                self.assertFalse(b.interrupted)
                self.assertEqual(a.value, b.value)
            actions = game.get_search_actions()
            if not actions:
                game = MonsterChessGame()
            else:
                game.apply_search_action(rng.choice(actions))

    def test_caches_have_real_hits_without_changing_value(self):
        a = native.alphabeta_search(START, max_depth=3, seconds=10, optimizations=False)
        b = native.alphabeta_search(START, max_depth=3, seconds=10, optimizations=True)
        self.assertEqual(a.value, b.value)
        self.assertFalse(a.interrupted or b.interrupted)
        self.assertGreater(b.eval_cache_hits, 0)
        self.assertGreater(b.tt_hits, 0)

    def test_timeout_returns_legal_fallback_without_inventing_value(self):
        r = native.alphabeta_search(START, seconds=1, node_limit=1)
        self.assertIsNone(r.value)
        self.assertEqual(r.completed_depth, 0)
        self.assertTrue(r.interrupted)
        self.assertIn(r.action, native.Game(START).search_actions())

    def test_completed_iteration_and_bounded_elapsed(self):
        r = native.alphabeta_search(START, seconds=.03)
        self.assertGreaterEqual(r.completed_depth, 1)
        self.assertLess(r.elapsed_seconds, .3)
        self.assertTrue(r.interrupted)

    def test_phase_changes_capture_reach(self):
        fen = '8/8/8/8/4k3/8/4K3/8 w - - 0 1'
        first = native.alphabeta_search(fen, max_depth=1)
        second = native.alphabeta_search(fen, pending=True, max_depth=1)
        self.assertEqual(first.value, 1)
        self.assertLess(second.value, 1)

    def test_root_repetition_has_no_move(self):
        key = ' '.join(START.split()[:4])
        r = native.alphabeta_search(START, prior_positions=[key, key])
        self.assertEqual(r.value, 0)
        self.assertIsNone(r.action)

    def test_bad_limits_and_history_rejected(self):
        for kwargs in ({'seconds': float('nan')}, {'seconds': 0},
                       {'max_depth': 0}, {'node_limit': 0},
                       {'prior_positions': [START]}):
            with self.assertRaises(ValueError):
                native.alphabeta_search(START, **kwargs)

    def test_capture_takes_precedence_over_turn_cap(self):
        r = native.alphabeta_search('8/8/8/8/8/8/8/4K3 b - - 0 1', turn_count=150)
        self.assertEqual(r.value, 1)
        self.assertIsNone(r.action)
