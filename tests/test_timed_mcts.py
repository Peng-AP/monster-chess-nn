"""Timed PUCT stops on complete batches; untimed behavior remains unchanged."""
import sys
from pathlib import Path
import time
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'native'))
import monster_native as native

START = 'rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1'


class TimedMCTSTests(unittest.TestCase):
    @staticmethod
    def bridge(buf, n, channels):
        return np.zeros(n, np.float32).tobytes(), np.zeros(n*4096, np.float32).tobytes()

    def test_no_clock_preserves_simulation_result(self):
        a, b = native.Tree(START), native.Tree(START)
        a.run_batched_puct(64, self.bridge, allow_early_stop=False, seed=3)
        b.run_batched_puct(64, self.bridge, allow_early_stop=False, seed=3, seconds=None)
        self.assertEqual(a.root_visits(), b.root_visits())
        self.assertEqual(a.best_action(temperature=0, seed=3), b.best_action(temperature=0, seed=3))

    def test_timed_stop_has_usable_nonnegative_visits(self):
        tree = native.Tree(START)
        start = time.monotonic()
        tree.run_batched_puct(10_000_000, self.bridge, allow_early_stop=False, seconds=.01)
        self.assertLess(time.monotonic()-start, .5)
        self.assertTrue(all(n>=0 for _,n in tree.root_visits()))
        self.assertIsNotNone(tree.best_action(temperature=0,seed=3)[0])

    def test_invalid_times_rejected(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                native.Tree(START).run_batched_puct(64, self.bridge, seconds=value)
