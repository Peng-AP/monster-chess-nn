"""The post-timeout hang: prove the next pool can start (DIRECTIVE E4 risk row).

Historic failure, three confirmed occurrences (March 2026): one game times out,
workers are killed, and the humanseed phase that follows never starts. No error
-- a frozen log and ~1.2GB processes still resident.

The fix has been in the tree for a while; what was missing is a test that it
*works*. This forces the exact sequence -- hang a worker, hit the timeout, tear
down, start a fresh pool -- and asserts the second pool completes. A regression
here reintroduces a failure whose signature is an overnight run that silently
produced nothing.
"""
import os
import sys
import time
import unittest
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_generation import terminate_pool  # noqa: E402


def _sleep_forever(_x):
    time.sleep(600)
    return "never"


def _quick(x):
    return x * 2


class TestPoolTeardown(unittest.TestCase):
    def test_a_timed_out_pool_does_not_block_the_next_one(self):
        executor = ProcessPoolExecutor(max_workers=2)
        futures = {executor.submit(_sleep_forever, i): i for i in range(2)}
        with self.assertRaises(TimeoutError):
            for _f in as_completed(futures, timeout=2):
                pass

        killed = terminate_pool(executor)
        self.assertGreaterEqual(killed, 1, "no live workers were killed")

        # The whole point: a fresh pool must start and finish promptly.
        started = time.time()
        with ProcessPoolExecutor(max_workers=2) as nxt:
            results = sorted(nxt.map(_quick, [1, 2, 3]))
        self.assertEqual(results, [2, 4, 6])
        self.assertLess(time.time() - started, 60,
                        "the second pool stalled, which is the original bug")

    def test_teardown_is_safe_on_a_pool_whose_work_is_done(self):
        # Idle workers are still *alive* -- a pool that has finished its tasks
        # has not shut down -- so a non-zero kill count here is correct, not a
        # leak. What matters is that teardown neither raises nor strands them.
        executor = ProcessPoolExecutor(max_workers=2)
        self.assertEqual(list(executor.map(_quick, [1, 2])), [2, 4])
        terminate_pool(executor)
        with ProcessPoolExecutor(max_workers=2) as nxt:
            self.assertEqual(list(nxt.map(_quick, [5])), [10])

    def test_teardown_reports_what_it_killed(self):
        executor = ProcessPoolExecutor(max_workers=3)
        for i in range(3):
            executor.submit(_sleep_forever, i)
        time.sleep(1.0)  # let the workers actually spawn
        self.assertGreaterEqual(terminate_pool(executor), 1)

    def test_snapshot_is_taken_before_shutdown(self):
        # The subtle half of the fix: shutdown(wait=False) clears _processes,
        # so a teardown that reads it afterwards finds nothing to kill and
        # leaves the zombies. Assert the source still snapshots first.
        import inspect
        src = inspect.getsource(terminate_pool)
        snapshot_at = src.index("_processes")
        shutdown_at = src.index("executor.shutdown")
        self.assertLess(snapshot_at, shutdown_at,
                        "_processes must be snapshotted before shutdown()")


if __name__ == "__main__":
    unittest.main()
