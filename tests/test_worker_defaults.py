"""Nothing that plays games defaults to a worker count that kills the box.

Every worker builds its own CUDA context. On the 5060 Ti, 14 of them dies at
init with "fatal : Memory allocation failure" and leaves orphaned ~1.4 GB
processes behind -- the same zombie-worker pattern that has eaten runs before.
`os.cpu_count()` on this box is 16 and `cpu_count() - 2` is 14, so either
idiom, written innocently, reintroduces the crash in an unattended driver.

Measured throughput plateaus long before the crash anyway (400 sims,
2026-08-01): 4 workers 5.39 decisions/s, 8 workers 7.11, 12 workers 7.38.

These tests pin the default, not the flag -- every tool still takes --workers.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from config import DEFAULT_GAME_WORKERS  # noqa: E402

# Files that build a pool of game-playing workers.
GAME_RUNNERS = [
    Path("src") / "data_generation.py",
    Path("tools") / "match.py",
    Path("tools") / "promotion_probe.py",
]


class TestDefaultIsSane(unittest.TestCase):
    def test_default_is_below_the_crash_point(self):
        # 14 crashed; 12 ran. Stay clear of both.
        self.assertLessEqual(DEFAULT_GAME_WORKERS, 12)
        self.assertGreaterEqual(DEFAULT_GAME_WORKERS, 2)


class TestNoRunnerDefaultsToCpuCount(unittest.TestCase):
    def test_runners_do_not_derive_a_default_from_cpu_count(self):
        # A --workers *flag* may exist; what must not exist is cpu_count()
        # feeding a default. Catches both `os.cpu_count()` and
        # `mp.cpu_count() - 2`.
        pattern = re.compile(r"cpu_count\s*\(")
        for rel in GAME_RUNNERS:
            text = (ROOT / rel).read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                self.assertIsNone(pattern.search(line),
                                  f"{rel}:{i} derives a worker count from "
                                  f"cpu_count(): {line.strip()!r}")

    def test_runners_share_the_one_constant(self):
        for rel in GAME_RUNNERS:
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("DEFAULT_GAME_WORKERS", text, str(rel))

    def test_generation_does_not_silently_restore_the_old_four_worker_cap(self):
        text = (ROOT / "src" / "data_generation.py").read_text(encoding="utf-8")
        self.assertNotIn("min(workers, 4)", text)


if __name__ == "__main__":
    unittest.main()
