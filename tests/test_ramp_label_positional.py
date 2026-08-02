"""Ramp labels are POSITIONAL: dropping records silently rewrites the survivors.

Found while trying to raise the pawn-phase density of generated cliff self-play
(DIRECTIVE D3, OVERNIGHT_REPORT §8.1). The obvious fix -- "filter the records
down to the wP>=3 ones before merging" -- is wrong, and wrong in the worst way:
it produces a corpus that looks fine, trains without error, and carries value
targets that no longer mean what they say.

`_discounted_results` derives each record's target from its distance to the END
OF ITS OWN LIST. It has no notion of a ply index, a clock, or a truncation. So
a record 40 plies from the finish, once its successors are removed, is relabelled
as though it were the final position -- full-strength win/loss instead of a
discounted one.

These tests exist so that whoever next tries to make D3's density target work
finds out from a red test rather than from a mysteriously worse arm.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processor import _discounted_results  # noqa: E402


def records(n, result=-1.0):
    return [{"fen": f"fen{i}", "game_result": result} for i in range(n)]


class TestLabelsAreDistanceToEndOfList(unittest.TestCase):
    def test_last_record_gets_the_undiscounted_result(self):
        out = _discounted_results(records(10), horizon=60, floor=0.5)
        self.assertAlmostEqual(out[-1], -1.0, places=6)

    def test_earlier_records_are_discounted_toward_the_floor(self):
        out = _discounted_results(records(10), horizon=60, floor=0.5)
        # Monotone in magnitude toward the end of the game.
        mags = [abs(v) for v in out]
        self.assertTrue(all(a <= b + 1e-9 for a, b in zip(mags, mags[1:])), mags)
        self.assertLess(mags[0], mags[-1])


class TestTruncationRewritesLabels(unittest.TestCase):
    """The hazard itself, stated as a test."""

    def test_dropping_the_tail_relabels_the_new_last_record(self):
        full = _discounted_results(records(40), horizon=60, floor=0.5)
        truncated = _discounted_results(records(20), horizon=60, floor=0.5)

        # Record 19 is mid-game in the full game and final in the truncated one.
        self.assertNotAlmostEqual(full[19], truncated[19], places=3)
        self.assertAlmostEqual(truncated[19], -1.0, places=6)
        self.assertLess(abs(full[19]), 0.95)

    def test_the_whole_prefix_shifts_not_just_the_boundary(self):
        full = _discounted_results(records(40), horizon=60, floor=0.5)
        truncated = _discounted_results(records(20), horizon=60, floor=0.5)
        # Every surviving record is relabelled, not merely the last one.
        differing = sum(1 for i in range(20)
                        if abs(full[i] - truncated[i]) > 1e-6)
        self.assertEqual(differing, 20)

    def test_a_filtered_game_cannot_be_relabelled_by_reprocessing(self):
        # Sanity: there is no "fix it afterwards" -- the information about how
        # far the real game ran is gone once the records are dropped.
        full = _discounted_results(records(40), horizon=60, floor=0.5)
        kept_indices = [0, 5, 11, 19]           # a wP>=3 filter, say
        filtered = _discounted_results(
            [records(40)[i] for i in kept_indices], horizon=60, floor=0.5)
        for slot, original in enumerate(kept_indices):
            self.assertNotAlmostEqual(filtered[slot], full[original], places=3)


if __name__ == "__main__":
    unittest.main()
