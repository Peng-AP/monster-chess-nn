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


class TestExplicitPliesToEndMakesFilteringSafe(unittest.TestCase):
    """The fix: stamp the distance at generation time, then dropping is safe.

    This is what unblocks D3. Generating cliff self-play produces only ~9.5%
    pawn-phase records by count because games leave the phase and the tail is
    long; the density target needs filtering, and filtering was unsafe.
    """

    @staticmethod
    def stamped(n, result=-1.0):
        return [{"fen": f"fen{i}", "game_result": result,
                 "plies_to_end": n - 1 - i} for i in range(n)]

    def test_explicit_field_reproduces_the_positional_labels(self):
        # A whole game must be labelled identically either way, or every
        # existing corpus stops being comparable to a regenerated one.
        plain = _discounted_results(records(40), horizon=60, floor=0.5)
        stamped = _discounted_results(self.stamped(40), horizon=60, floor=0.5)
        for a, b in zip(plain, stamped):
            self.assertAlmostEqual(a, b, places=12)

    def test_dropping_the_tail_no_longer_relabels_survivors(self):
        full = self.stamped(40)
        truncated = _discounted_results(full[:20], horizon=60, floor=0.5)
        whole = _discounted_results(full, horizon=60, floor=0.5)
        for i in range(20):
            self.assertAlmostEqual(whole[i], truncated[i], places=12)

    def test_filtering_out_the_middle_is_safe_too(self):
        full = self.stamped(40)
        whole = _discounted_results(full, horizon=60, floor=0.5)
        keep = [0, 5, 11, 19, 33]
        filtered = _discounted_results([full[i] for i in keep],
                                       horizon=60, floor=0.5)
        for slot, original in enumerate(keep):
            self.assertAlmostEqual(filtered[slot], whole[original], places=12)

    def test_the_last_surviving_record_is_not_promoted_to_a_finish(self):
        full = self.stamped(40)
        truncated = _discounted_results(full[:20], horizon=60, floor=0.5)
        # Without the stamp this read -1.0; it is mid-game and must stay so.
        self.assertLess(abs(truncated[-1]), 0.95)

    def test_a_corpus_without_the_field_is_unaffected(self):
        # Every corpus on disk predates this. Absent field -> old behaviour.
        self.assertNotIn("plies_to_end", records(5)[0])
        out = _discounted_results(records(5), horizon=60, floor=0.5)
        self.assertAlmostEqual(out[-1], -1.0, places=12)

    def test_segments_still_work_with_the_field_present(self):
        # Duplicated human games repeat records in one file; the stamp must
        # not disturb per-copy handling.
        seg = []
        for copy in range(3):
            for rec in self.stamped(10):
                rec = dict(rec, segment=copy)
                seg.append(rec)
        out = _discounted_results(seg, horizon=60, floor=0.5)
        first = out[:10]
        for copy in range(1, 3):
            self.assertEqual(out[copy * 10:(copy + 1) * 10], first)


if __name__ == "__main__":
    unittest.main()
