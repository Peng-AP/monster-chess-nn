"""Contracts for the two Phase-5 lifts: vectorized mirror_policy and
explicit segment boundaries in _discounted_results."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import POLICY_SIZE  # noqa: E402
from data_processor import _discounted_results  # noqa: E402
from encoding import _MIRROR_PERM, mirror_move_index, mirror_policy  # noqa: E402


def _reference_mirror_policy(policy_vec):
    """The pre-vectorization implementation, kept as the oracle."""
    mirrored = np.zeros_like(policy_vec)
    for idx in range(POLICY_SIZE):
        if policy_vec[idx] > 0:
            mirrored[mirror_move_index(idx)] = policy_vec[idx]
    return mirrored


class MirrorPolicyContracts(unittest.TestCase):
    def test_permutation_is_bijective_and_self_inverse(self):
        self.assertEqual(len(_MIRROR_PERM), POLICY_SIZE)
        np.testing.assert_array_equal(np.sort(_MIRROR_PERM), np.arange(POLICY_SIZE))
        np.testing.assert_array_equal(_MIRROR_PERM[_MIRROR_PERM],
                                      np.arange(POLICY_SIZE))

    def test_matches_the_reference_on_sparse_and_dense_vectors(self):
        rng = np.random.default_rng(0)
        vectors = [np.zeros(POLICY_SIZE, dtype=np.float32)]
        for nnz in (1, 5, 30, 200):
            v = np.zeros(POLICY_SIZE, dtype=np.float32)
            v[rng.choice(POLICY_SIZE, nnz, replace=False)] = rng.random(nnz)
            vectors.append(v)
        vectors.append(rng.random(POLICY_SIZE).astype(np.float32))
        for i, v in enumerate(vectors):
            np.testing.assert_array_equal(
                mirror_policy(v), _reference_mirror_policy(v),
                err_msg=f"vector {i}")

    def test_mirroring_twice_is_the_identity(self):
        rng = np.random.default_rng(1)
        v = np.zeros(POLICY_SIZE, dtype=np.float32)
        v[rng.choice(POLICY_SIZE, 40, replace=False)] = rng.random(40)
        np.testing.assert_array_equal(mirror_policy(mirror_policy(v)), v)

    def test_probability_mass_is_preserved(self):
        rng = np.random.default_rng(2)
        v = np.zeros(POLICY_SIZE, dtype=np.float32)
        idx = rng.choice(POLICY_SIZE, 12, replace=False)
        v[idx] = rng.random(12).astype(np.float32)
        v /= v.sum()
        self.assertAlmostEqual(float(mirror_policy(v).sum()), 1.0, places=5)

    def test_negative_entries_are_carried_not_dropped(self):
        """Documented behaviour change: the old loop silently zeroed any
        non-positive entry. Policy targets are probability distributions, so
        this is unreachable in practice — pinned so the difference is a
        recorded decision rather than a surprise."""
        v = np.zeros(POLICY_SIZE, dtype=np.float32)
        v[3] = -0.5
        self.assertEqual(mirror_policy(v)[mirror_move_index(3)], -0.5)


class DiscountedResultSegmentContracts(unittest.TestCase):
    FLOOR, HORIZON = 0.5, 4

    def _run(self, records):
        return _discounted_results(records, horizon=self.HORIZON,
                                   floor=self.FLOOR, mode="near_mate")

    @staticmethod
    def _rec(fen, result=1, **extra):
        return dict(fen=fen, game_result=result, **extra)

    def test_explicit_segment_field_splits_duplicated_games(self):
        recs = ([self._rec(f"fen{i}", segment=0) for i in range(3)]
                + [self._rec(f"fen{i}", segment=1) for i in range(3)])
        out = self._run(recs)
        # each segment ramps independently up to 1.0 at its own final ply
        self.assertAlmostEqual(out[2], 1.0, places=6)
        self.assertAlmostEqual(out[5], 1.0, places=6)
        self.assertAlmostEqual(out[0], out[3], places=6)

    def test_segment_field_beats_the_fen_heuristic(self):
        """A game that revisits its own start position must NOT be split.

        This is the failure the FEN rule cannot see: fen0 recurs mid-game, so
        the fallback would start a bogus second segment there.
        """
        recs = [self._rec("fen0", segment=0), self._rec("fenA", segment=0),
                self._rec("fen0", segment=0), self._rec("fenB", segment=0)]
        out = self._run(recs)
        self.assertAlmostEqual(out[3], 1.0, places=6)
        # one segment of 4 => strictly increasing toward the end
        self.assertTrue(all(a < b for a, b in zip(out, out[1:])), out)

        without_field = [{k: v for k, v in r.items() if k != "segment"}
                         for r in recs]
        self.assertNotEqual(self._run(without_field), out,
                            "FEN fallback should split here; that is why the "
                            "explicit field exists")

    def test_fen_fallback_still_works_for_legacy_records(self):
        recs = [self._rec("start"), self._rec("a"),
                self._rec("start"), self._rec("b")]
        out = self._run(recs)
        self.assertAlmostEqual(out[1], 1.0, places=6)
        self.assertAlmostEqual(out[3], 1.0, places=6)

    def test_floor_at_or_above_one_disables_discounting(self):
        recs = [self._rec("a", result=-1), self._rec("b", result=-1)]
        self.assertEqual(
            _discounted_results(recs, horizon=4, floor=1.0, mode="near_mate"),
            [-1, -1])


if __name__ == "__main__":
    unittest.main()
