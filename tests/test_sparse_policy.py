"""Parity tests for sparse policy storage.

The format change is only safe if it is INVISIBLE: every row the trainer reads
must be bit-identical to what the dense array held. A silent divergence here
would not crash -- it would quietly train on different targets and show up as
an unexplained strength regression weeks later. So these tests compare against
dense ground truth exactly, with no tolerance.
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import sparse_policy  # noqa: E402


def make_dense(rows=97, width=4096, seed=0, density=6):
    """A dense policy block shaped like the real thing: a handful of non-zero
    entries per row, some rows empty, values summing to about one."""
    rng = np.random.default_rng(seed)
    dense = np.zeros((rows, width), dtype=np.float32)
    for row in range(rows):
        n = int(rng.integers(0, density + 1))       # 0 exercises empty rows
        if not n:
            continue
        cols = rng.choice(width, n, replace=False)
        values = rng.random(n).astype(np.float32)
        dense[row, cols] = values / values.sum()
    return dense


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.dense = make_dense()
        builder = sparse_policy.Builder(self.dense.shape[1])
        builder.add_dense(self.dense)
        self.sparse = builder.build()

    def test_to_dense_is_bit_identical(self):
        np.testing.assert_array_equal(self.sparse.to_dense(), self.dense)

    def test_shape_len_ndim_and_dtype_match_the_dense_array(self):
        self.assertEqual(self.sparse.shape, self.dense.shape)
        self.assertEqual(len(self.sparse), len(self.dense))
        self.assertEqual(self.sparse.ndim, self.dense.ndim)
        self.assertEqual(self.sparse.dtype, self.dense.dtype)

    def test_fancy_indexing_matches(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            idx = rng.choice(len(self.dense), 24, replace=True)
            np.testing.assert_array_equal(self.sparse[idx], self.dense[idx])

    def test_shuffled_and_repeated_rows_match(self):
        # The trainer indexes with shuffled epoch orders, and balanced replay
        # can repeat a row; both must behave like the dense array.
        idx = np.array([5, 5, 0, len(self.dense) - 1, 3, 3, 3])
        np.testing.assert_array_equal(self.sparse[idx], self.dense[idx])

    def test_slice_int_and_boolean_indexing_match(self):
        np.testing.assert_array_equal(self.sparse[3:29], self.dense[3:29])
        np.testing.assert_array_equal(self.sparse[11], self.dense[11])
        mask = np.zeros(len(self.dense), dtype=bool)
        mask[::7] = True
        np.testing.assert_array_equal(self.sparse[mask], self.dense[mask])

    def test_empty_selection_keeps_the_dense_shape(self):
        out = self.sparse[np.array([], dtype=np.int64)]
        self.assertEqual(out.shape, (0, self.dense.shape[1]))

    def test_all_zero_rows_survive(self):
        empty = np.flatnonzero(self.dense.sum(axis=1) == 0)
        self.assertGreater(len(empty), 0, "fixture should include empty rows")
        np.testing.assert_array_equal(self.sparse[empty], self.dense[empty])

    def test_it_is_actually_smaller(self):
        self.assertLess(self.sparse.nbytes, self.dense.nbytes / 50)


class TestPersistence(unittest.TestCase):
    def test_save_load_round_trip(self):
        dense = make_dense(rows=41, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            builder = sparse_policy.Builder(dense.shape[1])
            builder.add_dense(dense)
            builder.save(directory)
            self.assertTrue(sparse_policy.has_sparse(directory))
            loaded = sparse_policy.load(directory)
            np.testing.assert_array_equal(loaded.to_dense(), dense)

    def test_chunked_writing_equals_one_shot(self):
        dense = make_dense(rows=200, seed=11)
        one = sparse_policy.Builder(dense.shape[1])
        one.add_dense(dense)
        chunked = sparse_policy.Builder(dense.shape[1])
        for start in range(0, len(dense), 32):
            chunked.add_dense(dense[start:start + 32])
        np.testing.assert_array_equal(one.build().to_dense(),
                                      chunked.build().to_dense())

    def test_open_policies_reads_a_legacy_dense_corpus(self):
        dense = make_dense(rows=23, seed=5)
        with tempfile.TemporaryDirectory() as directory:
            np.save(os.path.join(directory, "policies.npy"), dense)
            opened = sparse_policy.open_policies(directory)
            np.testing.assert_array_equal(np.asarray(opened), dense)

    def test_sparse_wins_when_a_stale_dense_file_sits_beside_it(self):
        # A leftover policies.npy must never silently outrank the format
        # written going forward.
        fresh = make_dense(rows=17, seed=1)
        stale = make_dense(rows=17, seed=2)
        with tempfile.TemporaryDirectory() as directory:
            np.save(os.path.join(directory, "policies.npy"), stale)
            builder = sparse_policy.Builder(fresh.shape[1])
            builder.add_dense(fresh)
            builder.save(directory)
            opened = sparse_policy.open_policies(directory)
            self.assertIsInstance(opened, sparse_policy.SparsePolicyTargets)
            np.testing.assert_array_equal(opened.to_dense(), fresh)

    def test_a_corpus_with_neither_representation_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                sparse_policy.open_policies(directory)


class TestConcatenation(unittest.TestCase):
    def _corpus(self, directory, dense, sparse):
        os.makedirs(directory, exist_ok=True)
        if sparse:
            builder = sparse_policy.Builder(dense.shape[1])
            builder.add_dense(dense)
            builder.save(directory)
        else:
            np.save(os.path.join(directory, "policies.npy"), dense)
        return {"name": os.path.basename(directory), "path": directory}

    def test_appending_sparse_to_sparse_matches_dense_concatenation(self):
        a, b = make_dense(rows=31, seed=1), make_dense(rows=19, seed=2)
        builder = sparse_policy.Builder(a.shape[1])
        for block in (a, b):
            inner = sparse_policy.Builder(a.shape[1])
            inner.add_dense(block)
            builder.add_sparse(inner.build())
        np.testing.assert_array_equal(builder.build().to_dense(),
                                      np.concatenate([a, b]))

    def test_compose_mixes_legacy_dense_and_new_sparse_sources(self):
        from compose_processed import _copy_policies
        a, b, c = (make_dense(rows=13, seed=4), make_dense(rows=21, seed=5),
                   make_dense(rows=7, seed=6))
        with tempfile.TemporaryDirectory() as root:
            sources = [
                self._corpus(os.path.join(root, "legacy"), a, sparse=False),
                self._corpus(os.path.join(root, "new"), b, sparse=True),
                self._corpus(os.path.join(root, "older"), c, sparse=False),
            ]
            out = os.path.join(root, "composed")
            os.makedirs(out)
            _copy_policies(sources, out, chunk_rows=4)
            composed = sparse_policy.load(out)
            np.testing.assert_array_equal(composed.to_dense(),
                                          np.concatenate([a, b, c]))

    def test_width_mismatch_is_refused_not_padded(self):
        builder = sparse_policy.Builder(4096)
        with self.assertRaises(ValueError):
            builder.add_dense(make_dense(rows=4, width=4288))


class TestTrainerLoadsSparseCorpora(unittest.TestCase):
    """load_data must return something the trainer can index exactly as before."""

    def _corpus(self, directory, rows=40, width=4096, sparse=True):
        rng = np.random.default_rng(9)
        dense = make_dense(rows=rows, width=width, seed=9)
        np.save(os.path.join(directory, "positions.npy"),
                rng.random((rows, 8, 8, 15)).astype(np.float32))
        np.save(os.path.join(directory, "mcts_values.npy"),
                rng.random(rows).astype(np.float32))
        np.save(os.path.join(directory, "game_results.npy"),
                rng.random(rows).astype(np.float32))
        np.save(os.path.join(directory, "policy_weights.npy"),
                np.ones(rows, dtype=np.float32))
        np.save(os.path.join(directory, "value_weights.npy"),
                np.ones(rows, dtype=np.float32))
        idx = np.arange(rows)
        np.savez(os.path.join(directory, "splits.npz"),
                 train=idx[:30], val=idx[30:36], test=idx[36:])
        if sparse:
            builder = sparse_policy.Builder(width)
            builder.add_dense(dense)
            builder.save(directory)
        else:
            np.save(os.path.join(directory, "policies.npy"), dense)
        return dense

    def _tempdir(self):
        # Windows keeps a memmap's file open until the object is collected, so
        # TemporaryDirectory's strict cleanup races the loader. Tolerant
        # cleanup keeps the test about the data, not the filesystem.
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return directory

    def test_load_data_returns_indexable_policies_for_both_formats(self):
        from train import load_data
        for use_sparse in (True, False):
            directory = self._tempdir()
            dense = self._corpus(directory, sparse=use_sparse)
            loaded = load_data(directory, memory_map=True)
            policies = loaded[3]
            self.assertEqual(policies.shape, dense.shape)
            self.assertEqual(len(policies), len(dense))
            train_idx = loaded[6]["train"]
            np.testing.assert_array_equal(
                np.asarray(policies[train_idx]), dense[train_idx],
                f"sparse={use_sparse} diverged on the train split")

    def test_the_two_formats_load_identically(self):
        from train import load_data
        a, b = self._tempdir(), self._tempdir()
        dense_a = self._corpus(a, sparse=True)
        dense_b = self._corpus(b, sparse=False)
        np.testing.assert_array_equal(dense_a, dense_b)   # same fixture
        rows = np.array([0, 5, 5, 29, 12])
        np.testing.assert_array_equal(
            np.asarray(load_data(a, memory_map=True)[3][rows]),
            np.asarray(load_data(b, memory_map=True)[3][rows]))


class TestCallSitesUseTheSharedReader(unittest.TestCase):
    def test_train_and_compose_no_longer_load_policies_densely(self):
        train = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
        self.assertIn("sparse_policy.open_policies(data_dir", train)
        self.assertNotIn('np.load(os.path.join(data_dir, "policies.npy")',
                         train)
        processor = (ROOT / "src" / "data_processor.py").read_text(
            encoding="utf-8")
        self.assertNotIn('np.save(os.path.join(output_dir, "policies.npy")',
                         processor)
        self.assertIn("sparse_policy.Builder", processor)


if __name__ == "__main__":
    unittest.main()
