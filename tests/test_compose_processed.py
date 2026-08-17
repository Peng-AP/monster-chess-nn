import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compose_processed


def make_source(path, base):
    path.mkdir()
    n = 3
    np.save(path / "positions.npy", np.full((n, 8, 8, 2), base, np.float32))
    np.save(path / "mcts_values.npy", np.arange(n, dtype=np.float32))
    np.save(path / "game_results.npy", np.zeros(n, np.float32))
    np.save(path / "policies.npy", np.full((n, 4), base, np.float32))
    np.save(path / "policy_weights.npy", np.ones(n, np.float32))
    np.save(path / "value_weights.npy", np.ones(n, np.float32))
    np.savez(path / "splits.npz",
             train=np.array([0], np.int64),
             val=np.array([1], np.int64),
             test=np.array([2], np.int64))
    with open(path / "split_game_ids.json", "w", encoding="utf-8") as handle:
        json.dump({"train": ["a"], "val": ["b"], "test": ["c"]}, handle)


class ProcessedCompositionContracts(unittest.TestCase):
    def test_concat_preserves_values_and_shifts_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one, two, out = root / "one", root / "two", root / "out"
            make_source(one, 1)
            make_source(two, 2)
            sources, arrays = compose_processed.inspect_sources(
                [f"one={one}", f"two={two}"])
            manifest = compose_processed.compose(sources, arrays, str(out), 2)
            positions = np.load(out / "positions.npy")
            self.assertEqual(len(positions), 6)
            self.assertTrue((positions[:3] == 1).all())
            self.assertTrue((positions[3:] == 2).all())
            with np.load(out / "splits.npz") as splits:
                self.assertEqual(splits["train"].tolist(), [0, 3])
                self.assertEqual(splits["val"].tolist(), [1, 4])
                self.assertEqual(splits["test"].tolist(), [2, 5])
            self.assertEqual(manifest["rows"], 6)

    def test_policy_abi_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one, two = root / "one", root / "two"
            make_source(one, 1)
            make_source(two, 2)
            np.save(two / "policies.npy", np.ones((3, 5), np.float32))
            with self.assertRaisesRegex(ValueError, "policy ABI"):
                compose_processed.inspect_sources(
                    [f"one={one}", f"two={two}"])

    def test_policy_only_multiplier_changes_only_value_masked_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / "source", root / "out"
            make_source(source, 1)
            np.save(source / "value_weights.npy",
                    np.array([1, 0, 0], np.float32))
            np.save(source / "policy_weights.npy",
                    np.array([1, 1, 0], np.float32))
            sources, arrays = compose_processed.inspect_sources(
                [f"source={source}"])
            sources[0]["policy_only_multiplier"] = 4.0
            compose_processed.compose(sources, arrays, str(out), 2)
            self.assertEqual(
                np.load(out / "policy_weights.npy").tolist(), [1, 4, 0])

    def test_balancing_smooths_general_strata_without_growing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            n = 12
            positions = np.zeros((n, 8, 8, 15), np.float32)
            positions[:, 0, 0, 12] = -1
            positions[-3:, 0, 0, 12] = 1
            for row in range(n):
                positions[row, :, :, :min(12, row + 1)] = 0
                for piece in range(min(12, row + 1)):
                    positions[row, piece // 8, piece % 8, piece] = 1
            np.save(root / "positions.npy", positions)
            captures = np.ones(n, np.float32)
            captures[-3:] = np.array([1, 0, -1], np.float32)
            np.save(root / "capture_results.npy", captures)
            indices = np.arange(n, dtype=np.int64)
            balanced, report = compose_processed.balance_training_indices(
                str(root), indices, alpha=0.5, seed=7, chunk_rows=4)
            self.assertEqual(len(balanced), len(indices))
            self.assertTrue(report["enabled"])
            source = [row["source_rows"] for row in report["strata"]
                      if row["source_rows"]]
            sampled = [row["sampled_rows"] for row in report["strata"]
                       if row["source_rows"]]
            self.assertLessEqual(max(sampled) - min(sampled),
                                 max(source) - min(source))


if __name__ == "__main__":
    unittest.main()
