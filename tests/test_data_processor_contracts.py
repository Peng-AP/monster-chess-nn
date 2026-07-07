import json
import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_processor as dp


def _record(current_player, game_result):
    is_white = current_player == "white"
    return {
        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "current_player": current_player,
        "mcts_value": 0.1 if is_white else -0.1,
        "game_result": float(game_result),
        "policy": {"a1a2,a2a3": 1.0} if is_white else {"h1h2": 1.0},
    }


def _write_game(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n",
                    encoding="utf-8")


class LoadAndRetentionTests(unittest.TestCase):
    def test_source_kind_detection_from_subdirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            _write_game(raw / "game_0.jsonl", [_record("black", -1.0)] * 3)
            _write_game(raw / "bf_blackfocus" / "game_0.jsonl", [_record("black", -1.0)] * 3)
            _write_game(raw / "human_games" / "game_0.jsonl", [_record("black", -1.0)] * 3)

            games = dp.load_all_games(str(raw))
            kinds = {g["game_id"]: g["source_kind"] for g in games}
            self.assertEqual(kinds["game_0.jsonl"], "selfplay")
            self.assertEqual(kinds["bf_blackfocus/game_0.jsonl"], "blackfocus")
            self.assertEqual(kinds["human_games/game_0.jsonl"], "human")

    def test_exclude_human_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            _write_game(raw / "game_0.jsonl", [_record("black", -1.0)] * 3)
            _write_game(raw / "human_games" / "game_0.jsonl", [_record("black", -1.0)] * 3)
            games = dp.load_all_games(str(raw), include_human=False)
            self.assertEqual(len(games), 1)
            self.assertEqual(games[0]["source_kind"], "selfplay")

    def test_min_nonhuman_plies_exempts_human(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            _write_game(raw / "game_short.jsonl", [_record("black", -1.0)] * 2)
            _write_game(raw / "human_games" / "game_short.jsonl", [_record("black", -1.0)] * 2)
            games, summary = dp.load_all_games(
                str(raw), min_nonhuman_plies=5, return_summary=True)
            self.assertEqual(len(games), 1)
            self.assertEqual(games[0]["source_kind"], "human")
            self.assertEqual(summary["dropped"]["short_nonhuman"]["games"], 1)

    def test_max_generation_age_drops_old_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            _write_game(raw / "nn_gen1" / "game_0.jsonl", [_record("black", -1.0)] * 3)
            _write_game(raw / "nn_gen5" / "game_0.jsonl", [_record("black", -1.0)] * 3)
            games = dp.load_all_games(str(raw), max_generation_age=2)
            self.assertEqual(len(games), 1)
            self.assertEqual(games[0]["generation"], 5)


class SplitAndConvertTests(unittest.TestCase):
    def _games(self, n, bucket):
        result = float(bucket)
        return [{
            "game_id": f"g{bucket}_{i}.jsonl",
            "records": [_record("black", result)] * 4,
            "source_kind": "selfplay",
            "result_bucket": bucket,
            "generation": None,
        } for i in range(n)]

    def test_split_is_game_level_and_disjoint(self):
        games = self._games(20, -1) + self._games(20, 1)
        split = dp._split_games_by_result(games, seed=42)
        ids = {k: {g["game_id"] for g in v} for k, v in split.items()}
        self.assertFalse(ids["train"] & ids["val"])
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["val"] & ids["test"])
        self.assertEqual(
            len(ids["train"]) + len(ids["val"]) + len(ids["test"]), 40)

    def test_convert_flat_no_weighting(self):
        games = self._games(2, -1)
        X, y_value, y_result, y_policy = dp._convert_games_to_arrays(games, augment=False)
        self.assertEqual(len(X), 8)  # 2 games x 4 records, once each
        self.assertEqual(len(y_value), 8)
        self.assertEqual(len(y_result), 8)
        self.assertEqual(y_policy.shape[1], 4096)

    def test_augment_doubles_positions_and_mirrors_policy(self):
        games = self._games(1, -1)
        X, _v, _r, y_policy = dp._convert_games_to_arrays(games, augment=True)
        self.assertEqual(len(X), 8)  # 4 records x 2 (mirror)
        # h1h2 mirrored -> a1a2
        idx = dp.move_to_index(__import__("chess").Move.from_uci("h1h2"))
        midx = dp.mirror_move_index(idx)
        self.assertGreater(y_policy[1][midx], 0.0)


class ProcessRawDataSmokeTests(unittest.TestCase):
    def test_end_to_end_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "out"
            for i in range(10):
                _write_game(raw / f"game_{i}.jsonl",
                            [_record("black", -1.0 if i % 2 else 1.0)] * 4)
            dp.process_raw_data(raw_dir=str(raw), output_dir=str(out), seed=42)

            X = np.load(out / "positions.npy")
            yr = np.load(out / "game_results.npy")
            with np.load(out / "splits.npz") as splits:
                total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
            self.assertEqual(len(X), 80)  # 10 games x 4 records x 2 augment
            self.assertEqual(len(yr), len(X))
            self.assertEqual(total, len(X))
            meta = json.loads((out / "split_game_ids.json").read_text())
            self.assertEqual(
                len(meta["train"]) + len(meta["val"]) + len(meta["test"]), 10)


if __name__ == "__main__":
    unittest.main()
