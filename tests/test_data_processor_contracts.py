import sys
import unittest
from pathlib import Path


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


def _game(game_id, source_kind, result_bucket, records):
    return {
        "game_id": game_id,
        "records": records,
        "is_human": source_kind == "human",
        "is_humanseed": source_kind == "humanseed",
        "is_blackfocus": source_kind == "blackfocus",
        "result_bucket": result_bucket,
        "source_kind": source_kind,
    }


class DataProcessorContracts(unittest.TestCase):
    def test_split_games_has_no_overlap(self):
        games = []
        for i in range(36):
            bucket = (-1, 0, 1)[i % 3]
            games.append(
                _game(
                    game_id=f"g{i}",
                    source_kind="selfplay",
                    result_bucket=bucket,
                    records=[_record("white", bucket)],
                )
            )
        split = dp._split_games_by_result(games, seed=42)
        train_ids = {g["game_id"] for g in split["train"]}
        val_ids = {g["game_id"] for g in split["val"]}
        test_ids = {g["game_id"] for g in split["test"]}
        self.assertTrue(train_ids)
        self.assertTrue(val_ids)
        self.assertTrue(test_ids)
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)

    def test_allocate_source_quotas_sums_to_cap(self):
        train_games = [
            _game("s1", "selfplay", 1, [_record("white", 1)]),
            _game("s2", "selfplay", 0, [_record("white", 0)]),
            _game("h1", "human", -1, [_record("black", -1)]),
        ]
        quotas = dp._allocate_source_quotas(
            train_cap=100,
            ratios={"selfplay": 0.5, "human": 0.25, "blackfocus": 0.15, "humanseed": 0.1},
            train_games=train_games,
        )
        self.assertIsNotNone(quotas)
        self.assertEqual(sum(int(v) for v in quotas.values()), 100)
        self.assertEqual(int(quotas["blackfocus"]), 0)
        self.assertEqual(int(quotas["humanseed"]), 0)

    def test_convert_games_respects_source_quotas(self):
        games = [
            _game(
                "self_a",
                "selfplay",
                1,
                [_record("white", 1), _record("white", 1), _record("white", 1)],
            ),
            _game(
                "hum_a",
                "human",
                -1,
                [_record("black", -1), _record("black", -1), _record("black", -1)],
            ),
        ]
        X, _yv, _yr, _yp, _yl, _ids, _capped, source_counts, _stats = dp._convert_games_to_arrays(
            games=games,
            augment=False,
            human_repeat=3,
            blackfocus_repeat=1,
            humanseed_repeat=1,
            source_quotas={"selfplay": 2, "human": 1, "blackfocus": 0, "humanseed": 0},
        )
        self.assertEqual(len(X), 3)
        self.assertLessEqual(int(source_counts["selfplay"]), 2)
        self.assertLessEqual(int(source_counts["human"]), 1)

    def test_processing_warnings_trigger_on_skew(self):
        warnings = dp._build_processing_warnings(
            train_source_counts={"selfplay": 980, "human": 20, "blackfocus": 0, "humanseed": 0},
            train_source_stats={
                "selfplay": {
                    "count": 980,
                    "mcts_value": {"mean": 0.81},
                    "policy_entropy": {"mean": 0.2},
                },
                "human": {
                    "count": 20,
                    "mcts_value": {"mean": 0.0},
                    "policy_entropy": {"mean": 1.0},
                },
            },
            train_source_quotas={"selfplay": 700, "human": 250, "blackfocus": 25, "humanseed": 25},
        )
        self.assertTrue(warnings)
        self.assertTrue(any("underfilled" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
