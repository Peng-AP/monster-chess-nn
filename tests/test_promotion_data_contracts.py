import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import promotion_data


def _record(fen, player, result, policy):
    return {
        "fen": fen,
        "current_player": player,
        "mcts_value": -0.2 if player == "black" else 0.2,
        "game_result": result,
        "policy": policy,
    }


def _write_game(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n",
                    encoding="utf-8")


class PrepareWhiteRunnerGamesTests(unittest.TestCase):
    def test_failed_defense_keeps_values_but_masks_black_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            prepared = Path(tmp) / "prepared"
            records = [
                _record("7k/4P3/8/8/8/8/8/K7 b - - 0 1", "black", 1.0,
                        {"h8h7": 1.0}),
                _record("7k/4P3/8/8/8/8/8/K7 w - - 1 2", "white", 1.0,
                        {"e7e8q": 1.0}),
            ]
            _write_game(raw / "game_00000.jsonl", records)

            summary = promotion_data.prepare_white_runner_games(raw, prepared)

            out = [json.loads(line) for line in
                   (prepared / "game_00000.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["game_result"] for r in out], [1.0, 1.0])
            self.assertEqual(out[0]["policy_weight"], 0.0)
            self.assertEqual(out[1]["policy_weight"], 1.0)
            self.assertEqual(summary["failed_games"], 1)
            self.assertEqual(summary["masked_black_positions"], 1)

    def test_black_win_without_promotion_remains_a_policy_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            prepared = Path(tmp) / "prepared"
            records = [
                _record("7k/4P3/8/8/8/8/8/K7 b - - 0 1", "black", -1.0,
                        {"h8h7": 1.0}),
                _record("8/4Pk2/8/8/8/8/8/K7 w - - 1 2", "white", -1.0,
                        {"e7e8q": 1.0}),
            ]
            _write_game(raw / "game_00000.jsonl", records)

            summary = promotion_data.prepare_white_runner_games(raw, prepared)

            out = [json.loads(line) for line in
                   (prepared / "game_00000.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["policy_weight"] for r in out], [1.0, 1.0])
            self.assertEqual(summary["successful_preventions"], 1)
            self.assertEqual(summary["masked_black_positions"], 0)

    def test_preparation_rejects_black_runner_contamination(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            prepared = Path(tmp) / "prepared"
            rec = _record("8/8/8/8/8/4p3/8/K6k w - - 0 1", "white", 1.0,
                          {"a1a2": 1.0})
            rec["start_source"] = "promo_black_runner"
            _write_game(raw / "game_00000.jsonl", [rec])

            with self.assertRaises(ValueError):
                promotion_data.prepare_white_runner_games(
                    raw, prepared, expected_start_source="promo_white_runner")


if __name__ == "__main__":
    unittest.main()
