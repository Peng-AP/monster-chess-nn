import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PretrainPromotionGateTests(unittest.TestCase):
    def test_gate_rejects_generated_black_runner_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus" / "promo_races"
            corpus.mkdir(parents=True)
            record = {
                "fen": "8/8/8/8/8/4p3/8/K6k w - - 0 1",
                "current_player": "white",
                "mcts_value": 0.0,
                "policy": {"a1a2": 1.0},
                "policy_weight": 1.0,
                "game_result": 1.0,
                "start_source": "promo_black_runner",
            }
            (corpus / "game_00000.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "tools/pretrain_check.py", str(corpus.parent)],
                cwd=ROOT, capture_output=True, text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("generated Black-runner", result.stdout)

    def test_gate_accepts_white_runner_records_with_explicit_black_policy_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus" / "promo_races"
            corpus.mkdir(parents=True)
            record = {
                "fen": "7k/4P3/8/8/8/8/8/K7 b - - 0 1",
                "current_player": "black",
                "mcts_value": -0.5,
                "policy": {"h8h7": 1.0},
                "policy_weight": 0.0,
                "game_result": 1.0,
                "start_source": "promo_white_runner",
            }
            (corpus / "game_00000.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "tools/pretrain_check.py", str(corpus.parent)],
                cwd=ROOT, capture_output=True, text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("promo policy weights explicit", result.stdout)

    def test_gate_rejects_failed_black_defense_left_as_policy_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus" / "promo_races"
            corpus.mkdir(parents=True)
            record = {
                "fen": "7k/4P3/8/8/8/8/8/K7 b - - 0 1",
                "current_player": "black",
                "mcts_value": -0.5,
                "policy": {"h8h7": 1.0},
                "policy_weight": 1.0,
                "game_result": 1.0,
                "start_source": "promo_white_runner",
            }
            (corpus / "game_00000.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "tools/pretrain_check.py", str(corpus.parent)],
                cwd=ROOT, capture_output=True, text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("promo policy-weight mismatch", result.stdout)


if __name__ == "__main__":
    unittest.main()
