import sys
import unittest
from pathlib import Path

import chess


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import promotion_probe


class _ScriptedEngine:
    def __init__(self, actions):
        self.actions = list(actions)

    def get_best_action(self, _game, temperature=0.0):
        return self.actions.pop(0), {}, 0.0


class PromotionProbeTests(unittest.TestCase):
    def test_probe_detects_black_already_promoted_from_excess_material(self):
        result = promotion_probe.play_probe_game(
            _ScriptedEngine([]), _ScriptedEngine([]),
            start_fen="3q3k/8/8/8/8/3q4/8/K7 w - - 0 1",
            runner_color="black",
            defender_color="white",
            max_plies=0,
        )

        self.assertTrue(result["runner_promoted"])
        self.assertFalse(result["promotion_prevented"])

    def test_probe_detects_white_promotion_and_black_king_survival(self):
        white = _ScriptedEngine([chess.Move.from_uci("e7e8q")])
        black = _ScriptedEngine([])

        result = promotion_probe.play_probe_game(
            white, black,
            start_fen="7k/4P3/8/8/8/8/8/K7 w - - 0 1",
            runner_color="white",
            defender_color="black",
            max_plies=1,
        )

        self.assertTrue(result["runner_promoted"])
        self.assertFalse(result["promotion_prevented"])
        self.assertTrue(result["defender_king_survived"])
        self.assertEqual(result["plies"], 1)

    def test_summary_keeps_prevention_king_survival_and_score_separate(self):
        report = promotion_probe.summarize_probe_results([
            {"promotion_prevented": True, "defender_king_survived": True,
             "defender_result": 1.0, "plies": 10},
            {"promotion_prevented": False, "defender_king_survived": True,
             "defender_result": -1.0, "plies": 20},
        ])

        self.assertEqual(report["prevention_rate"], 0.5)
        self.assertEqual(report["king_survival_rate"], 1.0)
        self.assertEqual(report["defender_score"], 0.5)
        self.assertEqual(report["mean_plies"], 15.0)

    def test_gate_rejects_king_survival_regression_even_if_prevention_improves(self):
        incumbent = {
            "prevention_rate": 0.5,
            "king_survival_rate": 1.0,
            "defender_score": 0.5,
        }
        candidate = {
            "prevention_rate": 0.7,
            "king_survival_rate": 0.8,
            "defender_score": 0.6,
        }

        gate = promotion_probe.compare_probe_reports(
            candidate, incumbent,
            max_prevention_drop=0.0,
            max_king_survival_drop=0.0,
            max_score_drop=0.05,
        )

        self.assertFalse(gate["passed"])
        self.assertIn("king_survival_rate", gate["failures"])


if __name__ == "__main__":
    unittest.main()
