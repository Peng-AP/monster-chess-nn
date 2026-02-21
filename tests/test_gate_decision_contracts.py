import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import iterate as it


def _std_gate(score=0.55, white_score=0.55, black_score=0.55):
    return {
        "score": score,
        "total_games": 8,
        "candidate_white": {"score": white_score, "games": 4},
        "candidate_black": {"score": black_score, "games": 4},
    }


def _args(alternating=True):
    return SimpleNamespace(
        arena_games=8,
        arena_sims=24,
        arena_workers=2,
        gate_threshold=0.54,
        gate_min_side_score=0.45,
        gate_min_other_side=0.42,
        alternating=alternating,
        black_focus_arena_games=8,
        black_focus_arena_tier_min=1,
        black_focus_arena_tier_max=5,
        black_focus_gate_threshold=0.40,
        arena_temperature=0.01,
        arena_temperature_moves=0,
    )


class GateDecisionContracts(unittest.TestCase):
    def test_black_training_uses_effective_black_floor_with_focus_arena(self):
        args = _args(alternating=True)

        def _fake_run_arena(**kwargs):
            if kwargs.get("arena_tag") == "black_focus":
                return {
                    "score": 0.70,
                    "total_games": 8,
                    "candidate_white": {"score": 0.45, "games": 4},
                    "candidate_black": {"score": 0.70, "games": 4},
                }
            return _std_gate(score=0.58, white_score=0.62, black_score=0.20)

        with patch("iterate._run_arena", side_effect=_fake_run_arena):
            gate_info, accepted = it._evaluate_candidate_gate(
                args=args,
                candidate_path="cand.pt",
                incumbent_model_path="inc.pt",
                gen=1,
                model_dir="models",
                base_seed=42,
                train_side="black",
                gate_min_other_side_white=0.42,
                gate_min_other_side_black=0.42,
                black_focus_arena_sims=24,
            )
        self.assertTrue(accepted)
        self.assertAlmostEqual(float(gate_info["primary_score"]), 0.70, places=6)
        self.assertAlmostEqual(float(gate_info["black_side_floor_score"]), 0.70, places=6)
        self.assertEqual(gate_info.get("effective_black_gate_source"), "black_focus")

    def test_black_focus_required_for_black_training(self):
        args = _args(alternating=True)

        def _fake_run_arena(**kwargs):
            if kwargs.get("arena_tag") == "black_focus":
                return {
                    "score": 0.45,
                    "total_games": 8,
                    "candidate_white": {"score": 0.50, "games": 4},
                    "candidate_black": {"score": 0.30, "games": 4},
                }
            return _std_gate(score=0.60, white_score=0.62, black_score=0.58)

        with patch("iterate._run_arena", side_effect=_fake_run_arena) as mocked:
            gate_info, accepted = it._evaluate_candidate_gate(
                args=args,
                candidate_path="cand.pt",
                incumbent_model_path="inc.pt",
                gen=1,
                model_dir="models",
                base_seed=42,
                train_side="black",
                gate_min_other_side_white=0.42,
                gate_min_other_side_black=0.42,
                black_focus_arena_sims=24,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertTrue(gate_info["black_focus_required"])
        self.assertFalse(gate_info["black_focus_pass"])
        self.assertFalse(accepted)

    def test_white_training_has_no_black_focus_requirement(self):
        args = _args(alternating=True)

        with patch("iterate._run_arena", return_value=_std_gate(score=0.58, white_score=0.60, black_score=0.50)) as mocked:
            gate_info, accepted = it._evaluate_candidate_gate(
                args=args,
                candidate_path="cand.pt",
                incumbent_model_path="inc.pt",
                gen=1,
                model_dir="models",
                base_seed=42,
                train_side="white",
                gate_min_other_side_white=0.48,
                gate_min_other_side_black=0.42,
                black_focus_arena_sims=24,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertFalse(gate_info["black_focus_required"])
        self.assertTrue(accepted)
        self.assertEqual(gate_info["decision_mode"], "side_aware_strict")

    def test_overall_mode_uses_overall_strict_sides(self):
        args = _args(alternating=False)

        with patch("iterate._run_arena", return_value=_std_gate(score=0.50, white_score=0.70, black_score=0.70)):
            gate_info, accepted = it._evaluate_candidate_gate(
                args=args,
                candidate_path="cand.pt",
                incumbent_model_path="inc.pt",
                gen=1,
                model_dir="models",
                base_seed=42,
                train_side="black",
                gate_min_other_side_white=0.42,
                gate_min_other_side_black=0.42,
                black_focus_arena_sims=24,
            )
        self.assertEqual(gate_info["decision_mode"], "overall_strict_sides")
        self.assertFalse(accepted)


if __name__ == "__main__":
    unittest.main()
