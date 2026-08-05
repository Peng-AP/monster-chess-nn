import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import tune_training as tuner  # noqa: E402


class FixedTrial:
    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_int(self, name, low, high):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


def match(black, white, overall):
    return {
        "a_score": overall,
        "a_as_black": {"score": black},
        "a_as_white": {"score": white},
    }


class TrainingTunerContracts(unittest.TestCase):
    def test_fidelity_ladder_is_black_first_and_bounded(self):
        self.assertEqual(
            [(s.epochs, s.games, s.sims) for s in tuner.STAGES],
            [(6, 16, 200), (12, 40, 400), (30, 80, 800)],
        )
        self.assertTrue(all(
            b.seed - a.seed >= 100000
            for a, b in zip(tuner.STAGES, tuner.STAGES[1:])
        ))

    def test_objective_rewards_black_and_penalizes_white_collapse(self):
        baseline = match(0.20, 0.85, 0.525)
        stronger = tuner.arena_objective(
            match(0.30, 0.85, 0.575), baseline)
        collapsed = tuner.arena_objective(
            match(0.30, 0.70, 0.50), baseline)
        self.assertAlmostEqual(stronger["black_delta"], 0.10)
        self.assertGreater(stronger["score"], collapsed["score"])
        self.assertGreater(collapsed["white_shortfall"], 0)

    def test_search_space_is_training_only(self):
        params = tuner.suggest_parameters(FixedTrial())
        self.assertEqual(set(params), {
            "lr", "lr_gamma", "weight_decay", "policy_loss_weight",
            "warmup_epochs", "grad_clip", "batch_size",
        })
        self.assertEqual(set(tuner.BASELINE_PARAMS), set(params))

    def test_training_command_preserves_exact_b_recipe(self):
        params = tuner.suggest_parameters(FixedTrial())
        model_dir = tuner.ROOT / "models" / "tuning" / "fixture"
        command = tuner.training_command(params, tuner.STAGES[0], model_dir)
        joined = " ".join(command)
        self.assertIn(tuner.DATA, joined)
        self.assertIn("--lr-gamma", command)
        self.assertIn("--target game_result", joined)
        self.assertIn("--value-head scalar", joined)
        self.assertIn("--select-metric decisive", joined)
        self.assertNotIn("--black-weight", command)
        self.assertNotIn("--moves-left-head", command)
        self.assertNotIn("--legal-policy-mask", command)


if __name__ == "__main__":
    unittest.main()
