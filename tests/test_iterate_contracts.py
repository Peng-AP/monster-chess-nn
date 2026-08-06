import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import iterate as it


class BootstrapPipelineContracts(unittest.TestCase):
    def test_phase_order_puts_cheap_gate_before_matches(self):
        self.assertLess(it.PHASES.index("offline_gate"),
                        it.PHASES.index("binding_gate"))
        self.assertLess(it.PHASES.index("binding_gate"),
                        it.PHASES.index("self_skew"))
        self.assertEqual(it.PHASES[-1], "promote")

    def test_next_generation_uses_isolated_run_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(it._next_generation(directory), 1)
            os.makedirs(os.path.join(directory, "gen_0003"))
            os.makedirs(os.path.join(directory, "gen_notes"))
            os.makedirs(os.path.join(directory, "gen_0011"))
            self.assertEqual(it._next_generation(directory), 12)

    def test_promoting_multiple_generations_must_be_explicit(self):
        args = argparse.Namespace(
            games=1, sims=1, workers=1, epochs=1, batch_size=1,
            reanalysis_sample=1, reanalysis_keep=1, reanalysis_sims=1,
            offline_positions=1, self_skew_games=1,
            reanalysis_black_fraction=0.5, generations=2,
            promote_on_pass=False, gate_protocol="full",
        )
        with self.assertRaisesRegex(ValueError, "promote-on-pass"):
            it._validate_args(args)

    def test_quick_gate_can_never_promote(self):
        args = argparse.Namespace(
            games=1, sims=1, workers=1, epochs=1, batch_size=1,
            reanalysis_sample=1, reanalysis_keep=1, reanalysis_sims=1,
            offline_positions=1, self_skew_games=1,
            reanalysis_black_fraction=0.5, generations=1,
            promote_on_pass=True, gate_protocol="quick",
        )
        with self.assertRaisesRegex(ValueError, "full"):
            it._validate_args(args)

    def test_champion_resolution_falls_back_to_immutable_v20(self):
        resolved = it._resolve_champion(explicit=it.DEFAULT_CHAMPION)
        self.assertEqual(resolved, it.DEFAULT_CHAMPION.resolve())

    def test_architecture_inference_reads_v20_attention_geometry(self):
        spec = it._checkpoint_spec(it.DEFAULT_CHAMPION)
        self.assertEqual(spec["policy_head"], "attention")
        self.assertEqual(spec["policy_attention_channels"], 64)
        self.assertEqual(spec["input_channels"], 15)
        self.assertEqual(spec["value_head"], "scalar")

    def test_pipeline_never_overwrites_numbered_v20(self):
        source = Path(it.__file__).read_text(encoding="utf-8")
        self.assertNotIn("shutil.copy(candidate, INCUMBENT", source)
        self.assertIn("champion.json", source)

    def test_resume_rejects_configuration_drift(self):
        args = argparse.Namespace(games=100, resume=True, dry_run=False,
                                  through_phase="train", generations=1)
        state = {"config": {"games": 200, "resume": False,
                            "dry_run": False, "through_phase": None,
                            "generations": 1}}
        with self.assertRaisesRegex(ValueError, "resume configuration differs"):
            it._assert_resume_config(args, state)


if __name__ == "__main__":
    unittest.main()
