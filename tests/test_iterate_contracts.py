import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import iterate as it


class BootstrapPipelineContracts(unittest.TestCase):
    def test_phase_order_puts_cheap_gate_before_matches(self):
        self.assertLess(it.PHASES.index("offline_gate"),
                        it.PHASES.index("binding_gate"))
        self.assertLess(it.PHASES.index("binding_gate"),
                        it.PHASES.index("high_fidelity_gate"))
        self.assertLess(it.PHASES.index("high_fidelity_gate"),
                        it.PHASES.index("self_skew"))
        self.assertEqual(it.PHASES[-1], "promote")

    def test_offline_metrics_are_advisory_by_default(self):
        args = it.build_parser().parse_args([])
        architecture = it._checkpoint_spec(it.DEFAULT_CHAMPION)
        paths = it._paths_for_generation(it.DEFAULT_RUN_ROOT, 99)
        command = it._command_plan(
            args, 99, it.DEFAULT_CHAMPION, architecture, paths, [],
        )["offline_gate"]["commands"][0]
        self.assertNotIn("--enforce", command)

        args.reject_on_offline_regression = True
        command = it._command_plan(
            args, 99, it.DEFAULT_CHAMPION, architecture, paths, [],
        )["offline_gate"]["commands"][0]
        self.assertIn("--enforce", command)

    def test_iteration_uses_measured_game_worker_default(self):
        args = it.build_parser().parse_args([])
        self.assertEqual(args.workers, it.DEFAULT_GAME_WORKERS)

    def test_iteration_defaults_to_successful_scratch_recipe(self):
        args = it.build_parser().parse_args([])
        architecture = it._checkpoint_spec(it.DEFAULT_CHAMPION)
        paths = it._paths_for_generation(it.DEFAULT_RUN_ROOT, 99)
        plan = it._command_plan(
            args, 99, it.DEFAULT_CHAMPION, architecture, paths, [])
        train = plan["train"]["commands"][0]
        process = plan["process"]["commands"][0]
        generate = plan["generate"]["commands"][0]
        reanalyze = plan["reanalyze"]["commands"][0]
        self.assertNotIn("--resume-from", train)
        self.assertNotIn("--select-relative-to-resume", train)
        self.assertEqual(train[train.index("--lr") + 1], "0.002")
        self.assertEqual(train[train.index("--epochs") + 1], "30")
        self.assertEqual(train[train.index("--patience") + 1], "10")
        self.assertEqual(train[train.index("--warmup-epochs") + 1], "3")
        self.assertEqual(process[process.index("--value-floor") + 1], "0.5")
        self.assertEqual(process[process.index("--value-horizon") + 1], "60")
        self.assertEqual(
            process[process.index("--min-nonhuman-plies") + 1], "0")
        self.assertEqual(len(plan["process"]["commands"]), 2)
        self.assertEqual(generate[generate.index("--num-games") + 1], "500")
        self.assertEqual(generate[generate.index("--simulations") + 1], "700")
        self.assertEqual(len(plan["generate"]["commands"]), 1)
        self.assertEqual(reanalyze[reanalyze.index("--sample") + 1], "8000")
        self.assertEqual(reanalyze[reanalyze.index("--keep") + 1], "4000")
        self.assertEqual(
            reanalyze[reanalyze.index("--simulations") + 1], "3200")

    def test_book_blocks_are_disjoint_across_every_play_phase(self):
        args = it.build_parser().parse_args([
            "--book",
            "books/gate_mixed_v20_v21_v21b_gen7_gen9_p8_20260816.json",
        ])
        architecture = it._checkpoint_spec(it.DEFAULT_CHAMPION)
        paths = it._paths_for_generation(it.DEFAULT_RUN_ROOT, 99)
        plan = it._command_plan(
            args, 99, it.DEFAULT_CHAMPION, architecture, paths, [])

        def offset(phase):
            command = plan[phase]["commands"][0]
            return int(command[command.index("--book-offset") + 1])

        self.assertEqual(offset("checkpoint_screen"), 0)
        self.assertEqual(offset("binding_gate"), 120)
        self.assertEqual(offset("high_fidelity_gate"), 340)
        self.assertEqual(offset("self_skew"), 380)

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
            offline_positions=1, self_skew_games=2,
            checkpoint_screen_games=2, checkpoint_screen_sims=1,
            high_fidelity_games=2, high_fidelity_sims=1,
            reanalysis_black_fraction=0.5, generations=2,
            promote_on_pass=False, gate_protocol="full",
        )
        with self.assertRaisesRegex(ValueError, "promote-on-pass"):
            it._validate_args(args)

    def test_data_only_iterations_can_continue_explicitly(self):
        args = argparse.Namespace(
            games=1, sims=1, workers=1, epochs=1, batch_size=1,
            reanalysis_sample=1, reanalysis_keep=1, reanalysis_sims=1,
            offline_positions=1, self_skew_games=2,
            checkpoint_screen_games=2, checkpoint_screen_sims=1,
            high_fidelity_games=2, high_fidelity_sims=1,
            reanalysis_black_fraction=0.5, replay_balance_alpha=0.5,
            generations=2, promote_on_pass=False,
            continue_after_reject=True, gate_protocol="full",
        )
        it._validate_args(args)

    def test_quick_gate_can_never_promote(self):
        args = argparse.Namespace(
            games=1, sims=1, workers=1, epochs=1, batch_size=1,
            reanalysis_sample=1, reanalysis_keep=1, reanalysis_sims=1,
            offline_positions=1, self_skew_games=2,
            checkpoint_screen_games=2, checkpoint_screen_sims=1,
            high_fidelity_games=2, high_fidelity_sims=1,
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

    def test_accepted_data_survives_candidate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            processed.mkdir()
            np.save(processed / "positions.npy",
                    np.zeros((3, 8, 8, 15), np.float32))
            np.save(processed / "policies.npy",
                    np.zeros((3, 4096), np.float32))
            for name in ("mcts_values.npy", "game_results.npy",
                         "policy_weights.npy", "value_weights.npy"):
                np.save(processed / name, np.zeros((3,), np.float32))
            np.savez(processed / "splits.npz",
                     train=np.array([0]), val=np.array([1]),
                     test=np.array([2]))
            (processed / "split_game_ids.json").write_text(
                json.dumps({"train": [], "val": [], "test": []}),
                encoding="utf-8")
            (processed / "generation_audit.json").write_text(
                json.dumps({"verdict": "PASS"}), encoding="utf-8")
            state = {
                "generation": 1,
                "status": "rejected",
                "incumbent": "models/fresh_start_v20/best_value_net.pt",
                "incumbent_sha256": "abc",
                "paths": {"state": str(root / "gen_0001" / "state.json")},
            }
            paths = {"new_processed": processed}
            accepted = it._accept_generation_data(state, paths, root)
            self.assertEqual(accepted["rows"], 3)
            replay = it._recent_replay_sources(root, 2, 4)
            self.assertEqual(replay, [(1, processed.resolve())])
            registry = json.loads(
                (root / "accepted_data.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["entries"][0]["generation"], 1)
            self.assertEqual(accepted["generation_audit"]["verdict"], "PASS")
            np.savez(processed / "splits.npz",
                     train=np.array([0, 1]), val=np.array([1]),
                     test=np.array([2]))
            with self.assertRaisesRegex(RuntimeError, "was modified"):
                it._recent_replay_sources(root, 2, 4)

    def test_accepted_data_hashes_sparse_policy_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            processed.mkdir()
            np.save(processed / "positions.npy",
                    np.zeros((1, 8, 8, 15), np.float32))
            np.savez(processed / "policies_sparse.npz",
                     indptr=np.array([0, 0], np.int64),
                     indices=np.array([], np.int32),
                     values=np.array([], np.float32), shape=np.array([1, 4096]))
            for name in ("mcts_values.npy", "game_results.npy",
                         "policy_weights.npy", "value_weights.npy"):
                np.save(processed / name, np.zeros((1,), np.float32))
            np.savez(processed / "splits.npz", train=np.array([0]),
                     val=np.array([], np.int64), test=np.array([], np.int64))
            (processed / "split_game_ids.json").write_text(
                json.dumps({"train": [], "val": [], "test": []}),
                encoding="utf-8")
            (processed / "generation_audit.json").write_text(
                json.dumps({"verdict": "PASS"}), encoding="utf-8")
            state = {
                "generation": 1, "incumbent": "bar.pt",
                "incumbent_sha256": "abc",
                "paths": {"state": "iterations/test/state.json"},
            }
            accepted = it._accept_generation_data(
                state, {"new_processed": processed}, root)
            self.assertIn("policies_sparse.npz", accepted["artifact_sha256"])
            self.assertNotIn("policies.npy", accepted["artifact_sha256"])

    def test_generation_quality_rejects_incomplete_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "generation_summary.json"
            summary.write_text(json.dumps({
                "num_games_requested": 100,
                "saved_games": 99,
                "failed_games": 1,
                "timed_out_games": 0,
                "skipped_empty": 0,
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "success rate"):
                it._validate_generation_summaries(
                    {"outputs": [str(summary)]}, minimum_success_rate=1.0)

    def test_generation_and_league_seeds_do_not_overlap(self):
        args = it.build_parser().parse_args([])
        architecture = it._checkpoint_spec(it.DEFAULT_CHAMPION)
        paths = it._paths_for_generation(it.DEFAULT_RUN_ROOT, 99)
        plan = it._command_plan(
            args, 99, it.DEFAULT_CHAMPION, architecture, paths, [])
        commands = plan["generate"]["commands"]

        def seed(command):
            return int(command[command.index("--seed") + 1])

        selfplay = set(range(seed(commands[0]), seed(commands[0]) + args.games))
        for command in commands[1:]:
            count = int(command[command.index("--num-games") + 1])
            league = set(range(seed(command), seed(command) + count))
            self.assertTrue(selfplay.isdisjoint(league))


if __name__ == "__main__":
    unittest.main()
