import sys
import tempfile
import unittest
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import checkpoint_screen as screen  # noqa: E402


def test_screen_interruption_resumes_journals_and_pins_nominee(tmp_path, monkeypatch):
    import pytest
    import worker_lease
    from match import build_tasks
    from match_evidence import MatchJournal, task_id
    monkeypatch.setattr(worker_lease, "DEFAULT_PATH", tmp_path / "workers.lock")
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen, "runtime_identity", lambda: {"engine": "test"})
    model_dir = tmp_path / "candidate"
    model_dir.mkdir()
    torch.save({"weight": torch.ones(2)}, model_dir / "best_value_net.pt")
    incumbent = tmp_path / "bar.pt"
    torch.save({"weight": torch.zeros(2)}, incumbent)
    output = model_dir / "nominee.pt"
    report = tmp_path / "screen.json"
    monkeypatch.setattr(sys, "argv", ["checkpoint_screen", "--model-dir", str(model_dir),
        "--incumbent", str(incumbent), "--output-model", str(output),
        "--report-path", str(report), "--games", "4", "--probe-games", "4"])
    calls, fail_once = [], [True]
    def fake(a, b, games, sims, seed, **kw):
        tasks = build_tasks(games, seed, 16)
        journal = MatchJournal(kw["game_log"], {"a": a, "b": b, "sims": sims}, tasks, resume=True)
        for task in journal.pending:
            aw, game_seed = task[:2]
            calls.append((Path(kw["game_log"]).stem, game_seed))
            journal.append({"a_is_white": aw, "result_for_a": 0 if a == b else 1,
                            "plies": 20, "pair": None, "seed": game_seed,
                            "task_id": task_id(task)})
            if len(calls) == 5 and fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("interrupted")
        score = .5 if a == b else 1.0
        return {"a_score": score, "a_as_white": {"score": score}, "a_as_black": {"score": score}}
    monkeypatch.setattr(screen, "run_match", fake)
    with pytest.raises(RuntimeError, match="interrupted"):
        screen.main()
    screen.main()
    assert len(calls) == len(set(calls)) == 16
    assert output.exists() and report.exists()
    screen.main()
    assert len(calls) == 16
    output.write_bytes(b"changed")
    with pytest.raises(ValueError, match="nominee"):
        screen.main()


class CheckpointScreenTests(unittest.TestCase):
    def test_free_calibration_uses_all_games_and_caps_are_draws(self):
        rows = [{"a_is_white": True, "result_for_a": 1},
                {"a_is_white": True, "result_for_a": .5},
                {"a_is_white": False, "result_for_a": -1},
                {"a_is_white": False, "result_for_a": -.5}]
        original = {"a_score": .5}
        calibration = screen.actual_color_calibration(original, rows)
        self.assertEqual(calibration["a_as_white"]["score"], .75)
        self.assertEqual(calibration["a_as_black"]["score"], .25)
        self.assertEqual(calibration["a_as_white"]["n"], 4)
        self.assertEqual(calibration["a_as_black"]["n"], 4)
        self.assertEqual(calibration["games"], 4)
        self.assertIs(calibration["role_split_match"], original)
        self.assertEqual(calibration["a_score"], .5)

    def test_free_probe_and_final_seed_blocks_are_disjoint(self):
        from match import build_tasks
        probe = {t[1] for t in build_tasks(2000, 100, 16)}
        final = {t[1] for t in build_tasks(2000, screen.final_stage_seed(100), 16)}
        self.assertFalse(probe & final)

    def test_ranking_maximizes_aggregate_once_no_colour_has_collapsed(self):
        """The screen is a SHORTLIST and must not out-rule the gate.

        It previously ranked on the worst colour, which demanded improvement on
        BOTH sides while the gate asks only for aggregate > 0.50 with neither
        side under the 0.40 floor (owner, 2026-08-16). Since Gen9 every arm
        trades a little White for more Black, so the old key nominated by the
        size of the sacrifice and hid candidates the gate might pass.
        """
        balanced = {
            "minimum_color_delta": 0.01,
            "deltas": {"white": 0.01, "black": 0.02, "aggregate": 0.015},
        }
        stronger_aggregate = {
            "minimum_color_delta": -0.02,
            "deltas": {"white": 0.30, "black": -0.02, "aggregate": 0.14},
        }
        self.assertGreater(screen.rank_key(stronger_aggregate),
                           screen.rank_key(balanced))

    def test_a_collapsed_colour_is_ranked_below_everything_intact(self):
        """"Neither side collapses" is the owner's other condition."""
        collapsed = {
            "minimum_color_delta": -0.25,
            "deltas": {"white": 0.60, "black": -0.25, "aggregate": 0.175},
        }
        modest = {
            "minimum_color_delta": -0.01,
            "deltas": {"white": 0.02, "black": -0.01, "aggregate": 0.005},
        }
        self.assertGreater(screen.rank_key(modest), screen.rank_key(collapsed))

    def test_calibration_is_per_color(self):
        calibration = {
            "a_score": 0.50,
            "a_as_white": {"score": 0.70},
            "a_as_black": {"score": 0.30},
        }
        candidate = {
            "a_score": 0.55,
            "a_as_white": {"score": 0.75},
            "a_as_black": {"score": 0.35},
        }
        result = screen.calibrated_result(candidate, calibration)
        self.assertAlmostEqual(result["minimum_color_delta"], 0.05)
        self.assertTrue(result["passes_both_colors"])

    def test_discovery_deduplicates_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            state = {"weight": torch.arange(4, dtype=torch.float32)}
            torch.save(state, model_dir / "selected_epoch_002.pt")
            torch.save(state, model_dir / "best_value_net.pt")
            checkpoints = screen.discover_checkpoints(model_dir)
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(checkpoints[0]["name"], "selected_epoch_002")
            self.assertTrue(checkpoints[0]["offline_selected"])

    def test_finalists_protect_black_best_and_offline_selected(self):
        def result(name, white, black, offline=False):
            return {
                "name": name,
                "minimum_color_delta": min(white, black),
                "deltas": {
                    "white": white,
                    "black": black,
                    "aggregate": (white + black) / 2,
                },
                "offline_selected": offline,
            }

        results = [
            result("balanced", 0.10, 0.10),
            result("runner_up", 0.09, 0.09),
            result("black_best", -0.30, 0.40),
            result("offline", -0.40, -0.40, offline=True),
        ]
        finalists = screen.choose_finalists(results, 2)
        names = {item["name"] for item in finalists}
        self.assertEqual(
            names, {"balanced", "runner_up", "black_best", "offline"})

    def test_shortlist_covers_peak_neighborhood_black_metrics_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            checkpoints = []
            for epoch in range(1, 15):
                checkpoints.append({
                    "name": f"selected_epoch_{epoch:03d}",
                    "weights_sha256": str(epoch),
                    "offline_selected": epoch == 4,
                })
            rows = []
            for epoch in range(1, 15):
                rows.append({
                    "epoch": epoch,
                    "val_decisive": {
                        "policy_top1_black": 1.0 if epoch == 8 else 0.0,
                        "sign_acc_black": 1.0 if epoch == 10 else 0.0,
                    },
                })
            (model_dir / "train_run_test.json").write_text(json.dumps({
                "best_epoch": 4, "epochs": rows,
            }), encoding="utf-8")
            shortlist = screen.shortlist_checkpoints(
                checkpoints, model_dir, maximum=8)
            epochs = {
                int(row["name"].removeprefix("selected_epoch_"))
                for row in shortlist
            }
            self.assertTrue({2, 3, 4, 5, 6, 8, 10, 14} <= epochs)
            self.assertEqual(len(shortlist), 8)


if __name__ == "__main__":
    unittest.main()
