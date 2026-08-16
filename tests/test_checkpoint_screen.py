import sys
import tempfile
import unittest
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import checkpoint_screen as screen  # noqa: E402


class CheckpointScreenTests(unittest.TestCase):
    def test_ranking_maximizes_worst_color_before_aggregate(self):
        balanced = {
            "minimum_color_delta": 0.01,
            "deltas": {"white": 0.01, "black": 0.02, "aggregate": 0.015},
        }
        white_only = {
            "minimum_color_delta": -0.02,
            "deltas": {"white": 0.30, "black": -0.02, "aggregate": 0.14},
        }
        self.assertGreater(screen.rank_key(balanced), screen.rank_key(white_only))

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
