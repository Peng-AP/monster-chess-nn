import json
import tempfile
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


def _game(game_id, source_kind, result_bucket, records, generation=1, mean_policy_entropy=0.5):
    return {
        "game_id": game_id,
        "records": records,
        "is_human": source_kind == "human",
        "is_humanseed": source_kind == "humanseed",
        "is_blackfocus": source_kind == "blackfocus",
        "result_bucket": result_bucket,
        "generation": generation,
        "mean_policy_entropy": float(mean_policy_entropy),
        "source_kind": source_kind,
    }


class DataProcessorContracts(unittest.TestCase):
    def test_load_all_games_blackfocus_filter_fallback_when_keep_ratio_too_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            bf_dir = raw / "nn_gen1_blackfocus"
            bf_dir.mkdir(parents=True, exist_ok=True)

            recs = [_record("white", 1.0) for _ in range(6)]
            for idx in range(2):
                (bf_dir / f"g{idx}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in recs) + "\n",
                    encoding="utf-8",
                )

            games = dp.load_all_games(
                raw_dir=str(raw),
                include_human=False,
                min_blackfocus_plies=0,
                blackfocus_result_filter="nonloss",
                blackfocus_result_min_keep_ratio=0.50,
                max_generation_age=0,
                min_nonhuman_plies=0,
                min_humanseed_policy_entropy=0.0,
                return_summary=False,
            )
            self.assertEqual(len(games), 2)
            self.assertTrue(all(bool(g.get("is_blackfocus")) for g in games))

    def test_load_all_games_blackfocus_filter_keeps_empty_when_floor_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            bf_dir = raw / "nn_gen1_blackfocus"
            bf_dir.mkdir(parents=True, exist_ok=True)

            recs = [_record("white", 1.0) for _ in range(6)]
            for idx in range(2):
                (bf_dir / f"g{idx}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in recs) + "\n",
                    encoding="utf-8",
                )

            games = dp.load_all_games(
                raw_dir=str(raw),
                include_human=False,
                min_blackfocus_plies=0,
                blackfocus_result_filter="nonloss",
                blackfocus_result_min_keep_ratio=0.0,
                max_generation_age=0,
                min_nonhuman_plies=0,
                min_humanseed_policy_entropy=0.0,
                return_summary=False,
            )
            self.assertEqual(len(games), 0)

    def test_load_all_games_blackfocus_filter_keeps_filtered_subset_when_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            bf_dir = raw / "nn_gen1_blackfocus"
            bf_dir.mkdir(parents=True, exist_ok=True)

            win_recs = [_record("white", 1.0) for _ in range(6)]
            loss_recs = [_record("black", -1.0) for _ in range(6)]
            (bf_dir / "g0.jsonl").write_text(
                "\n".join(json.dumps(r) for r in win_recs) + "\n",
                encoding="utf-8",
            )
            (bf_dir / "g1.jsonl").write_text(
                "\n".join(json.dumps(r) for r in win_recs) + "\n",
                encoding="utf-8",
            )
            (bf_dir / "g2.jsonl").write_text(
                "\n".join(json.dumps(r) for r in win_recs) + "\n",
                encoding="utf-8",
            )
            (bf_dir / "g3.jsonl").write_text(
                "\n".join(json.dumps(r) for r in loss_recs) + "\n",
                encoding="utf-8",
            )

            games = dp.load_all_games(
                raw_dir=str(raw),
                include_human=False,
                min_blackfocus_plies=0,
                blackfocus_result_filter="nonloss",
                blackfocus_result_min_keep_ratio=0.50,
                max_generation_age=0,
                min_nonhuman_plies=0,
                min_humanseed_policy_entropy=0.0,
                return_summary=False,
            )
            self.assertEqual(len(games), 1)
            self.assertEqual(int(games[0]["result_bucket"]), -1)

    def test_load_all_games_humanseed_filter_keeps_filtered_subset_when_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            hs_dir = raw / "nn_gen1_humanseed"
            hs_dir.mkdir(parents=True, exist_ok=True)

            win_recs = [_record("white", 1.0) for _ in range(6)]
            loss_recs = [_record("black", -1.0) for _ in range(6)]
            (hs_dir / "g0.jsonl").write_text(
                "\n".join(json.dumps(r) for r in win_recs) + "\n",
                encoding="utf-8",
            )
            (hs_dir / "g1.jsonl").write_text(
                "\n".join(json.dumps(r) for r in win_recs) + "\n",
                encoding="utf-8",
            )
            (hs_dir / "g2.jsonl").write_text(
                "\n".join(json.dumps(r) for r in win_recs) + "\n",
                encoding="utf-8",
            )
            (hs_dir / "g3.jsonl").write_text(
                "\n".join(json.dumps(r) for r in loss_recs) + "\n",
                encoding="utf-8",
            )

            games = dp.load_all_games(
                raw_dir=str(raw),
                include_human=False,
                min_blackfocus_plies=0,
                blackfocus_result_filter="any",
                blackfocus_result_min_keep_ratio=0.25,
                humanseed_result_filter="nonloss",
                humanseed_result_min_keep_ratio=0.50,
                max_generation_age=0,
                min_nonhuman_plies=0,
                min_humanseed_policy_entropy=0.0,
                return_summary=False,
            )
            self.assertEqual(len(games), 1)
            self.assertEqual(int(games[0]["result_bucket"]), -1)
            self.assertEqual(str(games[0]["source_kind"]), "humanseed")

    def test_load_all_games_humanseed_filter_fallback_when_filtered_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            hs_dir = raw / "nn_gen1_humanseed"
            hs_dir.mkdir(parents=True, exist_ok=True)

            recs = [_record("white", 1.0) for _ in range(6)]
            for idx in range(2):
                (hs_dir / f"g{idx}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in recs) + "\n",
                    encoding="utf-8",
                )

            games = dp.load_all_games(
                raw_dir=str(raw),
                include_human=False,
                min_blackfocus_plies=0,
                blackfocus_result_filter="any",
                blackfocus_result_min_keep_ratio=0.25,
                humanseed_result_filter="nonloss",
                humanseed_result_min_keep_ratio=0.50,
                max_generation_age=0,
                min_nonhuman_plies=0,
                min_humanseed_policy_entropy=0.0,
                return_summary=False,
            )
            self.assertEqual(len(games), 2)
            self.assertTrue(all(str(g.get("source_kind")) == "humanseed" for g in games))

    def test_apply_game_retention_policy_drops_old_generations(self):
        games = [
            _game("nn_gen10/g1.jsonl", "selfplay", 0, [_record("white", 0)] * 8, generation=10),
            _game("nn_gen42/g2.jsonl", "selfplay", 0, [_record("white", 0)] * 8, generation=42),
        ]
        kept, summary = dp._apply_game_retention_policy(
            games,
            max_generation_age=4,
            min_nonhuman_plies=0,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["generation"], 42)
        self.assertEqual(int(summary["dropped"]["age"]["games"]), 1)

    def test_apply_game_retention_policy_drops_short_nonhuman_keeps_human(self):
        games = [
            _game("nn_gen5/self_short.jsonl", "selfplay", 0, [_record("white", 0)] * 2, generation=5),
            _game("human_games/h1.jsonl", "human", 0, [_record("white", 0)] * 2, generation=None),
        ]
        kept, summary = dp._apply_game_retention_policy(
            games,
            max_generation_age=0,
            min_nonhuman_plies=4,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["source_kind"], "human")
        self.assertEqual(int(summary["dropped"]["short_nonhuman"]["games"]), 1)

    def test_apply_game_retention_policy_keeps_blackfocus_under_short_nonhuman_filter(self):
        games = [
            _game("nn_gen5/self_short.jsonl", "selfplay", 0, [_record("white", 0)] * 2, generation=5),
            _game("nn_gen5_blackfocus/bf_short.jsonl", "blackfocus", -1, [_record("black", -1)] * 2, generation=5),
        ]
        kept, summary = dp._apply_game_retention_policy(
            games,
            max_generation_age=0,
            min_nonhuman_plies=4,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["source_kind"], "blackfocus")
        self.assertEqual(int(summary["dropped"]["short_nonhuman"]["games"]), 1)
        self.assertEqual(int(summary["dropped"]["short_nonhuman"]["by_source"].get("selfplay", 0)), 1)
        self.assertEqual(int(summary["dropped"]["short_nonhuman"]["by_source"].get("blackfocus", 0)), 0)

    def test_apply_game_retention_policy_drops_low_entropy_humanseed(self):
        games = [
            _game(
                "nn_gen5_humanseed/g1.jsonl",
                "humanseed",
                1,
                [_record("white", 1)] * 8,
                generation=5,
                mean_policy_entropy=0.0,
            ),
            _game(
                "nn_gen5_humanseed/g2.jsonl",
                "humanseed",
                1,
                [_record("white", 1)] * 8,
                generation=5,
                mean_policy_entropy=0.2,
            ),
        ]
        kept, summary = dp._apply_game_retention_policy(
            games,
            max_generation_age=0,
            min_nonhuman_plies=0,
            min_humanseed_policy_entropy=0.05,
        )
        self.assertEqual(len(kept), 1)
        self.assertGreaterEqual(float(kept[0]["mean_policy_entropy"]), 0.05)
        self.assertEqual(int(summary["dropped"]["low_entropy_humanseed"]["games"]), 1)

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

    def test_rebalance_source_quotas_backfills_underfilled_sources(self):
        quotas = {"selfplay": 5, "human": 5, "blackfocus": 0, "humanseed": 0}
        capacities = {"selfplay": 20, "human": 2, "blackfocus": 0, "humanseed": 0}
        rebalanced = dp._rebalance_source_quotas(
            train_cap=10,
            quotas=quotas,
            source_capacity=capacities,
            ratios={"selfplay": 0.5, "human": 0.5, "blackfocus": 0.0, "humanseed": 0.0},
        )
        self.assertEqual(sum(int(v) for v in rebalanced.values()), 10)
        self.assertEqual(int(rebalanced["human"]), 2)
        self.assertEqual(int(rebalanced["selfplay"]), 8)

    def test_rebalance_source_quotas_strict_does_not_backfill_zero_ratio(self):
        quotas = {"selfplay": 5, "human": 0, "blackfocus": 0, "humanseed": 0}
        capacities = {"selfplay": 5, "human": 20, "blackfocus": 0, "humanseed": 0}
        rebalanced = dp._rebalance_source_quotas(
            train_cap=10,
            quotas=quotas,
            source_capacity=capacities,
            ratios={"selfplay": 1.0, "human": 0.0, "blackfocus": 0.0, "humanseed": 0.0},
            allow_zero_ratio_backfill=False,
        )
        self.assertEqual(int(rebalanced["selfplay"]), 5)
        self.assertEqual(int(rebalanced["human"]), 0)
        self.assertLess(sum(int(v) for v in rebalanced.values()), 10)

    def test_estimate_source_capacity_dedupes_humanseed_only(self):
        rec_w = _record("white", 1)
        rec_b = _record("black", -1)
        games = [
            _game("h1", "human", 1, [rec_w, rec_w, rec_b], generation=None),
            _game("h2", "humanseed", 1, [rec_w, rec_w, rec_b], generation=1),
            _game("h3", "humanseed", 1, [rec_w], generation=1),
            _game("s1", "selfplay", 1, [rec_w, rec_b], generation=1),
        ]
        cap = dp._estimate_train_source_capacity(
            train_games=games,
            augment=False,
            human_repeat=8,
            blackfocus_repeat=1,
            humanseed_repeat=1,
            dedupe_positions=True,
        )
        self.assertEqual(int(cap["human"]), 24)  # repeats now apply to human source
        self.assertEqual(int(cap["humanseed"]), 2)  # deduped by position for humanseed
        self.assertEqual(int(cap["selfplay"]), 2)

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
        X, _yv, _yr, _yp, _yl, _ys, _ids, _capped, source_counts, _stats = dp._convert_games_to_arrays(
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

    def test_process_raw_data_emits_processing_summary_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            out = root / "processed"
            nn1 = raw / "nn_gen1"
            nn1.mkdir(parents=True, exist_ok=True)

            # Create a few tiny games with mixed outcomes so split fallback can
            # populate train/val/test without overlap.
            for i in range(6):
                recs = []
                outcome = (-1.0, 0.0, 1.0)[i % 3]
                for p in range(2):
                    player = "white" if p % 2 == 0 else "black"
                    recs.append(_record(player, outcome))
                (nn1 / f"g{i}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in recs) + "\n",
                    encoding="utf-8",
                )

            dp.process_raw_data(
                raw_dir=str(raw),
                output_dir=str(out),
                augment=False,
                keep_generations=None,
                position_budget=None,
                position_budget_max=None,
                seed=7,
                include_human=False,
                min_blackfocus_plies=0,
                blackfocus_result_filter="any",
                max_generation_age=0,
                min_nonhuman_plies=0,
                min_humanseed_policy_entropy=0.0,
                human_repeat=1,
                humanseed_repeat=1,
                blackfocus_repeat=1,
                max_positions=None,
                use_source_quotas=True,
                quota_selfplay=1.0,
                quota_human=0.0,
                quota_blackfocus=0.0,
                quota_humanseed=0.0,
                human_target_mcts_lambda=0.2,
                humanseed_target_mcts_lambda=0.85,
                blackfocus_target_mcts_lambda=0.9,
                selfplay_target_mcts_lambda=1.0,
            )

            summary_path = out / "processing_summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("source_stats", summary)
            self.assertIn("warnings", summary)
            self.assertIn("split_sizes", summary)
            self.assertIn("source_counts", summary)
            self.assertIn("retention", summary)
            self.assertIn("train", summary["source_stats"])
            self.assertIsInstance(summary["warnings"], list)


if __name__ == "__main__":
    unittest.main()
