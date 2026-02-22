import os
import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import iterate as it


class IterateContracts(unittest.TestCase):
    def test_black_human_quota_auto_adjustment_enables_human_quota(self):
        args = SimpleNamespace(
            black_human_as_ai=True,
            use_source_quotas=True,
            black_iter_quota_human=None,
            quota_human=0.25,
        )
        q_selfplay, q_human, q_blackfocus, q_humanseed, adjusted = it._apply_black_human_quota_adjustment(
            args,
            black_iter_quota_selfplay=0.45,
            black_iter_quota_human=0.0,
            black_iter_quota_blackfocus=0.35,
            black_iter_quota_humanseed=0.20,
        )
        self.assertTrue(adjusted)
        self.assertAlmostEqual(q_human, 0.25)
        self.assertAlmostEqual(q_selfplay, 0.20)
        self.assertAlmostEqual(q_blackfocus, 0.35)
        self.assertAlmostEqual(q_humanseed, 0.20)

    def test_black_human_quota_auto_adjustment_skips_explicit_override(self):
        args = SimpleNamespace(
            black_human_as_ai=True,
            use_source_quotas=True,
            black_iter_quota_human=0.05,
            quota_human=0.25,
        )
        q_selfplay, q_human, q_blackfocus, q_humanseed, adjusted = it._apply_black_human_quota_adjustment(
            args,
            black_iter_quota_selfplay=0.45,
            black_iter_quota_human=0.05,
            black_iter_quota_blackfocus=0.35,
            black_iter_quota_humanseed=0.20,
        )
        self.assertFalse(adjusted)
        self.assertAlmostEqual(q_human, 0.05)
        self.assertAlmostEqual(q_selfplay, 0.45)
        self.assertAlmostEqual(q_blackfocus, 0.35)
        self.assertAlmostEqual(q_humanseed, 0.20)

    def test_black_human_quota_auto_adjustment_disabled(self):
        args = SimpleNamespace(
            black_human_as_ai=False,
            use_source_quotas=True,
            black_iter_quota_human=None,
            quota_human=0.25,
        )
        q_selfplay, q_human, q_blackfocus, q_humanseed, adjusted = it._apply_black_human_quota_adjustment(
            args,
            black_iter_quota_selfplay=0.45,
            black_iter_quota_human=0.0,
            black_iter_quota_blackfocus=0.35,
            black_iter_quota_humanseed=0.20,
        )
        self.assertFalse(adjusted)
        self.assertAlmostEqual(q_human, 0.0)
        self.assertAlmostEqual(q_selfplay, 0.45)
        self.assertAlmostEqual(q_blackfocus, 0.35)
        self.assertAlmostEqual(q_humanseed, 0.20)

    def test_black_human_quota_auto_adjustment_requires_allow_flag(self):
        args = SimpleNamespace(
            black_human_as_ai=True,
            use_source_quotas=True,
            black_iter_quota_human=None,
            quota_human=0.25,
        )
        q_selfplay, q_human, q_blackfocus, q_humanseed, adjusted = it._apply_black_human_quota_adjustment(
            args,
            black_iter_quota_selfplay=0.45,
            black_iter_quota_human=0.0,
            black_iter_quota_blackfocus=0.35,
            black_iter_quota_humanseed=0.20,
            allow_auto_adjust=False,
        )
        self.assertFalse(adjusted)
        self.assertAlmostEqual(q_human, 0.0)
        self.assertAlmostEqual(q_selfplay, 0.45)
        self.assertAlmostEqual(q_blackfocus, 0.35)
        self.assertAlmostEqual(q_humanseed, 0.20)

    def test_count_human_record_sides_counts_white_and_black(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "sample.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"current_player":"white"}\n')
                f.write('{"current_player":"black"}\n')
                f.write('{"current_player":"white"}\n')
            counts = it._count_human_record_sides(td)
            self.assertEqual(counts["files"], 1)
            self.assertEqual(counts["white"], 2)
            self.assertEqual(counts["black"], 1)

    def test_resolve_blackfocus_filters_defaults_black_iter_to_nonloss(self):
        base, black_iter = it._resolve_blackfocus_filters("any", None)
        self.assertEqual(base, "any")
        self.assertEqual(black_iter, "nonloss")

    def test_resolve_blackfocus_filters_honors_explicit_override(self):
        base, black_iter = it._resolve_blackfocus_filters("any", "win")
        self.assertEqual(base, "any")
        self.assertEqual(black_iter, "win")

    def test_resolve_train_result_filter_auto(self):
        self.assertEqual(it._resolve_train_result_filter("auto", alternating=True, train_side="black"), "nonloss")
        self.assertEqual(it._resolve_train_result_filter("auto", alternating=True, train_side="white"), "any")
        self.assertEqual(it._resolve_train_result_filter("auto", alternating=False, train_side="both"), "any")
        self.assertEqual(it._resolve_train_result_filter("win", alternating=True, train_side="black"), "win")

    def test_apply_promotion_guards_rejects_low_black_score(self):
        args = SimpleNamespace(
            epochs=4,
            min_accept_epochs=0,
            min_accept_black_score=0.55,
            black_survival_games=0,
            black_survival_threshold=0.35,
        )
        gate_info = {
            "accepted": True,
            "candidate_black": {"score": 0.40, "games": 12},
        }
        accepted, out = it._apply_promotion_guards(args, gate_info, accepted=True)
        self.assertFalse(accepted)
        self.assertTrue(out.get("promotion_guard_failed"))
        self.assertTrue(any("candidate_black score" in x for x in out.get("promotion_guard_reasons", [])))

    def test_apply_promotion_guards_rejects_missing_survival(self):
        args = SimpleNamespace(
            epochs=8,
            min_accept_epochs=0,
            min_accept_black_score=0.0,
            black_survival_games=6,
            black_survival_threshold=0.40,
            black_survival_relative=False,
            black_survival_min_delta=0.0,
        )
        gate_info = {"accepted": True}
        accepted, out = it._apply_promotion_guards(args, gate_info, accepted=True)
        self.assertFalse(accepted)
        self.assertTrue(out.get("promotion_guard_failed"))
        self.assertTrue(any("missing black survival score" in x for x in out.get("promotion_guard_reasons", [])))

    def test_apply_promotion_guards_rejects_relative_survival_delta(self):
        args = SimpleNamespace(
            epochs=8,
            min_accept_epochs=0,
            min_accept_black_score=0.0,
            black_survival_games=6,
            black_survival_threshold=0.0,
            black_survival_relative=True,
            black_survival_min_delta=0.05,
        )
        gate_info = {
            "accepted": True,
            "black_survival": {
                "score": 0.18,
                "total_games": 6,
                "baseline_score": 0.20,
                "relative_delta": -0.02,
            },
        }
        accepted, out = it._apply_promotion_guards(args, gate_info, accepted=True)
        self.assertFalse(accepted)
        self.assertTrue(out.get("promotion_guard_failed"))
        self.assertTrue(any("black_survival delta" in x for x in out.get("promotion_guard_reasons", [])))

    def test_apply_promotion_guards_accepts_relative_survival_delta(self):
        args = SimpleNamespace(
            epochs=8,
            min_accept_epochs=0,
            min_accept_black_score=0.0,
            black_survival_games=6,
            black_survival_threshold=0.0,
            black_survival_relative=True,
            black_survival_min_delta=-0.03,
        )
        gate_info = {
            "accepted": True,
            "black_survival": {
                "score": 0.18,
                "total_games": 6,
                "baseline_score": 0.20,
                "relative_delta": -0.02,
            },
        }
        accepted, out = it._apply_promotion_guards(args, gate_info, accepted=True)
        self.assertTrue(accepted)
        self.assertFalse(out.get("promotion_guard_failed"))

    def test_apply_promotion_guards_noop_when_already_rejected(self):
        args = SimpleNamespace(
            epochs=8,
            min_accept_epochs=5,
            min_accept_black_score=0.55,
            black_survival_games=4,
            black_survival_threshold=0.40,
        )
        gate_info = {"accepted": False}
        accepted, out = it._apply_promotion_guards(args, gate_info, accepted=False)
        self.assertFalse(accepted)
        self.assertFalse(out.get("promotion_guard_failed"))
        self.assertEqual(out.get("promotion_guard_reasons"), [])

    def test_should_run_black_survival_true_when_enabled(self):
        args = SimpleNamespace(black_survival_games=6)
        self.assertTrue(it._should_run_black_survival(args))

    def test_should_run_black_survival_false_when_disabled(self):
        args = SimpleNamespace(black_survival_games=0)
        self.assertFalse(it._should_run_black_survival(args))

    def test_resolve_project_path_handles_relative_existing_path(self):
        rel = os.path.join("data", "raw", "human_games")
        resolved = it._resolve_project_path(rel)
        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(os.path.isdir(resolved))


if __name__ == "__main__":
    unittest.main()
