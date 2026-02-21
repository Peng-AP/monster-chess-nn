import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import iterate as it


class IterateContracts(unittest.TestCase):
    def test_resolve_blackfocus_filters_defaults_black_iter_to_any(self):
        base, black_iter = it._resolve_blackfocus_filters("any", None)
        self.assertEqual(base, "any")
        self.assertEqual(black_iter, "any")

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
        )
        gate_info = {"accepted": True}
        accepted, out = it._apply_promotion_guards(args, gate_info, accepted=True)
        self.assertFalse(accepted)
        self.assertTrue(out.get("promotion_guard_failed"))
        self.assertTrue(any("missing black survival score" in x for x in out.get("promotion_guard_reasons", [])))

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

    def test_resolve_project_path_handles_relative_existing_path(self):
        rel = os.path.join("data", "raw", "human_games")
        resolved = it._resolve_project_path(rel)
        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(os.path.isdir(resolved))


if __name__ == "__main__":
    unittest.main()
