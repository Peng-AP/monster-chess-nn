import sys
import json
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import reanalyze


class ReanalysisContracts(unittest.TestCase):
    def test_teacher_rows_are_not_reanalyzed_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            base = {
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "current_player": "white",
                "policy": {"a1a2": 1.0},
            }
            teacher = dict(base, source="deep_search_reanalysis")
            path.write_text(
                json.dumps(base) + "\n" + json.dumps(teacher) + "\n",
                encoding="utf-8",
            )
            rows = list(reanalyze.iter_records(directory))
            self.assertEqual(len(rows), 1)
            self.assertNotIn("source", rows[0]["record"])

    def test_js_is_zero_for_identical_policy(self):
        policy = {"a1a2": 0.25, "a1b1": 0.75}
        self.assertAlmostEqual(
            reanalyze.policy_js_divergence(policy, policy), 0.0)

    def test_js_is_symmetric_and_positive(self):
        left = {"a1a2": 1.0}
        right = {"a1b1": 1.0}
        lr = reanalyze.policy_js_divergence(left, right)
        rl = reanalyze.policy_js_divergence(right, left)
        self.assertGreater(lr, 0.0)
        self.assertAlmostEqual(lr, rl)

    def test_action_change_increases_priority(self):
        same = reanalyze.disagreement_score(
            {"a": 0.6, "b": 0.4}, {"a": 0.55, "b": 0.45}, 0.0, 0.0)
        changed = reanalyze.disagreement_score(
            {"a": 0.6, "b": 0.4}, {"a": 0.45, "b": 0.55}, 0.0, 0.0)
        self.assertFalse(same["action_changed"])
        self.assertTrue(changed["action_changed"])
        self.assertGreater(changed["priority"], same["priority"])

    def test_stratified_sample_fills_requested_black_share(self):
        rows = [
            {"path": f"b{i}", "line": 1,
             "record": {"current_player": "black"}} for i in range(10)
        ] + [
            {"path": f"w{i}", "line": 1,
             "record": {"current_player": "white"}} for i in range(10)
        ]
        selected = reanalyze.stratified_sample(rows, 10, 0.6, 7)
        black = sum(row["record"]["current_player"] == "black"
                    for row in selected)
        self.assertEqual(black, 6)


if __name__ == "__main__":
    unittest.main()
