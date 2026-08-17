"""E5 re-baseline protocol: independent samples and a two-read bar."""
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "rebaseline_tool", ROOT / "tools" / "rebaseline.py")
rebaseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rebaseline)


class TestRebaselineSeedPlan(unittest.TestCase):
    def test_every_pair_and_confirmation_use_disjoint_game_seeds(self):
        plan = rebaseline.build_seed_plan(20260804, pair_count=10, games=40)
        all_sets = []
        for _label, seed in plan:
            seeds = set(rebaseline.match_game_seeds(40, seed))
            self.assertEqual(len(seeds), 40)
            self.assertTrue(all(seeds.isdisjoint(previous)
                                for previous in all_sets))
            all_sets.append(seeds)

    def test_plan_contains_cross_table_plus_bar_confirmation(self):
        plan = rebaseline.build_seed_plan(42, pair_count=10, games=40)
        self.assertEqual(len(plan), 11)
        self.assertEqual(plan[-1][0], "bar_confirmation")


class TestBarAssessment(unittest.TestCase):
    def test_v19_requires_two_winning_reads(self):
        self.assertEqual(rebaseline.assess_bar(0.55, 0.60), "v19")

    def test_v19_b_requires_two_winning_reads(self):
        self.assertEqual(rebaseline.assess_bar(0.45, 0.40), "v19_B")

    def test_split_or_tied_reads_are_inconclusive(self):
        for scores in ((0.55, 0.45), (0.45, 0.55), (0.50, 0.40)):
            with self.subTest(scores=scores):
                self.assertIsNone(rebaseline.assess_bar(*scores))


if __name__ == "__main__":
    unittest.main()
