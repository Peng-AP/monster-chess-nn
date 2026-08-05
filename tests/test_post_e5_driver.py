"""Contract tests for the post-E5 data ladder."""
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gate  # noqa: E402
import post_e5_driver as driver  # noqa: E402


class TestPostE5Ladder(unittest.TestCase):
    def test_three_arms_change_only_the_documented_data_inputs(self):
        self.assertEqual(list(driver.ARMS), ["base", "e1500", "owner41"])
        self.assertEqual(driver.ARMS["base"]["raw"], driver.REFERENCE_RAW)
        self.assertIn("e1500", driver.ARMS["e1500"]["raw"])
        self.assertIn("owner41", driver.ARMS["owner41"]["raw"])
        self.assertEqual(len({spec["model"] for spec in driver.ARMS.values()}), 3)

    def test_e1500_merge_is_policy_only(self):
        source = (ROOT / "tools" / "post_e5_driver.py").read_text(encoding="utf-8")
        self.assertIn('"--value-weight", "0"', source)
        self.assertIn('"--as", "ps_monster_e1500"', source)
        self.assertIn('"--dedupe-against-base"', source)

    def test_processing_and_training_recipes_are_frozen(self):
        self.assertEqual(driver.PROCESS_RECIPE, [
            "--seed", "42", "--value-floor", "0.5", "--value-horizon", "60",
            "--value-discount-mode", "near_mate", "--channels", "15",
        ])
        for expected in (
                ("--epochs", "30"), ("--patience", "10"),
                ("--batch-size", "256"), ("--lr", "0.002"),
                ("--seed", "42"), ("--target", "game_result"),
                ("--value-head", "scalar"), ("--select-metric", "decisive")):
            flag, value = expected
            self.assertEqual(driver.RECIPE[driver.RECIPE.index(flag) + 1], value)

    def test_binding_gate_uses_the_operating_point_and_current_bar(self):
        self.assertEqual(driver.GATE_SIMS, 3200)
        self.assertEqual(gate.BAR, "vs_v19_B")
        source = (ROOT / "tools" / "post_e5_driver.py").read_text(encoding="utf-8")
        self.assertIn('"--engine", "native", "--sims", str(GATE_SIMS)', source)

    def test_thresholds_remain_owned_by_gate(self):
        source = (ROOT / "tools" / "post_e5_driver.py").read_text(encoding="utf-8")
        for flag in ("--floor", "--threshold", "--min-side", "--aggregate"):
            self.assertNotIn(flag, source)
        self.assertEqual(gate.PER_SIDE_FLOOR, 0.40)
        self.assertEqual(gate.AGGREGATE_MIN, 0.50)


if __name__ == "__main__":
    unittest.main()
