import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import iterate_presets as presets


def _arg_value(args, flag):
    if flag not in args:
        return None
    idx = args.index(flag)
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


class PresetContracts(unittest.TestCase):
    def test_all_expected_presets_exist(self):
        self.assertEqual(set(presets.PRESETS.keys()), {"smoke", "daily", "overnight"})

    def test_presets_include_core_gating_and_caps(self):
        for name, preset in presets.PRESETS.items():
            args = preset["args"]
            self.assertIn("--gate-threshold", args, msg=name)
            self.assertIn("--gate-min-side-score", args, msg=name)
            self.assertIn("--gate-min-other-side", args, msg=name)
            self.assertIn("--max-processed-positions", args, msg=name)
            self.assertIn("--position-budget", args, msg=name)
            self.assertIn("--position-budget-max", args, msg=name)
            self.assertIn("--alternating", args, msg=name)
            self.assertIn("--human-seed-dir", args, msg=name)

            max_positions = int(_arg_value(args, "--max-processed-positions"))
            self.assertGreater(max_positions, 0)
            self.assertLessEqual(max_positions, 500000)

            position_budget = int(_arg_value(args, "--position-budget"))
            position_budget_max = int(_arg_value(args, "--position-budget-max"))
            self.assertGreater(position_budget, 0)
            self.assertGreaterEqual(position_budget_max, position_budget)

    def test_smoke_is_shortest_profile(self):
        smoke = presets.PRESETS["smoke"]["args"]
        daily = presets.PRESETS["daily"]["args"]
        overnight = presets.PRESETS["overnight"]["args"]

        smoke_iters = int(_arg_value(smoke, "--iterations"))
        daily_iters = int(_arg_value(daily, "--iterations"))
        overnight_iters = int(_arg_value(overnight, "--iterations"))
        self.assertLessEqual(smoke_iters, daily_iters)
        self.assertLessEqual(daily_iters, overnight_iters)

        smoke_games = int(_arg_value(smoke, "--games"))
        daily_games = int(_arg_value(daily, "--games"))
        overnight_games = int(_arg_value(overnight, "--games"))
        self.assertLessEqual(smoke_games, daily_games)
        self.assertLessEqual(daily_games, overnight_games)


if __name__ == "__main__":
    unittest.main()
