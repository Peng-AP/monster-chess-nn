"""The gate protocol's thresholds, pinned without playing a game.

The owner's binding rule is that a threshold is never weakened to let a recipe
through.  These tests are what makes that rule enforceable: they assert the
constants, assert that a per-side collapse fails even when the aggregate looks
healthy (law 8 -- aggregates masking per-side collapse has burned this project
four times), and assert that a rehearsal run cannot report PASS.

The historical case is the dup1 arm (HANDOFF SS4.1), which scored 0.50 against
v17 with a 0.35 Black leg and had to fail on both counts.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import gate  # noqa: E402


def side(score, games=20):
    """A summarize_side-shaped block with the given score."""
    wins = round(score * games)
    return {"games": games, "wins": wins, "losses": games - wins, "draws": 0,
            "score": score, "mean_plies": 200.0,
            "mean_plies_when_won": 200.0, "mean_plies_when_lost": 200.0}


def leg(white, black, games=40):
    return {"games": games,
            "a_score": round((white + black) / 2, 4),
            "a_as_white": side(white, games // 2),
            "a_as_black": side(black, games // 2)}


def healthy():
    return {"vs_v17": leg(0.70, 0.60), "vs_ramp": leg(0.60, 0.55),
            "anchor": leg(0.80, 0.50, games=20)}


class TestThresholds(unittest.TestCase):
    def test_constants_are_the_owners_numbers(self):
        self.assertEqual(gate.PER_SIDE_FLOOR, 0.40)
        self.assertEqual(gate.V17_AGGREGATE_MIN, 0.50)

    def test_no_cli_flag_can_move_a_threshold(self):
        # A threshold reachable from argv is a threshold that gets tuned when a
        # run is disappointing.  Keep them constants.
        source = (ROOT / "tools" / "gate.py").read_text(encoding="utf-8")
        for flag in ("--floor", "--threshold", "--min-side", "--aggregate"):
            self.assertNotIn(flag, source)

    def test_healthy_candidate_passes(self):
        verdict, failures, _ = gate.evaluate_legs(healthy())
        self.assertEqual(verdict, "PASS", failures)
        self.assertEqual(failures, [])


class TestFailures(unittest.TestCase):
    def test_dup1_case_fails_on_both_counts(self):
        # vs_v17 0.50 aggregate with a 0.35 Black leg: the real rejected run.
        legs = healthy()
        legs["vs_v17"] = leg(0.65, 0.35)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("black" in f for f in failures), failures)
        self.assertTrue(any("aggregate" in f for f in failures), failures)

    def test_aggregate_exactly_at_the_line_fails(self):
        # "must beat 0.50" is strict; 0.50 is not beating it.
        legs = healthy()
        legs["vs_v17"] = leg(0.60, 0.40)
        self.assertEqual(legs["vs_v17"]["a_score"], 0.50)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("aggregate" in f for f in failures), failures)

    def test_floor_is_inclusive(self):
        # 0.40 exactly clears the floor; the dup1 anchor leg read exactly 0.40.
        legs = healthy()
        legs["anchor"] = leg(0.80, 0.40, games=20)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "PASS", failures)

    def test_side_collapse_fails_behind_a_healthy_aggregate(self):
        # 0.95 White / 0.30 Black aggregates to a respectable 0.625.
        legs = healthy()
        legs["vs_ramp"] = leg(0.95, 0.30)
        self.assertGreater(legs["vs_ramp"]["a_score"], 0.60)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("vs_ramp black" in f for f in failures), failures)

    def test_every_leg_is_checked_not_just_the_incumbent(self):
        for name in ("vs_v17", "vs_ramp", "anchor"):
            legs = healthy()
            games = legs[name]["a_as_white"]["games"] * 2
            legs[name] = leg(0.30, 0.90, games=games)
            verdict, failures, _ = gate.evaluate_legs(legs)
            self.assertEqual(verdict, "FAIL", name)
            self.assertTrue(any(f"{name} white" in f for f in failures), failures)


class TestTotals(unittest.TestCase):
    def test_totals_span_all_legs_with_a_noise_floor(self):
        _v, _f, totals = gate.evaluate_legs(healthy())
        # 20 + 20 + 10 games per side across the three legs.
        self.assertEqual(totals["black"]["games"], 50)
        self.assertEqual(totals["white"]["games"], 50)
        self.assertAlmostEqual(totals["black"]["se_points"], 3.54, places=2)


class TestRehearsalCannotPass(unittest.TestCase):
    def test_quick_protocol_is_smaller_on_every_leg(self):
        full = {n: g for n, _o, g, _a in gate.FULL_LEGS}
        quick = {n: g for n, _o, g, _a in gate.QUICK_LEGS}
        self.assertEqual(set(full), set(quick))
        for name in full:
            self.assertLess(quick[name], full[name], name)

    def test_full_protocol_is_twenty_per_side_against_both_models(self):
        games = {n: g for n, _o, g, _a in gate.FULL_LEGS}
        self.assertEqual(games["vs_v17"], 40)
        self.assertEqual(games["vs_ramp"], 40)

    def test_rehearsal_verdict_is_never_pass(self):
        source = (ROOT / "tools" / "gate.py").read_text(encoding="utf-8")
        self.assertIn('"verdict": verdict if binding else "REHEARSAL"', source)


if __name__ == "__main__":
    unittest.main()
