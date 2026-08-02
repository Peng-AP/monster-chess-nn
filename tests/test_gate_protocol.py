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
    """A candidate that has cleared everything, confirmation leg included."""
    return {"vs_ramp": leg(0.60, 0.55), "vs_v17": leg(0.70, 0.60),
            "anchor": leg(0.80, 0.50, games=20),
            "vs_ramp_confirm": leg(0.60, 0.55)}


class TestThresholds(unittest.TestCase):
    def test_constants_are_the_owners_numbers(self):
        self.assertEqual(gate.PER_SIDE_FLOOR, 0.40)
        self.assertEqual(gate.AGGREGATE_MIN, 0.50)

    def test_the_bar_is_ramp_not_the_incumbent(self):
        # Owner 2026-08-01: every model must be definitively better than the
        # last, and the last is ramp -- not v17, which merely holds the number.
        self.assertEqual(gate.BAR, "vs_ramp")
        self.assertIn("vs_ramp", gate.AGGREGATE_LEGS)
        self.assertIn("fresh_start_v18_ramp", gate.SPARRING)

    def test_the_bar_leg_is_played_first(self):
        # If a run dies partway, the leg that decides should already be in.
        self.assertEqual(gate.FULL_LEGS[0][0], gate.BAR)

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


class TestTheBar(unittest.TestCase):
    def test_losing_to_ramp_fails_however_good_the_rest_is(self):
        # v17's own numbers: it beats the anchor and ties itself, and scores
        # 0.275 against ramp. Under the owner's rule that is not a candidate.
        legs = healthy()
        legs["vs_ramp"] = leg(0.30, 0.25)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("vs_ramp" in f for f in failures), failures)

    def test_beating_the_incumbent_is_not_enough(self):
        # Comfortably past v17 and the anchor, but only level with ramp.
        legs = healthy()
        legs["vs_ramp"] = leg(0.55, 0.45)
        self.assertEqual(legs["vs_ramp"]["a_score"], 0.50)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("vs_ramp aggregate" in f for f in failures), failures)

    def test_an_unplayed_bar_leg_cannot_pass(self):
        legs = healthy()
        del legs["vs_ramp"]
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("not played" in f for f in failures), failures)

    def test_confirmation_leg_is_held_to_the_same_bar(self):
        # The first bar leg passed; the replay on fresh openings did not.
        # "Definitively better" means both, or it is one lucky opening set.
        legs = healthy()
        legs["vs_ramp_confirm"] = leg(0.50, 0.45)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("vs_ramp_confirm" in f for f in failures), failures)

    def test_confirmation_uses_a_different_opening_seed(self):
        # Replaying the same seeds would confirm nothing -- it is the sampled
        # opening set that moved the same matchup 0.575 -> 0.725.
        self.assertNotEqual(gate.CONFIRM_SEED_OFFSET, 0)
        self.assertNotIn(gate.CONFIRM_SEED_OFFSET,
                         [100 * i for i in range(len(gate.FULL_LEGS))])


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


class TestConfirmationWiring(unittest.TestCase):
    """run_gate's confirmation branch, without playing games.

    HANDOFF SS10.1: rehearse the whole chain. A branch that only executes for a
    passing candidate would otherwise first run months from now, on the one
    result anybody cares about.
    """

    def run_with(self, scores):
        """Drive run_gate with a stub match, returning (result, seeds played)."""
        import match
        played = []

        def fake_run_match(model_a, model_b, games, sims, seed, *a, **kw):
            name = ("anchor" if model_b is None
                    else "vs_ramp" if "ramp" in model_b else "vs_v17")
            if name == "vs_ramp" and any(s == "vs_ramp" for s, _ in played):
                name = "vs_ramp_confirm"
            played.append((name, seed))
            white, black = scores[name]
            out = leg(white, black, games)
            out["name_a"], out["name_b"] = "cand", name
            return out

        real, match.run_match = match.run_match, fake_run_match
        try:
            return gate.run_gate("models/candidates/x/best_value_net.pt",
                                 protocol="quick"), played
        finally:
            match.run_match = real

    def test_passing_candidate_gets_a_confirmation_on_a_new_seed(self):
        scores = {"vs_ramp": (0.60, 0.55), "vs_v17": (0.70, 0.60),
                  "anchor": (0.80, 0.50), "vs_ramp_confirm": (0.60, 0.55)}
        out, played = self.run_with(scores)
        names = [n for n, _s in played]
        self.assertIn("vs_ramp_confirm", names)
        self.assertTrue(out["confirmed"])
        seeds = dict(played)
        self.assertNotEqual(seeds["vs_ramp_confirm"], seeds["vs_ramp"])
        self.assertEqual(out["raw_verdict"], "PASS")

    def test_failing_candidate_does_not_spend_time_confirming(self):
        scores = {"vs_ramp": (0.30, 0.25), "vs_v17": (0.70, 0.60),
                  "anchor": (0.80, 0.50), "vs_ramp_confirm": (0.60, 0.55)}
        out, played = self.run_with(scores)
        self.assertNotIn("vs_ramp_confirm", [n for n, _s in played])
        self.assertFalse(out["confirmed"])
        self.assertEqual(out["raw_verdict"], "FAIL")

    def test_a_failed_confirmation_flips_the_verdict(self):
        scores = {"vs_ramp": (0.60, 0.55), "vs_v17": (0.70, 0.60),
                  "anchor": (0.80, 0.50), "vs_ramp_confirm": (0.40, 0.40)}
        out, played = self.run_with(scores)
        self.assertIn("vs_ramp_confirm", [n for n, _s in played])
        self.assertEqual(out["raw_verdict"], "FAIL")
        self.assertTrue(any("vs_ramp_confirm" in f for f in out["failures"]),
                        out["failures"])


class TestTotals(unittest.TestCase):
    def test_totals_span_all_legs_with_a_noise_floor(self):
        _v, _f, totals = gate.evaluate_legs(healthy())
        # 20 + 20 + 10 + 20 games per side across bar, v17, anchor, confirm.
        self.assertEqual(totals["black"]["games"], 70)
        self.assertEqual(totals["white"]["games"], 70)
        self.assertAlmostEqual(totals["black"]["se_points"], 4.18, places=2)


class TestRehearsalCannotPass(unittest.TestCase):
    def test_quick_protocol_is_smaller_on_every_leg(self):
        full = {n: g for n, _o, g in gate.FULL_LEGS}
        quick = {n: g for n, _o, g in gate.QUICK_LEGS}
        self.assertEqual(set(full), set(quick))
        for name in full:
            self.assertLess(quick[name], full[name], name)

    def test_full_protocol_is_twenty_per_side_against_both_models(self):
        games = {n: g for n, _o, g in gate.FULL_LEGS}
        self.assertEqual(games["vs_v17"], 40)
        self.assertEqual(games["vs_ramp"], 40)

    def test_rehearsal_verdict_is_never_pass(self):
        source = (ROOT / "tools" / "gate.py").read_text(encoding="utf-8")
        self.assertIn('"verdict": verdict if binding else "REHEARSAL"', source)


if __name__ == "__main__":
    unittest.main()
