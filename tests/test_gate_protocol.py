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
    return {gate.BAR: leg(0.60, 0.55), "vs_ramp": leg(0.70, 0.60),
            "anchor": leg(0.80, 0.50, games=20),
            gate.CONFIRM_LEG: leg(0.60, 0.55)}


class TestThresholds(unittest.TestCase):
    def test_constants_are_the_owners_numbers(self):
        self.assertEqual(gate.PER_SIDE_FLOOR, 0.40)
        self.assertEqual(gate.AGGREGATE_MIN, 0.50)

    def test_the_bar_is_the_strongest_model_on_record(self):
        # Owner 2026-08-01: every model must be definitively better than the
        # last. The owner promoted the two-color Wide64 successor as v20, then
        # on 2026-08-06 playtested the gen-5 teacher-3200 epoch-2 candidate
        # ("very strong player") and promoted it as v21. On 2026-08-17 the
        # generation-15 checkpoint was promoted as v23, +130.4 Elo above v22 on
        # a fit over all 15 pairs of the bootstrap chain. This assertion exists
        # to make the bar move only by deliberate edit, never by drift.
        self.assertEqual(gate.BAR, "vs_v23")
        self.assertIn(gate.BAR, gate.AGGREGATE_LEGS)
        self.assertIn("bootstrap_v23", gate.BAR_MODEL)
        self.assertEqual(gate.BAR_MODEL, gate.NUMBERED_INCUMBENT)
        self.assertIn("bootstrap_v23", gate.NUMBERED_INCUMBENT)
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
    def test_losing_to_the_bar_fails_however_good_the_rest_is(self):
        legs = healthy()
        legs[gate.BAR] = leg(0.30, 0.25)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any(gate.BAR in f for f in failures), failures)

    def test_level_with_the_bar_is_not_enough(self):
        # Comfortably past ramp and the anchor, but only level with the bar.
        legs = healthy()
        legs[gate.BAR] = leg(0.55, 0.45)
        self.assertEqual(legs[gate.BAR]["a_score"], 0.50)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any(f"{gate.BAR} aggregate" in f for f in failures), failures)

    def test_an_unplayed_bar_leg_cannot_pass(self):
        legs = healthy()
        del legs[gate.BAR]
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("not played" in f for f in failures), failures)

    def test_confirmation_leg_is_held_to_the_same_bar(self):
        # The first bar leg passed; the replay on fresh openings did not.
        # "Definitively better" means both, or it is one lucky opening set.
        legs = healthy()
        legs[gate.CONFIRM_LEG] = leg(0.50, 0.45)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any(gate.CONFIRM_LEG in f for f in failures), failures)

    def test_confirmation_uses_a_different_opening_seed(self):
        # Replaying the same seeds would confirm nothing -- it is the sampled
        # opening set that moved the same matchup 0.575 -> 0.725.
        self.assertNotEqual(gate.CONFIRM_SEED_OFFSET, 0)
        # Compare against the stride actually used, not a literal 100: the
        # stride is now derived from leg size, and a hardcoded one here would
        # keep passing while testing a formula the gate no longer runs.
        stride = max(100, 2 * (max(g for _n, _o, g in gate.FULL_LEGS) + 1000))
        self.assertNotIn(gate.CONFIRM_SEED_OFFSET,
                         [stride * i for i in range(len(gate.FULL_LEGS))])


class TestFailures(unittest.TestCase):
    def test_dup1_case_fails_on_both_counts(self):
        # 0.50 aggregate with a 0.35 Black leg: the real rejected run.
        legs = healthy()
        legs["vs_ramp"] = leg(0.65, 0.35)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("black" in f for f in failures), failures)
        self.assertTrue(any("aggregate" in f for f in failures), failures)

    def test_aggregate_exactly_at_the_line_fails(self):
        # "must beat 0.50" is strict; 0.50 is not beating it.
        legs = healthy()
        legs["vs_ramp"] = leg(0.60, 0.40)
        self.assertEqual(legs["vs_ramp"]["a_score"], 0.50)
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
        legs[gate.BAR] = leg(0.95, 0.30)
        self.assertGreater(legs[gate.BAR]["a_score"], 0.60)
        verdict, failures, _ = gate.evaluate_legs(legs)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any(f"{gate.BAR} black" in f for f in failures), failures)

    def test_every_leg_is_checked_not_just_the_incumbent(self):
        for name in (gate.BAR, "vs_ramp", "anchor"):
            legs = healthy()
            games = legs[name]["a_as_white"]["games"] * 2
            legs[name] = leg(0.30, 0.90, games=games)
            verdict, failures, _ = gate.evaluate_legs(legs)
            self.assertEqual(verdict, "FAIL", name)
            self.assertTrue(any(f"{name} white" in f for f in failures), failures)


class TestConfirmationTargetsTheBar(unittest.TestCase):
    def test_confirmation_leg_name_matches_the_bar(self):
        # A confirmation that replays a DIFFERENT opponent confirms nothing.
        # This regressed the moment the bar moved from ramp to v19.
        self.assertTrue(gate.CONFIRM_LEG.startswith(gate.BAR), gate.CONFIRM_LEG)

    def test_confirmation_opponent_is_the_bar_opponent(self):
        source = (ROOT / "tools" / "gate.py").read_text(encoding="utf-8")
        self.assertIn("bar_spec = {n: (o, g) for n, o, g in spec}[BAR]", source)
        self.assertNotIn("play(CONFIRM_LEG, SPARRING", source)


class TestConfirmationWiring(unittest.TestCase):
    """run_gate's confirmation branch, without playing games.

    HANDOFF SS10.1: rehearse the whole chain. A branch that only executes for a
    passing candidate would otherwise first run months from now, on the one
    result anybody cares about.

    Everything below is written against gate.BAR / gate.CONFIRM_LEG rather than
    literal leg names, so moving the bar cannot leave these tests asserting
    something that no longer exists.
    """

    def run_with(self, scores):
        """Drive run_gate with a stub match, returning (result, legs played)."""
        import match
        played = []
        spec = {n: o for n, o, _g in gate.QUICK_LEGS}

        def fake_run_match(model_a, model_b, games, sims, seed, *a, **kw):
            name = next((n for n, o in spec.items() if o == model_b), None)
            if name is None:
                raise AssertionError(f"unexpected opponent {model_b!r}")
            if name == gate.BAR and any(s == gate.BAR for s, _ in played):
                name = gate.CONFIRM_LEG
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

    @staticmethod
    def _scores(bar, confirm):
        s = {n: (0.70, 0.60) for n, _o, _g in gate.QUICK_LEGS}
        s["anchor"] = (0.80, 0.50)
        s[gate.BAR] = bar
        s[gate.CONFIRM_LEG] = confirm
        return s

    def test_passing_candidate_gets_a_confirmation_on_a_new_seed(self):
        out, played = self.run_with(self._scores((0.60, 0.55), (0.60, 0.55)))
        names = [n for n, _s in played]
        self.assertIn(gate.CONFIRM_LEG, names)
        self.assertTrue(out["confirmed"])
        seeds = dict(played)
        self.assertNotEqual(seeds[gate.CONFIRM_LEG], seeds[gate.BAR])
        self.assertEqual(out["raw_verdict"], "PASS")

    def test_confirmation_replays_the_bar_opponent_not_another_one(self):
        # The bug this catches: bar moves, confirmation keeps replaying the old
        # opponent, and a candidate is "confirmed" against something it was
        # never required to beat.
        out, played = self.run_with(self._scores((0.60, 0.55), (0.60, 0.55)))
        bar_opponent = {n: o for n, o, _g in gate.QUICK_LEGS}[gate.BAR]
        self.assertIn(gate.CONFIRM_LEG, [n for n, _s in played])
        self.assertEqual(out["legs"][gate.CONFIRM_LEG]["name_b"],
                         gate.CONFIRM_LEG)
        self.assertIsNotNone(bar_opponent)

    def test_failing_candidate_does_not_spend_time_confirming(self):
        out, played = self.run_with(self._scores((0.30, 0.25), (0.60, 0.55)))
        self.assertNotIn(gate.CONFIRM_LEG, [n for n, _s in played])
        self.assertFalse(out["confirmed"])
        self.assertEqual(out["raw_verdict"], "FAIL")

    def test_a_failed_confirmation_flips_the_verdict(self):
        out, played = self.run_with(self._scores((0.60, 0.55), (0.40, 0.40)))
        self.assertIn(gate.CONFIRM_LEG, [n for n, _s in played])
        self.assertEqual(out["raw_verdict"], "FAIL")
        self.assertTrue(any(gate.CONFIRM_LEG in f for f in out["failures"]),
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

    def test_the_bar_leg_carries_enough_games_to_decide(self):
        """The deciding leg must resolve better than the noise it is judging.

        At 40 games the per-side floor was checked on 20 (SE 0.112), and on
        2026-08-07 three candidates cleared this gate and then scored 0.4800,
        0.4825 and 0.5019 over 800 games each -- three false positives from
        three passes (REPORT 23). 200 games puts 100 per side, SE 0.05. The
        ramp and anchor legs stay small on purpose: they are floor checks and
        have never been the deciding leg.
        """
        games = {n: g for n, _o, g in gate.FULL_LEGS}
        self.assertGreaterEqual(games[gate.BAR], 200)
        self.assertEqual(games["vs_ramp"], 40)
        self.assertGreater(games[gate.BAR], games["vs_ramp"],
                           "the deciding leg must not be the smallest sample")

    def test_rehearsal_verdict_is_never_pass(self):
        source = (ROOT / "tools" / "gate.py").read_text(encoding="utf-8")
        self.assertIn('"verdict": verdict if binding else "REHEARSAL"', source)


if __name__ == "__main__":
    unittest.main()


class TestLegSeedIsolation(unittest.TestCase):
    """Legs must draw disjoint games, at any leg size.

    run_match derives per-game seeds as leg_seed+i (White) and leg_seed+1000+i
    (Black). A fixed stride of 100 kept today's 40/40/20 legs disjoint by a
    margin of 81, but would have silently overlapped had a leg grown past ~100
    games -- two legs replaying the same openings read as independent
    agreement, which is the one thing the confirmation leg exists to prevent.
    """

    @staticmethod
    def _game_seeds(base, games):
        n_white = games // 2
        n_black = games - n_white
        return (set(base + i for i in range(n_white))
                | set(base + 1000 + i for i in range(n_black)))

    def _assert_disjoint(self, spec):
        stride = max(100, 2 * (max(g for _n, _o, g in spec) + 1000))
        seeds = {name: self._game_seeds(gate.SEED_BASE_FOR_TEST + stride * i, g)
                 for i, (name, _o, g) in enumerate(spec)}
        confirm = self._game_seeds(
            gate.SEED_BASE_FOR_TEST + gate.CONFIRM_SEED_OFFSET, spec[0][2])
        seeds[gate.CONFIRM_LEG] = confirm
        names = list(seeds)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                self.assertEqual(
                    seeds[names[i]] & seeds[names[j]], set(),
                    f"{names[i]} and {names[j]} share per-game seeds")

    def test_current_legs_are_disjoint(self):
        self._assert_disjoint(gate.FULL_LEGS)

    def test_legs_stay_disjoint_when_game_counts_grow(self):
        big = [(name, opp, 400) for name, opp, _g in gate.FULL_LEGS]
        self._assert_disjoint(big)
