"""Two matches only sample different games if their seeds are far enough apart.

`match.run_match` derives per-game seeds as `seed + i` for the White half and
`seed + 1000 + i` for the Black half. So two runs whose seeds differ by less
than the per-side game count replay *almost the same games*: seeds 20260802 and
20260803 share 19 of 20 White seeds and 19 of 20 Black seeds.

This bit for real on 2026-08-02. A K-vs-B match was re-run at "a fresh seed"
one greater than the first, produced byte-identical scores, and was about to be
pooled as n=80 at 2.7 SE when it was really the same ~41 games twice. Identical
results across "different" seeds is the symptom.

The seed derivation is deliberately NOT changed here: every gate on record
(v17, ramp, control, O, K, B) was run with it, and rederiving would make new
gates incomparable to those. `tools/gate.py` is already safe — its legs are
spaced 100 apart and its confirmation leg by 424242. What was unsafe was
picking ad-hoc seeds by hand.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import gate  # noqa: E402


def game_seeds(seed, games):
    """Mirror run_match's derivation."""
    n_white = games // 2
    white = {seed + i for i in range(n_white)}
    black = {seed + 1000 + i for i in range(games - n_white)}
    return white, black


def overlap(seed_a, seed_b, games):
    wa, ba = game_seeds(seed_a, games)
    wb, bb = game_seeds(seed_b, games)
    return len(wa & wb) + len(ba & bb)


class TestAdjacentSeedsCollide(unittest.TestCase):
    """The hazard, recorded so nobody rediscovers it in a result."""

    def test_seeds_one_apart_share_almost_every_game(self):
        self.assertEqual(overlap(20260802, 20260803, 40), 38)

    def test_separation_must_exceed_the_black_offset(self):
        # Anything closer than 1000 + games/2 can collide across the halves.
        self.assertEqual(overlap(1000, 1000 + 1020, 40), 0)
        self.assertGreater(overlap(1000, 1000 + 19, 40), 0)

    def test_widely_separated_seeds_are_disjoint(self):
        self.assertEqual(overlap(20260802, 20760802, 40), 0)
        self.assertEqual(overlap(20260802, 21260802, 40), 0)


class TestGateLegsAreSafe(unittest.TestCase):
    """The committed driver must not have this problem."""

    def test_every_pair_of_full_protocol_legs_is_disjoint(self):
        base = 20260801
        specs = [(name, games) for name, _opp, games in gate.FULL_LEGS]
        # Read the stride the gate actually uses. Hardcoding it here let this
        # test pass while the bar leg grew past it.
        stride = gate.leg_seed_stride(gate.FULL_LEGS)
        seeds = [(name, base + stride * i, games)
                 for i, (name, games) in enumerate(specs)]
        seeds.append((gate.CONFIRM_LEG, base + gate.CONFIRM_SEED_OFFSET,
                      dict(specs)[gate.BAR]))
        for i, (n1, s1, g1) in enumerate(seeds):
            for n2, s2, g2 in seeds[i + 1:]:
                self.assertEqual(overlap(s1, s2, max(g1, g2)), 0,
                                 f"{n1} and {n2} sample overlapping games")

    def test_confirmation_leg_is_far_from_every_primary_leg(self):
        # The confirmation exists to be an independent opening set; if it
        # overlapped the bar leg it would confirm nothing.
        self.assertGreater(gate.CONFIRM_SEED_OFFSET,
                           1000 + max(g for _n, _o, g in gate.FULL_LEGS))


if __name__ == "__main__":
    unittest.main()
