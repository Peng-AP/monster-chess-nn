"""Elo round-robin fitter: recovers known ratings, stays finite on sweeps, anchors exactly."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]

import elo_tournament as et


def simulate(true, games, seed=7, draw_rate=0.3):
    rng = np.random.default_rng(seed)
    results = []
    names = list(true)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = et.expected(true[a], true[b])
            # Draws take a fixed share; wins split so the expected score is p.
            d = min(draw_rate, 2 * min(p, 1 - p))
            w, dr, l = rng.multinomial(games, [p - d / 2, d, 1 - p - d / 2])
            results.append(dict(a=a, b=b, games=games, wins=int(w), draws=int(dr), losses=int(l)))
    return results


def test_fit_recovers_known_ratings():
    true = {"p0": 1600.0, "p1": 1750.0, "p2": 1900.0, "p3": 2050.0, "p4": 2200.0}
    results = simulate(true, games=4000)
    points = [(r["a"], r["b"], r["wins"] + r["draws"] / 2, r["games"]) for r in results]
    fitted = et.anchored(et.fit(list(true), points), ("p0", 1600.0))
    for n, r in true.items():
        assert abs(fitted[n] - r) < 15, (n, fitted[n], r)


def test_sweeps_stay_finite_and_ordered():
    points = [("a", "b", 40, 40), ("b", "c", 40, 40), ("a", "c", 40, 40)]
    r = et.fit(["a", "b", "c"], points)
    assert all(np.isfinite(v) for v in r.values())
    assert r["a"] > r["b"] > r["c"]


def test_anchor_is_exact_and_even_match_is_level():
    r = et.anchored(et.fit(["x", "v21"], [("x", "v21", 20, 40)]), ("v21", 1600.0))
    assert r["v21"] == 1600.0 and abs(r["x"] - 1600.0) < 1e-6


def test_report_counts_colours_from_both_sides():
    results = [dict(a="v21", b="x", games=4, wins=2, draws=1, losses=1,
                    a_white=dict(games=2, wins=2, draws=0, losses=0),
                    a_black=dict(games=2, wins=0, draws=1, losses=1))]
    rep = et.report(["v21", "x"], results, reps=0)
    row = {s["player"]: s for s in rep["standings"]}
    assert row["v21"]["white_score"] == 1.0 and row["v21"]["black_score"] == 0.25
    assert row["x"]["white_score"] == 0.75 and row["x"]["black_score"] == 0.0
    assert row["v21"]["elo"] == 1600.0 and row["x"]["games"] == 4


def test_schedule_covers_every_pair_once_in_a_fixed_order():
    names = [n for n, _ in et.PLAYERS]
    pairs = et.schedule(names)
    assert len(pairs) == len(set(frozenset(p) for p in pairs)) == len(names) * (len(names) - 1) // 2
    assert pairs == et.schedule(names)
    assert et.ANCHOR[0] in names
