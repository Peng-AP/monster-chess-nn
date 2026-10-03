"""Teacher selection (CPU only): eligibility, scoring, tie rule, seed blocks."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]

import elo_tournament as et
import teacher_rr as trr
import teacher_select as ts
import top_rr_extend as ext
import top_round_robin as top


def synthetic(true, games, seed):
    rng = np.random.default_rng(seed)
    names = [n for n, _ in trr.CANDIDATES]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = et.expected(true[a], true[b])
            w = int(rng.binomial(games, p))
            rows.append(dict(a=a, b=b, games=games, wins=w, draws=0, losses=games - w))
    return rows


BIAS = {"v29": 0.017, "gen52R": -0.005, "gen52LR": -0.007, "gen52L": 0.071, "gen52C": 0.049}


def test_ineligible_candidates_are_never_selected_even_when_strongest():
    true = {"v29": 0, "gen52R": 20, "gen52LR": 10, "gen52L": 300, "gen52C": 300}
    rows = {d: synthetic(true, 2000, d) for d in ts.DEPTHS}
    r = ts.select(rows, BIAS, reps=50)
    assert r["eligible"] == ["v29", "gen52R", "gen52LR"] and r["recommended_teacher"] in ("gen52R", "gen52LR")


def test_clear_winner_and_tie_break_at_the_deepest_depth():
    clear = {d: synthetic({"v29": 0, "gen52R": 200, "gen52LR": 0, "gen52L": 0, "gen52C": 0}, 2000, d) for d in ts.DEPTHS}
    r = ts.select(clear, BIAS, reps=50)
    assert r["recommended_teacher"] == "gen52R" and not r["tied"]
    # Level on average, but LR is the stronger one at 12,800 -> a tie broken toward LR.
    rows = {1600: synthetic({"v29": 0, "gen52R": 30, "gen52LR": 0, "gen52L": -100, "gen52C": -100}, 300, 1),
            6400: synthetic({"v29": 0, "gen52R": 30, "gen52LR": 0, "gen52L": -100, "gen52C": -100}, 300, 2),
            12800: synthetic({"v29": 0, "gen52R": 0, "gen52LR": 60, "gen52L": -100, "gen52C": -100}, 300, 3)}
    r = ts.select(rows, BIAS, reps=200)
    if r["tied"]:
        assert r["recommended_teacher"] == max(r["top_two"], key=lambda n: r["elo_by_depth"]["12800"][n])


def test_seed_blocks_are_disjoint_and_below_2_32():
    blocks = sorted(trr.SEED_BASE.values())
    for a, b in zip(blocks, blocks[1:]):
        assert b - a >= 10 * trr.SEED_STRIDE
    assert blocks[-1] + 10 * trr.SEED_STRIDE < 2 ** 32
    for b in blocks:
        assert not ext.SEED_BASE <= b < ext.SEED_BASE + 9 * top.SEED_STRIDE
        assert not top.SEED_BASE <= b < top.SEED_BASE + 28 * top.SEED_STRIDE


def test_candidates_exist_and_audits_cover_them():
    for _, path in trr.CANDIDATES:
        assert (ROOT / path).exists()
    assert set(ts.bias()) == {n for n, _ in trr.CANDIDATES}
