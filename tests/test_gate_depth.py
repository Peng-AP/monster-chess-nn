"""Gate v4 contracts: constants, schedule layout and verdict arithmetic (no GPU)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gate_depth as g


def row(i, a_is_white, result):
    return {"result_for_a": result, "a_is_white": a_is_white, "plies": 60, "task_id": f"t{i}",
            "opening": {"fen": f"fen{i}", "half": False, "turn_count": 8, "complete": True,
                        "history_sha256": f"h{i}", "repetition_sha256": f"r{i}"},
            "game": {"termination": "king_capture" if result else "repetition"}}


def leg(white, black, start=0):
    """white/black: lists of candidate results as that colour (1 win, 0 draw, -1 loss)."""
    rows = [row(start + i, True, r) for i, r in enumerate(white)]
    return rows + [row(start + 1000 + i, False, r) for i, r in enumerate(black)]


def self_rows(white_results, start=0):
    """Self-play par: model A alternates colours; results are from A's perspective."""
    out = []
    for i, white_result in enumerate(white_results):
        a_white = i % 2 == 0
        out.append(row(start + i, a_white, white_result if a_white else -white_result))
    return out


COUNTS = dict(par=8, per_side=4, deep_par=8, deep=8)
PROTO = dict(counts=COUNTS)


def test_constants_match_the_approved_plan():
    assert g.VERSION == "free_sampled_depth_guard_v4"
    assert (g.PRIMARY_SIMS, g.GUARD_SIMS) == (3200, 12800)
    assert g.PRODUCTION_COUNTS == dict(par=400, per_side=200, deep_par=160, deep=160)
    assert (g.GUARD_AGGREGATE_MIN, g.GUARD_SIDE_BAND) == (0.475, 0.10)
    assert g.WORKERS == 8
    options = {a.dest for a in g.parser()._actions}
    assert not options & {"guard_aggregate_min", "guard_side_band", "counts", "sims", "workers"}


def test_schedule_blocks_are_disjoint_and_subsets_are_exact():
    proto = dict(counts=g.PRODUCTION_COUNTS, seed=2_900_000_000, primary_sims=3200, guard_sims=12800)
    legs = g.schedule(proto)
    assert [l["name"] for l in legs] == ["par", "vs_bar", "vs_bar_confirm", "deep_par", "deep_guard"]
    ranges = sorted((l["seed"], l["seed"] + 1000 + l["games"]) for l in legs)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(ranges, ranges[1:]))
    assert all(l["games"] <= 2000 and l["games"] % 2 == 0 for l in legs)
    assert [l["name"] for l in g.schedule(proto, par_only=True)] == ["par", "deep_par"]
    assert [l["name"] for l in g.schedule(proto, external_par=True)] == ["vs_bar", "vs_bar_confirm", "deep_guard"]
    assert {l["sims"] for l in legs if l["name"].startswith("deep")} == {12800}


def test_par_legs_play_the_bar_with_default_search():
    candidate_search = {"c_puct": 2.0, "fpu_reduction": 0.3}
    assert g.leg_players({"role": "par"}, "m", "bar", candidate_search) == ("bar", g.default_search())
    assert g.leg_players({"role": "h2h"}, "m", "bar", candidate_search) == ("m", candidate_search)


def rows_by_leg(guard_white, guard_black):
    return {"vs_bar": leg([1, 1, 0, -1], [1, 1, 1, 0]),
            "vs_bar_confirm": leg([1, 0, 0, 1], [1, 1, 0, 1], start=100),
            "deep_guard": leg(guard_white, guard_black, start=200)}


def pars():
    # 3,200 par: White scores 3/8 of 8 games; deep par: White 4/8.
    return {"par": self_rows([1, -1, -1, 0, -1, 1, -1, 0]),
            "deep_par": self_rows([1, -1, 1, -1, 0, 0, 1, -1], start=500)}


def test_pass_requires_both_the_primary_gate_and_the_guard():
    result = g.score(rows_by_leg([1, 0, 0, -1], [1, 0, 0, 1]), PROTO, pars())
    assert result["primary"]["verdict"] == "PASS"
    assert result["guard"]["verdict"] == "PASS", result["guard"]
    assert result["verdict"] == "PASS"


def test_a_deep_white_collapse_fails_even_when_3200_passes():
    # Epoch15's failure mode: strong Black, White far below deep par.
    result = g.score(rows_by_leg([-1, -1, -1, 0], [1, 1, 1, 1]), PROTO, pars())
    assert result["primary"]["verdict"] == "PASS"
    assert result["guard"]["verdict"] == "FAIL"
    assert any("white below deep par" in f for f in result["guard"]["failures"])
    assert result["verdict"] == "FAIL"


def test_guard_aggregate_floor_is_inclusive_at_the_threshold():
    stats = {"score": 0.475, "sides": {"white": {"n": 4, "score": .5, "se": .1},
                                       "black": {"n": 4, "score": .45, "se": .1}}}
    par = {"n": 8, "sides": {"white": {"score": .5, "se": .1}, "black": {"score": .5, "se": .1}}}
    assert g.guard_verdict(par, stats, COUNTS)["verdict"] == "PASS"
    stats["score"] = 0.4749
    assert g.guard_verdict(par, stats, COUNTS)["verdict"] == "FAIL"


def test_missing_or_short_guard_is_inconclusive_not_fail():
    by_leg = rows_by_leg([1, 0, 0, -1], [1, 0, 0, 1])
    del by_leg["deep_guard"]
    result = g.score(by_leg, PROTO, pars())
    assert result["guard"]["verdict"] == "INCONCLUSIVE"
    assert result["verdict"] == "INCONCLUSIVE"
    short = rows_by_leg([1, 0, 0], [1, 0, 0, 1])
    assert g.score(short, PROTO, pars())["verdict"] == "INCONCLUSIVE"


def test_overall_verdict_table():
    v = lambda s: {"verdict": s}
    assert g.overall_verdict(v("PASS"), v("PASS")) == "PASS"
    assert g.overall_verdict(v("PASS"), v("FAIL")) == "FAIL"
    assert g.overall_verdict(v("FAIL"), v("PASS")) == "FAIL"
    assert g.overall_verdict(v("FAIL"), v("INCONCLUSIVE")) == "INCONCLUSIVE"
