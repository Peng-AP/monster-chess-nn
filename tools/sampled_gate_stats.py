"""Fixed-sample free-play scoring; endpoint coverage is a separate diagnostic.

PASS is an operational, point-estimate rule, not a confidence claim that both
colors improved. Repeated endpoints remain observations of the declared opening
distribution. Independent RNG draws can repeat; endpoint identity alone neither
establishes nor refutes independence. All uncertainty estimates are descriptive.
"""
import math

from free_gate_stats import AGGREGATE_MIN, PER_SIDE_BAND, side_stats

SCORING_VERSION = "free_sampled_equal_color_v3"
LEG_NAMES = ("vs_bar", "vs_bar_confirm")


def self_par(rows):
    """Use every self-game for both actual-color estimates, not arbitrary A/B.

    The two estimates are complements, NOT independent samples. There are n
    games, not 2*n. Equal-color self-play aggregate is exactly one half.
    """
    white = [dict(r, result_for_a=r["result_for_a"] if r["a_is_white"]
                  else -r["result_for_a"]) for r in rows]
    black = [dict(r, result_for_a=-r["result_for_a"]) for r in white]
    return {"n": len(rows), "score": .5 if rows else None,
            "sides": {"white": side_stats(white), "black": side_stats(black)},
            "color_estimates_are_complements": True}


def compare(stats, par):
    out = {}
    for color in ("white", "black"):
        value, baseline = stats["sides"][color], par["sides"][color]
        if value["score"] is None or baseline["score"] is None:
            continue
        se = (math.hypot(value["se"], baseline["se"])
              if value["se"] is not None and baseline["se"] is not None else None)
        delta = value["score"] - baseline["score"]
        out[color] = {"delta_from_par": delta, "nominal_se": se,
                      "nominal_95_interval": [delta - 1.96 * se, delta + 1.96 * se]
                      if se is not None else None}
    return out


def verdict(par, legs, target_per_side, par_games):
    """Exact preregistered sample counts; no outcome/novelty-based extensions."""
    missing, failures, comparisons = [], [], {}
    if par["n"] != par_games:
        missing.append("self-par does not have the declared game count")
    for name in LEG_NAMES:
        leg = legs.get(name)
        if leg is None:
            missing.append(f"{name} is missing")
            continue
        stats = leg["sampled"]
        if any(stats["sides"][s]["n"] != target_per_side for s in ("white", "black")):
            missing.append(f"{name} does not have the declared per-color sample count")
        comparisons[name] = compare(stats, par)
        if stats["score"] is not None and stats["score"] <= AGGREGATE_MIN:
            failures.append(f"{name} aggregate <= {AGGREGATE_MIN}")
        for color, comparison in comparisons[name].items():
            if comparison["delta_from_par"] < -PER_SIDE_BAND - 1e-12:
                failures.append(f"{name} {color} below par - {PER_SIDE_BAND}")
    status = "INCONCLUSIVE" if missing else "FAIL" if failures else "PASS"
    return {"verdict": status, "raw_verdict": status,
            "confirmed": status != "INCONCLUSIVE", "eligible": status == "PASS",
            "inconclusive_reasons": missing, "failures": failures,
            "par_comparisons": comparisons,
            "interpretation": "point-estimate gate; not proof of per-color improvement"}
