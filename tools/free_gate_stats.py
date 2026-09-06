"""Versioned, captures-only free-opening statistics (CPU-only)."""
import math
from collections import Counter

from match import game_score

SCORING_VERSION = "free_endpoint_equal_color_v2"
AGGREGATE_MIN = 0.50
PER_SIDE_BAND = 0.05


def opening_key(row):
    op = row.get("opening") or {}
    if not isinstance(row.get("a_is_white"), bool):
        raise ValueError("game is missing an explicit color")
    if not op.get("fen") or any(k not in op for k in ("half", "turn_count", "complete")):
        raise ValueError("game is missing opening-state provenance")
    if row.get("result_for_a") not in (-1, -.5, 0, .5, 1):
        raise ValueError("invalid captures-only game result")
    # Early terminal states remain valid endpoints; do not drop them.
    return (row["a_is_white"], op["fen"], bool(op["half"]), int(op["turn_count"]))


def unique_rows(rows):
    seen = {}
    for row in rows:
        seen.setdefault(opening_key(row), row)
    return list(seen.values())


def side_stats(rows):
    scores = [game_score(row["result_for_a"]) for row in rows]
    n = len(scores)
    mean = sum(scores) / n if n else None
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else None
    return {"n": n, "wins": scores.count(1), "draws": scores.count(.5),
            "losses": scores.count(0), "score": mean,
            "se": math.sqrt(variance / n) if variance is not None else None}


def summary(rows):
    sides = {side: side_stats([r for r in rows if r["a_is_white"] == aw])
             for side, aw in (("white", True), ("black", False))}
    w, b = sides["white"], sides["black"]
    score = (w["score"] + b["score"]) / 2 if w["n"] and b["n"] else None
    se = (math.hypot(w["se"], b["se"]) / 2
          if w["se"] is not None and b["se"] is not None else None)
    return {"n": len(rows), "score": score, "se": se, "sides": sides,
            "elo": 400 * math.log10(score / (1 - score)) if score is not None and 0 < score < 1 else None}


def leg_stats(rows, excluded=()):
    excluded = set(excluded)
    unique = unique_rows(rows)
    novel = [r for r in unique if opening_key(r) not in excluded]
    first, conflicts, history_collisions, repetition_collisions = {}, set(), set(), set()
    for row in rows:
        key = opening_key(row)
        old = first.setdefault(key, row)
        if (old["result_for_a"], old["plies"]) != (row["result_for_a"], row["plies"]):
            conflicts.add(key)
        for field, bucket in (("history_sha256", history_collisions),
                              ("repetition_sha256", repetition_collisions)):
            if old["opening"].get(field) != row["opening"].get(field):
                bucket.add(key)
    return {"sampled": summary(rows), "unique": summary(unique),
            "novel": summary(novel), "overlap": len(unique) - len(novel),
            "duplicates": len(rows) - len(unique),
            "incomplete_openings": sum(not r["opening"]["complete"] for r in rows),
            "endpoint_outcome_conflicts": len(conflicts),
            "endpoint_history_collisions": len(history_collisions),
            "endpoint_repetition_collisions": len(repetition_collisions),
            "missing_history_records": sum("history_sha256" not in r["opening"] for r in rows),
            "endings": dict(Counter(r.get("game", {}).get("termination", "unrecorded") for r in rows)),
            "mean_plies": sum(r["plies"] for r in rows) / len(rows) if rows else None}


def covered(stats, target):
    return all(stats["sides"][s]["n"] >= target for s in ("white", "black"))


def verdict(par, legs, target, par_target):
    """Coverage first. Novel confirmation is conditional, not a new win rate.

    Fixed original score floors apply to the full distinct-opening legs. The
    novel subset establishes additional coverage and is reported separately.
    SEs are descriptive (draw-aware), not independence or sequential-CI claims.
    """
    missing, failures, comparisons = [], [], {}
    if not covered(par["unique"], par_target):
        missing.append("bar self-par lacks required per-color coverage")
    for name in ("vs_bar", "vs_bar_confirm"):
        leg = legs.get(name)
        if leg is None:
            missing.append(f"{name} is missing")
            continue
        coverage = leg["novel"] if name.endswith("confirm") else leg["unique"]
        if not covered(coverage, target):
            missing.append(f"{name} lacks required {'unseen ' if name.endswith('confirm') else ''}per-color coverage")
        stats = leg["unique"]
        if stats["score"] is not None and stats["score"] <= AGGREGATE_MIN:
            failures.append(f"{name} aggregate <= {AGGREGATE_MIN}")
        comparisons[name] = {}
        for side in ("white", "black"):
            value, baseline = stats["sides"][side], par["unique"]["sides"][side]
            if value["score"] is None or baseline["score"] is None:
                continue
            delta = value["score"] - baseline["score"]
            comparisons[name][side] = {"delta_from_par": delta,
                "nominal_se": math.hypot(value["se"], baseline["se"])
                if value["se"] is not None and baseline["se"] is not None else None}
            if delta < -PER_SIDE_BAND:
                failures.append(f"{name} {side} below par - {PER_SIDE_BAND}")
    for name, leg in {"par": par, **legs}.items():
        if leg and leg["endpoint_outcome_conflicts"]:
            missing.append(f"{name} has conflicting outcomes at merged endpoints")
    status = "INCONCLUSIVE" if missing else "FAIL" if failures else "PASS"
    return {"verdict": status, "raw_verdict": status, "confirmed": status != "INCONCLUSIVE",
            "eligible": status == "PASS", "inconclusive_reasons": missing,
            "failures": failures, "par_comparisons": comparisons}
