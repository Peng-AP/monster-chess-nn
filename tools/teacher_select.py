"""Apply the pre-declared teacher-selection rule (docs/plans/TEACHER_SELECTION_PLAN.md).

Reads the three depth round robins and the value audits, keeps candidates whose
strong-games bias is within +/-0.03, scores each by mean Elo across 1,600 /
6,400 / 12,800 (relative to v29), and compares the top two with a joint
bootstrap of the score difference. A tie (interval includes 0) is broken by
Elo at 12,800. Writes benchmarks/teacher_selection_20261003/selection.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import elo_tournament as et  # noqa: E402
import teacher_rr as trr  # noqa: E402

DEPTHS = (1600, 6400, 12800)
BIAS_LIMIT = 0.03
AUDITS = ["benchmarks/value_colour_audit_20261001/report.json",
          "benchmarks/value_colour_audit_20261003_lr/report.json"]
AUDIT_NAME = {"v29": "v29", "gen52R": "gen52R", "gen52LR": "gen52LR", "gen52L": "gen52L", "gen52C": "gen52C"}
REPS = 1000


def bias():
    out = {}
    for path in AUDITS:
        models = json.load(open(os.path.join(ROOT, path), encoding="utf-8"))["models"]
        for name, audit_name in AUDIT_NAME.items():
            if audit_name in models:
                out[name] = models[audit_name]["strong_games"]["bias"]
    return out


def fit_relative(names, rows):
    r = et.fit(names, [(x["a"], x["b"], x["wins"] + x["draws"] / 2, x["games"]) for x in rows])
    return {n: v - r["v29"] for n, v in r.items()}


def select(depth_rows, biases, reps=REPS, seed=20261003):
    names = [n for n, _ in trr.CANDIDATES]
    point = {d: fit_relative(names, rows) for d, rows in depth_rows.items()}
    score = {n: float(np.mean([point[d][n] for d in depth_rows])) for n in names}
    eligible = [n for n in names if abs(biases[n]) <= BIAS_LIMIT]
    ranked = sorted(eligible, key=lambda n: -score[n])
    rng = np.random.default_rng(seed)
    boot = {n: [] for n in names}
    for _ in range(reps):
        per_depth = []
        for rows in depth_rows.values():
            sample = []
            for x in rows:
                p = np.array([x["wins"], x["draws"], x["losses"]], float)
                w, d, l = rng.multinomial(x["games"], p / p.sum())
                sample.append(dict(a=x["a"], b=x["b"], games=x["games"], wins=w, draws=d, losses=l))
            per_depth.append(fit_relative(names, sample))
        for n in names:
            boot[n].append(np.mean([pd[n] for pd in per_depth]))
    first, second = ranked[0], ranked[1]
    diff = np.array(boot[first]) - np.array(boot[second])
    interval = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    tied = interval[0] <= 0
    deepest = max(depth_rows)
    teacher = (max((first, second), key=lambda n: point[deepest][n]) if tied else first)
    return dict(rule="docs/plans/TEACHER_SELECTION_PLAN.md", bias=biases, bias_limit=BIAS_LIMIT, eligible=eligible,
                elo_by_depth={str(d): {n: round(v, 1) for n, v in point[d].items()} for d in depth_rows},
                score={n: round(v, 1) for n, v in score.items()},
                score_ci95={n: [round(float(np.percentile(boot[n], 2.5)), 1), round(float(np.percentile(boot[n], 97.5)), 1)]
                            for n in names},
                top_two=[first, second], difference_ci95=[round(x, 1) for x in interval], tied=bool(tied),
                tiebreak=f"Elo at {deepest}" if tied else None, recommended_teacher=teacher)


def main():
    depth_rows = {d: trr.results(d) for d in DEPTHS}
    missing = [d for d, rows in depth_rows.items() if len(rows) != 10]
    if missing:
        raise SystemExit(f"round robins incomplete at {missing}")
    result = select(depth_rows, bias())
    with open(os.path.join(trr.ROOT_OUT, "selection.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({k: result[k] for k in ("score", "eligible", "top_two", "difference_ci95", "tied",
                                             "recommended_teacher")}, indent=1))
    print("TEACHER SELECTION COMPLETE: " + result["recommended_teacher"], flush=True)


if __name__ == "__main__":
    main()
