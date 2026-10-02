"""Top-group round robin: is any candidate stronger than v29?

docs/plans/OVERNIGHT_20261002_PLAN.md step 1. Eight models, every pair, 100
games each (50 per colour) at 3,200 simulations from sampled openings. Ratings
are Bradley-Terry Elo relative to v29 = 0 with bootstrap intervals; the
head-to-head matrix is kept because these models are non-transitive. Resumable:
finished pairings are skipped and an interrupted one resumes from its journal.

    py -3 tools/top_round_robin.py
    py -3 tools/top_round_robin.py --fit-only
"""
import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import elo_tournament as et  # noqa: E402  (fit, expected, summarize, result_files)

PLAYERS = [
    ("v29", "models/bootstrap_v29/best_value_net.pt"),
    ("v28", "models/bootstrap_v28/best_value_net.pt"),
    ("gen49", "models/candidates/bootstrap_main_gen_0049/arena_selected.pt"),
    ("gen52A", "models/candidates/bootstrap_main_gen_0052/arena_selected.pt"),
    ("gen52B", "models/candidates/bootstrap_main_gen_0052_pool/arena_selected.pt"),
    ("gen52C", "models/candidates/bootstrap_main_gen_0052_poolcap/arena_selected.pt"),
    ("gen52L", "models/candidates/bootstrap_main_gen_0052_large/arena_selected.pt"),
    ("gen52R", "models/candidates/bootstrap_main_gen_0052_ramp/arena_selected.pt"),
]
REFERENCE = "v29"
OUT = os.path.join(ROOT, "benchmarks", "top_rr_20261002")
GAMES, SIMS = 100, 3200
SEED_BASE, SEED_STRIDE = 3_900_000_000, 1_000_000   # < 2**32 for all 28 pairings
ORDER_SEED = 20261002


def schedule(names):
    pairs = list(itertools.combinations(names, 2))
    random.Random(ORDER_SEED).shuffle(pairs)
    return pairs


def rate(names, results, reps=1000, seed=ORDER_SEED):
    """Elo relative to REFERENCE = 0, with a parametric bootstrap over each pairing's games."""
    def fit(rows):
        r = et.fit(names, [(x["a"], x["b"], x["wins"] + x["draws"] / 2, x["games"]) for x in rows])
        return {n: v - r[REFERENCE] for n, v in r.items()}
    point = fit(results)
    rng = np.random.default_rng(seed)
    draws = {n: [] for n in names}
    for _ in range(reps):
        sample = []
        for x in results:
            p = np.array([x["wins"], x["draws"], x["losses"]], float)
            w, d, l = rng.multinomial(x["games"], p / p.sum())
            sample.append(dict(a=x["a"], b=x["b"], games=x["games"], wins=w, draws=d, losses=l))
        for n, v in fit(sample).items():
            draws[n].append(v)
    return {n: dict(elo=round(point[n], 1), ci95=[round(float(np.percentile(draws[n], 2.5)), 1),
                                                  round(float(np.percentile(draws[n], 97.5)), 1)])
            for n in names}


def report(out, reps=1000):
    names = [n for n, _ in PLAYERS]
    results = [json.load(open(p, encoding="utf-8"))["summary"] for p in et.result_files(out)]
    if not results:
        return None
    ratings = rate(names, results, reps)
    matrix = {f"{r['a']} vs {r['b']}": dict(score=round((r["wins"] + r["draws"] / 2) / r["games"], 4),
                                            wdl=[r["wins"], r["draws"], r["losses"]], games=r["games"],
                                            a_white=r["a_white"], a_black=r["a_black"])
              for r in results}
    rep = dict(reference=REFERENCE, games_per_pairing=GAMES, sims=SIMS, pairings_played=len(results),
               ratings=dict(sorted(ratings.items(), key=lambda kv: -kv[1]["elo"])), matrix=matrix)
    with open(os.path.join(out, "ratings.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    return rep


def play(args):
    from match import run_match
    from match_evidence import model_identity, runtime_identity
    missing = [n for n, p in PLAYERS if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit(f"missing model files for: {missing}")
    os.makedirs(os.path.join(args.out, "pairings"), exist_ok=True)
    manifest = dict(players=[dict(name=n, path=p, sha256=model_identity(os.path.join(ROOT, p))["sha256"])
                             for n, p in PLAYERS], games=GAMES, sims=SIMS, seed_base=SEED_BASE,
                    order_seed=ORDER_SEED, runtime=runtime_identity())
    path = os.path.join(args.out, "manifest.json")
    if os.path.exists(path):
        prior = json.load(open(path, encoding="utf-8"))
        if prior["players"] != manifest["players"] or prior["seed_base"] != SEED_BASE:
            raise SystemExit(f"{path} describes a different run; use a new --out")
    else:
        json.dump(manifest, open(path, "w", encoding="utf-8"), indent=2)
    pairs, path_of, started = schedule([n for n, _ in PLAYERS]), dict(PLAYERS), time.time()
    for i, (a, b) in enumerate(pairs):
        stem = os.path.join(args.out, "pairings", f"{i:02d}_{a}_vs_{b}")
        if os.path.exists(stem + ".json"):
            continue
        print(f"TOP PAIRING {i + 1}/{len(pairs)}: {a} vs {b}", flush=True)
        match = run_match(os.path.join(ROOT, path_of[a]), os.path.join(ROOT, path_of[b]), games=GAMES, sims=SIMS,
                          seed=SEED_BASE + i * SEED_STRIDE, opening_temp_plies=16, workers=args.workers,
                          engine="native", stall_timeout=1800.0, game_log=stem + ".jsonl",
                          resume=os.path.exists(stem + ".jsonl"))
        match.update(pair=[a, b], index=i, summary=et.summarize(match, a, b))
        with open(stem + ".json.tmp", "w", encoding="utf-8") as fh:
            json.dump(match, fh, indent=2)
        os.replace(stem + ".json.tmp", stem + ".json")
        s = match["summary"]
        print(f"TOP RESULT {a} vs {b}: {match['a_score']:.3f} (W/D/L {s['wins']}/{s['draws']}/{s['losses']}) "
              f"{(time.time() - started) / 3600:.2f}h into session", flush=True)
    print("TOP ROUND ROBIN COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fit-only", action="store_true")
    args = ap.parse_args()
    if not args.fit_only:
        play(args)
    rep = report(args.out)
    if rep:
        for n, r in rep["ratings"].items():
            print(f"{n:8} {r['elo']:+7.1f}  [{r['ci95'][0]:+.0f}, {r['ci95'][1]:+.0f}]")


if __name__ == "__main__":
    main()
