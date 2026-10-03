"""One round robin of the gen53 teacher candidates at one search depth.

docs/plans/TEACHER_SELECTION_PLAN.md. Five candidates, all 10 pairings, sampled
openings, colours split evenly, every player at --sims. Ratings are relative
to v29 = 0 with bootstrap intervals (tools/top_round_robin.rate). Resumable.

    py -3 tools/teacher_rr.py --sims 1600 --games 100
"""
import argparse
import itertools
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import elo_tournament as et  # noqa: E402
import top_round_robin as top  # noqa: E402

CANDIDATES = [
    ("v29", "models/bootstrap_v29/best_value_net.pt"),
    ("gen52R", "models/candidates/bootstrap_main_gen_0052_ramp/arena_selected.pt"),
    ("gen52LR", "models/candidates/bootstrap_main_gen_0052_large_ramp/arena_selected.pt"),
    ("gen52L", "models/candidates/bootstrap_main_gen_0052_large/arena_selected.pt"),
    ("gen52C", "models/candidates/bootstrap_main_gen_0052_poolcap/arena_selected.pt"),
]
SEED_BASE = {1600: 4_020_000_000, 6400: 4_040_000_000, 12800: 4_060_000_000}
SEED_STRIDE = 1_000_000
ORDER_SEED = 20261003
ROOT_OUT = os.path.join(ROOT, "benchmarks", "teacher_selection_20261003")


def out_dir(sims):
    return os.path.join(ROOT_OUT, f"rr_{sims}")


def schedule():
    pairs = list(itertools.combinations([n for n, _ in CANDIDATES], 2))
    random.Random(ORDER_SEED).shuffle(pairs)
    return pairs


def results(sims):
    return [json.load(open(p, encoding="utf-8"))["summary"] for p in et.result_files(out_dir(sims))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, choices=sorted(SEED_BASE), required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    from match import run_match
    from match_evidence import model_identity
    out = out_dir(args.sims)
    os.makedirs(os.path.join(out, "pairings"), exist_ok=True)
    manifest = dict(candidates=[dict(name=n, path=p, sha256=model_identity(os.path.join(ROOT, p))["sha256"])
                                for n, p in CANDIDATES], sims=args.sims, games=args.games,
                    seed_base=SEED_BASE[args.sims], order_seed=ORDER_SEED)
    path = os.path.join(out, "manifest.json")
    if os.path.exists(path):
        if json.load(open(path, encoding="utf-8")) != json.loads(json.dumps(manifest)):
            raise SystemExit(f"{path} describes a different run")
    else:
        json.dump(manifest, open(path, "w", encoding="utf-8"), indent=2)
    path_of = dict(CANDIDATES)
    for i, (a, b) in enumerate(schedule()):
        stem = os.path.join(out, "pairings", f"{i:02d}_{a}_vs_{b}")
        if os.path.exists(stem + ".json"):
            continue
        print(f"TEACHER {args.sims} PAIRING {i + 1}/10: {a} vs {b}", flush=True)
        match = run_match(os.path.join(ROOT, path_of[a]), os.path.join(ROOT, path_of[b]), games=args.games,
                          sims=args.sims, seed=SEED_BASE[args.sims] + i * SEED_STRIDE, opening_temp_plies=16,
                          workers=args.workers, engine="native", stall_timeout=3600.0, game_log=stem + ".jsonl",
                          resume=os.path.exists(stem + ".jsonl"))
        match.update(pair=[a, b], index=i, summary=et.summarize(match, a, b))
        with open(stem + ".json.tmp", "w", encoding="utf-8") as fh:
            json.dump(match, fh, indent=2)
        os.replace(stem + ".json.tmp", stem + ".json")
        print(f"TEACHER {args.sims} RESULT {a} vs {b}: {match['a_score']:.3f} ({time.strftime('%H:%M')})", flush=True)
    ratings = top.rate([n for n, _ in CANDIDATES], results(args.sims))
    json.dump(dict(sims=args.sims, ratings=ratings), open(os.path.join(out, "ratings.json"), "w", encoding="utf-8"),
              indent=2)
    print(f"TEACHER {args.sims} COMPLETE: " + ", ".join(f"{n} {r['elo']:+.0f}" for n, r in ratings.items()), flush=True)


if __name__ == "__main__":
    main()
