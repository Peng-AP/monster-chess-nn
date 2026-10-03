"""Add one model to the October 2 top-group round robin without replaying it.

The new player meets each of the 8 existing players for 100 games (50 per
colour) at 3,200 simulations, in its own directory and seed block; ratings are
then fitted jointly over the original 28 pairings plus these. Resumable.

    py -3 tools/top_rr_extend.py --name gen52LR --model models/candidates/bootstrap_main_gen_0052_large_ramp/arena_selected.pt
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import elo_tournament as et  # noqa: E402
import top_round_robin as top  # noqa: E402

SEED_BASE = 4_000_000_000   # clear of every earlier block; + 8 x 1e6 stays below 2**32


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "top_rr_20261003_extend"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    from match import run_match
    if not os.path.exists(os.path.join(ROOT, args.model)):
        raise SystemExit(f"missing model {args.model}")
    os.makedirs(os.path.join(args.out, "pairings"), exist_ok=True)
    for i, (opp, path) in enumerate(top.PLAYERS):
        stem = os.path.join(args.out, "pairings", f"{i:02d}_{args.name}_vs_{opp}")
        if os.path.exists(stem + ".json"):
            continue
        print(f"EXTEND PAIRING {i + 1}/{len(top.PLAYERS)}: {args.name} vs {opp}", flush=True)
        match = run_match(os.path.join(ROOT, args.model), os.path.join(ROOT, path), games=top.GAMES, sims=top.SIMS,
                          seed=SEED_BASE + i * top.SEED_STRIDE, opening_temp_plies=16, workers=args.workers,
                          engine="native", stall_timeout=1800.0, game_log=stem + ".jsonl",
                          resume=os.path.exists(stem + ".jsonl"))
        match.update(pair=[args.name, opp], index=i, summary=et.summarize(match, args.name, opp))
        with open(stem + ".json.tmp", "w", encoding="utf-8") as fh:
            json.dump(match, fh, indent=2)
        os.replace(stem + ".json.tmp", stem + ".json")
        print(f"EXTEND RESULT {args.name} vs {opp}: {match['a_score']:.3f} ({time.strftime('%H:%M')})", flush=True)
    names = [n for n, _ in top.PLAYERS] + [args.name]
    rows = [json.load(open(p, encoding="utf-8"))["summary"] for p in et.result_files(top.OUT)]
    rows += [json.load(open(p, encoding="utf-8"))["summary"] for p in et.result_files(args.out)]
    ratings = top.rate(names, rows)
    with open(os.path.join(args.out, "ratings.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(reference=top.REFERENCE, added=args.name, pairings=len(rows),
                       ratings=dict(sorted(ratings.items(), key=lambda kv: -kv[1]["elo"]))), fh, indent=2)
    for n, r in sorted(ratings.items(), key=lambda kv: -kv[1]["elo"]):
        print(f"{n:8} {r['elo']:+7.1f}  [{r['ci95'][0]:+.0f}, {r['ci95'][1]:+.0f}]")
    print("EXTEND COMPLETE", flush=True)


if __name__ == "__main__":
    main()
