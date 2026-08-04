"""E5: the re-baseline — the record's new zero point.

Everything measured before 2026-08-03 used the old scoring rule, where a game
reaching the move limit while one side was ahead counted as a **win**. The
owner's ruling ("a win by time shouldn't be counted the same as win by
capturing the king") made every one of those numbers incomparable with anything
measured after it. This is the single event that discharges that debt: one full
cross-table of every incumbent, under captures-only scoring, on the native
engine.

It is deliberately one event rather than a trickle of re-measurements, because
a half-converted record is worse than either — you cannot tell which numbers
belong to which era.

**Per-side scores are reported for every pair, not just aggregates.** In this
variant White is enormously strong, so an aggregate near 0.5 routinely hides
White 1.0 / Black 0.0. The aggregate is the least informative number here.

Two things this settles beyond the numbers:

* **The bar.** `tools/gate.py` tracks v19, but `v19_B` is unrejected and beat
  it head-to-head under the old rule. The cross-table says whether that still
  holds when only king captures count.
* **Whether the ladder ever meant anything.** If v17/ramp/v19 collapse into
  each other under captures-only scoring, the ordering the project has been
  navigating by was an artefact of counting shuffles as wins.

    py -3 tools/rebaseline.py --games 40 --sims 400
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from match import run_match  # noqa: E402

# heuristic is `None`: the anchor every era shares, and the only opponent whose
# strength cannot have drifted.
PLAYERS = [
    ("v17", "models/fresh_start_v17/best_value_net.pt"),
    ("ramp", "models/rejected/fresh_start_v18_ramp/best_value_net.pt"),
    ("v19", "models/fresh_start_v19/best_value_net.pt"),
    ("v19_B", "models/candidates/v19_B/best_value_net.pt"),
    ("heuristic", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    missing = [name for name, path in PLAYERS
               if path and not os.path.isfile(os.path.join(ROOT, path))]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")

    pairs = list(itertools.combinations(range(len(PLAYERS)), 2))
    print(f"re-baseline: {len(PLAYERS)} players, {len(pairs)} pairs, "
          f"{args.games} games each, {args.sims} sims, engine={args.engine}",
          flush=True)

    results = {}
    started = time.time()
    for k, (i, j) in enumerate(pairs):
        name_a, path_a = PLAYERS[i]
        name_b, path_b = PLAYERS[j]
        leg_seed = args.seed + 1000 * k   # disjoint openings per pair
        t0 = time.time()
        out = run_match(
            os.path.join(ROOT, path_a) if path_a else None,
            os.path.join(ROOT, path_b) if path_b else None,
            args.games, args.sims, leg_seed, workers=args.workers,
            engine=args.engine)
        results[f"{name_a}_vs_{name_b}"] = {
            "a": name_a, "b": name_b,
            "a_score": out["a_score"],
            "a_as_white": out["a_as_white"]["score"],
            "a_as_black": out["a_as_black"]["score"],
            "a_white_time_leaning_wins": out["a_as_white"].get("time_leaning_wins"),
            "a_black_time_leaning_wins": out["a_as_black"].get("time_leaning_wins"),
            "games": out["a_as_white"]["games"] + out["a_as_black"]["games"],
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print(f"  [{k+1}/{len(pairs)}] {name_a} vs {name_b}: "
              f"a={out['a_score']:.3f} W={out['a_as_white']['score']:.3f} "
              f"B={out['a_as_black']['score']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    se = math.sqrt(0.25 / args.games)
    summary = {
        "protocol": "captures-only scoring; only a king capture is a win",
        "engine": args.engine,
        "games_per_pair": args.games,
        "sims": args.sims,
        "se_per_pair": round(se, 4),
        "players": [name for name, _ in PLAYERS],
        "pairs": results,
        "elapsed_sec": round(time.time() - started, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Cross-table of aggregate scores, row player's score against the column.
    table = {a: {} for a, _ in PLAYERS}
    for rec in results.values():
        table[rec["a"]][rec["b"]] = rec["a_score"]
        table[rec["b"]][rec["a"]] = round(1.0 - rec["a_score"], 4)
    summary["cross_table"] = table

    names = [n for n, _ in PLAYERS]
    print()
    print("cross-table (row score vs column, aggregate):")
    print("            " + "".join(f"{n:>11}" for n in names))
    for a in names:
        cells = "".join(f"{table[a].get(b, float('nan')):>11.3f}"
                        if b != a else f"{'-':>11}" for b in names)
        print(f"{a:>12}" + cells)
    print(f"\nSE per pair: {se:.3f} ({args.games} games)")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir,
                            f"rebaseline_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
