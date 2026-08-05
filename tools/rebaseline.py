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

from match import match_game_seeds, run_match  # noqa: E402

# heuristic is `None`: the anchor every era shares, and the only opponent whose
# strength cannot have drifted.
PLAYERS = [
    ("v17", "models/fresh_start_v17/best_value_net.pt"),
    ("ramp", "models/rejected/fresh_start_v18_ramp/best_value_net.pt"),
    ("v19", "models/fresh_start_v19/best_value_net.pt"),
    ("v19_B", "models/candidates/v19_B/best_value_net.pt"),
    ("heuristic", None),
]

# A match consumes seed..seed+games/2 and seed+1000.. for its two colours.
# Spacing pairs by only 1000 therefore overlaps one pair's Black sample with
# the next pair's White sample.  Keep the stride comfortably above both bands.
PAIR_SEED_STRIDE = 100_000
BAR_CONFIRM_SEED_OFFSET = 2_000_000
BAR_PAIR = ("v19", "v19_B")


def build_seed_plan(base_seed, pair_count, games):
    """Build and validate disjoint seeds for every cross-table read.

    The final entry is the independent confirmation of v19 vs v19_B required
    by the owner's two-read definition of "definitive".
    """
    plan = [(f"pair_{k}", base_seed + PAIR_SEED_STRIDE * k)
            for k in range(pair_count)]
    plan.append(("bar_confirmation", base_seed + BAR_CONFIRM_SEED_OFFSET))

    used = {}
    for label, match_seed in plan:
        for game_seed in match_game_seeds(games, match_seed):
            previous = used.get(game_seed)
            if previous is not None:
                raise ValueError(
                    f"seed plan overlap: {label} and {previous} use {game_seed}")
            used[game_seed] = label
    return plan


def result_record(out, name_a, name_b, seed, elapsed):
    return {
        "a": name_a, "b": name_b,
        "a_score": out["a_score"],
        "a_as_white": out["a_as_white"]["score"],
        "a_as_black": out["a_as_black"]["score"],
        "a_white_time_leaning_wins": out["a_as_white"].get("time_leaning_wins"),
        "a_black_time_leaning_wins": out["a_as_black"].get("time_leaning_wins"),
        "games": out["a_as_white"]["games"] + out["a_as_black"]["games"],
        "seed": seed,
        "elapsed_sec": round(elapsed, 1),
    }


def assess_bar(first_v19_score, confirm_v19_score):
    """Apply the two-independent-reads rule without over-reading a split."""
    if first_v19_score > 0.5 and confirm_v19_score > 0.5:
        return "v19"
    if first_v19_score < 0.5 and confirm_v19_score < 0.5:
        return "v19_B"
    return None


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
    seed_plan = build_seed_plan(args.seed, len(pairs), args.games)
    print(f"re-baseline: {len(PLAYERS)} players, {len(pairs)} pairs, "
          f"{args.games} games each, {args.sims} sims, engine={args.engine}",
          flush=True)

    results = {}
    started = time.time()
    for k, (i, j) in enumerate(pairs):
        name_a, path_a = PLAYERS[i]
        name_b, path_b = PLAYERS[j]
        leg_seed = seed_plan[k][1]
        t0 = time.time()
        out = run_match(
            os.path.join(ROOT, path_a) if path_a else None,
            os.path.join(ROOT, path_b) if path_b else None,
            args.games, args.sims, leg_seed, workers=args.workers,
            engine=args.engine)
        results[f"{name_a}_vs_{name_b}"] = result_record(
            out, name_a, name_b, leg_seed, time.time() - t0)
        print(f"  [{k+1}/{len(pairs)}] {name_a} vs {name_b}: "
              f"a={out['a_score']:.3f} W={out['a_as_white']['score']:.3f} "
              f"B={out['a_as_black']['score']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # One cross-table read is not enough to move the bar.  Repeat the exact
    # v19/v19_B matchup on a fully disjoint seed range.
    by_name = dict(PLAYERS)
    confirm_seed = seed_plan[-1][1]
    t0 = time.time()
    confirm_out = run_match(
        os.path.join(ROOT, by_name[BAR_PAIR[0]]),
        os.path.join(ROOT, by_name[BAR_PAIR[1]]),
        args.games, args.sims, confirm_seed, workers=args.workers,
        engine=args.engine)
    bar_confirmation = result_record(
        confirm_out, BAR_PAIR[0], BAR_PAIR[1], confirm_seed, time.time() - t0)
    first_bar_read = results[f"{BAR_PAIR[0]}_vs_{BAR_PAIR[1]}"]
    recommended_bar = assess_bar(first_bar_read["a_score"],
                                 bar_confirmation["a_score"])
    print(f"  [confirm] {BAR_PAIR[0]} vs {BAR_PAIR[1]}: "
          f"a={confirm_out['a_score']:.3f} "
          f"W={confirm_out['a_as_white']['score']:.3f} "
          f"B={confirm_out['a_as_black']['score']:.3f} "
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
        "bar_confirmation": bar_confirmation,
        "bar_assessment": {
            "rule": "same engine must score above 0.50 in two independent reads",
            "first_v19_score": first_bar_read["a_score"],
            "confirm_v19_score": bar_confirmation["a_score"],
            "recommended_bar": recommended_bar,
            "definitive": recommended_bar is not None,
        },
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
