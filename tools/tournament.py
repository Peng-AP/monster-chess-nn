"""Round-robin tournament across model versions: how has progression gone?

Every pair plays a match with colours split evenly, so each player's record
separates into a White score and a Black score. That split is the interesting
part here -- this project's models have improved far more as White than as
Black, and an overall standings table alone would hide it.

Not a gate. `tools/gate.py` decides promotion, with a bar leg, a ramp leg, an
anchor, per-side floors and a confirmation replay on a disjoint seed. This is a
progression picture: one match per pair, no confirmation, so read a single cell
as noisy and the column totals as the signal.

    py -3 tools/tournament.py --games 30 --sims 400
    py -3 tools/tournament.py --player v21=models/fresh_start_v21/best_value_net.pt

Resumable: results land in the artifact after every pairing, and a rerun with
the same --out skips pairings already played.
"""
import argparse
import itertools
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from match import run_match  # noqa: E402

# The v2-v5 checkpoints predate the current DualHeadNet and cannot be loaded,
# so the ladder starts at v17. v18_ramp is included because it held the bar
# before v20 despite never holding a version number.
DEFAULT_PLAYERS = [
    ("v17", "models/fresh_start_v17/best_value_net.pt"),
    ("v18_ramp", "models/rejected/fresh_start_v18_ramp/best_value_net.pt"),
    ("v19", "models/fresh_start_v19/best_value_net.pt"),
    ("v19_B", "models/candidates/v19_B/best_value_net.pt"),
    ("lc0b_attn_ema", "models/candidates/lc0b_attention_ema/best_value_net.pt"),
    ("v20", "models/fresh_start_v20/best_value_net.pt"),
    ("v21", "models/fresh_start_v21/best_value_net.pt"),
]

# run_match's own warning: per-game seeds are seed+i and seed+1000+i, so two
# matches seeded closer than ~1000+games/2 replay overlapping games. Spacing
# pairings a million apart keeps every cell an independent sample.
SEED_STRIDE = 1_000_000


def load_prior(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return {tuple(r["pair"]): r for r in json.load(fh).get("pairings", [])}
    except Exception:
        return {}


def standings(players, pairings):
    """Per-player points, split by colour. Draws count a half."""
    table = {name: {"points": 0.0, "games": 0,
                    "white_points": 0.0, "white_games": 0,
                    "black_points": 0.0, "black_games": 0} for name, _p in players}

    def add(name, side, score, games):
        table[name]["points"] += score * games
        table[name]["games"] += games
        table[name][f"{side}_points"] += score * games
        table[name][f"{side}_games"] += games

    for r in pairings:
        a, b = r["pair"]
        aw, ab = r["a_as_white"], r["a_as_black"]
        # A's White games are B's Black games, and vice versa.
        add(a, "white", aw["score"], aw["games"])
        add(a, "black", ab["score"], ab["games"])
        add(b, "black", 1.0 - aw["score"], aw["games"])
        add(b, "white", 1.0 - ab["score"], ab["games"])
    for name, t in table.items():
        for key in ("", "white_", "black_"):
            g = t[f"{key}games"]
            t[f"{key}score"] = round(t[f"{key}points"] / g, 4) if g else None
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", action="append", default=[],
                    help="name=path (repeatable); replaces the default ladder")
    ap.add_argument("--games", type=int, default=30,
                    help="games per pairing, split evenly between colours")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--engine", choices=("python", "native"), default="native")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "benchmarks", "tournament_progression.json"))
    args = ap.parse_args()

    players = ([tuple(p.split("=", 1)) for p in args.player] or DEFAULT_PLAYERS)
    missing = [n for n, p in players if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit(f"missing model files for: {missing}")

    pairs = list(itertools.combinations([n for n, _p in players], 2))
    path_of = dict(players)
    prior = load_prior(args.out)
    print(f"{len(players)} players, {len(pairs)} pairings, {args.games} games each "
          f"({len(pairs) * args.games} games total) at {args.sims} sims", flush=True)
    if prior:
        print(f"resuming: {len(prior)} pairings already played", flush=True)

    results = []
    started = time.time()
    for i, (a, b) in enumerate(pairs):
        if (a, b) in prior:
            results.append(prior[(a, b)])
            continue
        r = run_match(os.path.join(ROOT, path_of[a]), os.path.join(ROOT, path_of[b]),
                      games=args.games, sims=args.sims,
                      seed=args.seed + i * SEED_STRIDE,
                      workers=args.workers, engine=args.engine)
        r["pair"] = [a, b]
        results.append(r)
        print(f"[{i + 1}/{len(pairs)}] {a} vs {b}: {r['a_score']:.3f} "
              f"(as White {r['a_as_white']['score']:.3f}, "
              f"as Black {r['a_as_black']['score']:.3f}) "
              f"elapsed {(time.time() - started) / 60:.1f}m", flush=True)
        # Written every pairing so a crash costs one match, not the run.
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"players": players, "games_per_pairing": args.games,
                       "sims": args.sims, "engine": args.engine,
                       "seed": args.seed, "pairings": results,
                       "standings": standings(players, results),
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=2)

    table = standings(players, results)
    order = sorted(table, key=lambda n: table[n]["score"] or 0, reverse=True)
    print("\n" + "=" * 66)
    print(f"{'player':16} {'overall':>9} {'as White':>10} {'as Black':>10} "
          f"{'colour gap':>11}")
    print("-" * 66)
    for name in order:
        t = table[name]
        gap = (t["white_score"] - t["black_score"]
               if None not in (t["white_score"], t["black_score"]) else None)
        print(f"{name:16} {t['score']:9.3f} {t['white_score']:10.3f} "
              f"{t['black_score']:10.3f} {gap:11.3f}")
    print("=" * 66)
    print(f"\n{len(results)} pairings in {(time.time() - started) / 60:.1f}m -> {args.out}")


if __name__ == "__main__":
    main()
