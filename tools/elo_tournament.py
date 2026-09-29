"""Elo round robin: every pair of a 16-model pool, fitted to one rating scale.

Owner request, September 29, 2026: a large pool, ratings anchored so the
weakest version that beats the owner is 1600. That is v21 -- the first release
the owner called "a very strong player"; v19's White was still "easy to beat",
and the September 4 round robin already recorded human play as below v21.
Re-anchoring is a constant shift, so a different anchor never needs a rerun.

Instrument: free play from sampled openings (16 temperature plies, match.py's
NN-vs-NN default), 3,200 simulations for both sides -- the site's and the
gates' operating point. Colours split evenly within every pairing.

Ratings: Bradley-Terry maximum likelihood on points (a draw is half a win for
each side), with one virtual draw per pairing so a 40-0 sweep stays finite.
95% intervals come from a parametric bootstrap that resamples every pairing's
games. Free play is non-transitive (September 4: RMS residual 73.5 Elo), so the
report also lists the pairings the fit explains worst.

Pairings run in a fixed shuffled order, so a run stopped early still has an
even spread of opponents per player; `--fit-only` rates whatever is finished.

    py -3 tools/elo_tournament.py --smoke          # every model loads: 15-pair chain, 2 games each
    py -3 tools/elo_tournament.py                  # production (resumable)
    py -3 tools/elo_tournament.py --fit-only
"""
import argparse
import itertools
import json
import math
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

PLAYERS = [
    ("v17", "models/fresh_start_v17/best_value_net.pt"),
    ("v19", "models/fresh_start_v19/best_value_net.pt"),
    ("v20", "models/fresh_start_v20/best_value_net.pt"),
    ("v21", "models/fresh_start_v21/best_value_net.pt"),
    ("v22", "models/fresh_start_v22/best_value_net.pt"),
    ("v23", "models/bootstrap_v23/best_value_net.pt"),
    ("v24", "models/bootstrap_v24/best_value_net.pt"),
    ("v25", "models/bootstrap_v25/best_value_net.pt"),
    ("v26", "models/bootstrap_v26/best_value_net.pt"),
    ("v27", "models/bootstrap_v27/best_value_net.pt"),
    ("v28", "models/bootstrap_v28/best_value_net.pt"),
    ("v29", "models/bootstrap_v29/best_value_net.pt"),
    ("B2", "models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt"),
    ("gen49", "models/candidates/bootstrap_main_gen_0049/arena_selected.pt"),
    ("gen52B", "models/candidates/bootstrap_main_gen_0052_pool/arena_selected.pt"),
    ("gen52C", "models/candidates/bootstrap_main_gen_0052_poolcap/arena_selected.pt"),
]
ANCHOR = ("v21", 1600.0)
OUT = os.path.join(ROOT, "benchmarks", "elo_rr_20260929")
# Clear of every earlier campaign's blocks (gen52 used 3.10e9-3.13e9). match.py
# seeds games at seed+i and seed+1000+i, so a million apart never overlaps.
SEED_BASE = 3_400_000_000
SEED_STRIDE = 1_000_000
ORDER_SEED = 20260929
PRIOR_DRAWS = 1.0
BOOTSTRAP = 1000


def schedule(names):
    pairs = list(itertools.combinations(names, 2))
    random.Random(ORDER_SEED).shuffle(pairs)
    return pairs


# --- rating fit ------------------------------------------------------------------

def fit(names, pairings, prior_draws=PRIOR_DRAWS, iterations=5000, tol=1e-10):
    """Bradley-Terry MLE by Hunter's MM iteration. pairings: (a, b, points_a, games).

    Returns Elo on an arbitrary origin (mean 0). prior_draws adds that many drawn
    games to every played pairing, which keeps an all-wins pairing finite.
    """
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    n = np.zeros((k, k))
    wins = np.zeros(k)
    for a, b, points_a, games in pairings:
        i, j = idx[a], idx[b]
        total = games + prior_draws
        n[i, j] += total
        n[j, i] += total
        wins[i] += points_a + prior_draws / 2
        wins[j] += games - points_a + prior_draws / 2
    gamma = np.ones(k)
    for _ in range(iterations):
        denom = (n / (gamma[:, None] + gamma[None, :])).sum(axis=1)
        new = np.where(denom > 0, wins / np.where(denom > 0, denom, 1), gamma)
        new /= np.exp(np.log(new).mean())
        if np.max(np.abs(np.log(new) - np.log(gamma))) < tol:
            gamma = new
            break
        gamma = new
    elo = 400 * np.log10(gamma)
    return {name: float(elo[idx[name]] - elo.mean()) for name in names}


def anchored(ratings, anchor=ANCHOR):
    name, value = anchor
    shift = value - ratings[name]
    return {n: r + shift for n, r in ratings.items()}


def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def bootstrap(names, results, reps=BOOTSTRAP, seed=ORDER_SEED):
    """Parametric bootstrap: redraw each pairing's games from its own W/D/L rates."""
    rng = np.random.default_rng(seed)
    draws = {n: [] for n in names}
    for _ in range(reps):
        sample = []
        for r in results:
            counts = np.array([r["wins"], r["draws"], r["losses"]], dtype=float)
            games = int(counts.sum())
            w, d, _l = rng.multinomial(games, counts / games)
            sample.append((r["a"], r["b"], w + d / 2, games))
        rated = anchored(fit(names, sample))
        for n in names:
            draws[n].append(rated[n])
    return {n: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for n, v in draws.items()}


def summarize(match, a, b):
    """One pairing reduced to A's W/D/L, overall and by colour."""
    w, bl = match["a_as_white"], match["a_as_black"]
    return dict(a=a, b=b, games=w["games"] + bl["games"],
                wins=w["wins"] + bl["wins"], draws=w["draws"] + bl["draws"],
                losses=w["losses"] + bl["losses"],
                a_white=dict(games=w["games"], wins=w["wins"], draws=w["draws"], losses=w["losses"]),
                a_black=dict(games=bl["games"], wins=bl["wins"], draws=bl["draws"], losses=bl["losses"]),
                mean_plies=round((w["mean_plies"] * w["games"] + bl["mean_plies"] * bl["games"])
                                 / max(1, w["games"] + bl["games"]), 1))


def report(names, results, reps=BOOTSTRAP):
    points = [(r["a"], r["b"], r["wins"] + r["draws"] / 2, r["games"]) for r in results]
    ratings = anchored(fit(names, points))
    intervals = bootstrap(names, results, reps) if reps else {}
    table = {n: dict(games=0, points=0.0, wins=0, draws=0, losses=0,
                     white_games=0, white_points=0.0, black_games=0, black_points=0.0) for n in names}

    def add(name, side, wdl, flip):
        wins, draws, losses = (wdl["losses"], wdl["draws"], wdl["wins"]) if flip else \
            (wdl["wins"], wdl["draws"], wdl["losses"])
        t, games, pts = table[name], wins + draws + losses, wins + draws / 2
        t["games"] += games; t["points"] += pts
        t["wins"] += wins; t["draws"] += draws; t["losses"] += losses
        t[f"{side}_games"] += games; t[f"{side}_points"] += pts

    residuals = []
    for r in results:
        add(r["a"], "white", r["a_white"], False)
        add(r["a"], "black", r["a_black"], False)
        add(r["b"], "black", r["a_white"], True)   # A's White games are B's Black games
        add(r["b"], "white", r["a_black"], True)
        observed = (r["wins"] + r["draws"] / 2) / r["games"]
        predicted = expected(ratings[r["a"]], ratings[r["b"]])
        residuals.append(dict(a=r["a"], b=r["b"], observed=round(observed, 4),
                              predicted=round(predicted, 4), residual=round(observed - predicted, 4),
                              z=round((observed - predicted) / math.sqrt(max(predicted * (1 - predicted), 1e-4)
                                                                         / r["games"]), 2)))
    standings = []
    for n in sorted(names, key=lambda x: -ratings[x]):
        t = table[n]
        lo, hi = intervals.get(n, (None, None))
        standings.append(dict(
            player=n, elo=round(ratings[n], 1),
            ci95=[round(lo, 1), round(hi, 1)] if lo is not None else None,
            games=t["games"], score=round(t["points"] / t["games"], 4) if t["games"] else None,
            wins=t["wins"], draws=t["draws"], losses=t["losses"],
            white_score=round(t["white_points"] / t["white_games"], 4) if t["white_games"] else None,
            black_score=round(t["black_points"] / t["black_games"], 4) if t["black_games"] else None))
    rms = math.sqrt(sum(x["residual"] ** 2 for x in residuals) / len(residuals)) if residuals else None
    return dict(anchor=dict(player=ANCHOR[0], elo=ANCHOR[1]), prior_draws=PRIOR_DRAWS,
                bootstrap_reps=reps, pairings_played=len(results), standings=standings,
                rms_residual_score=round(rms, 4) if rms is not None else None,
                worst_fit=sorted(residuals, key=lambda x: -abs(x["z"]))[:12], residuals=residuals)


# --- play ------------------------------------------------------------------------

def play(args, out):
    from match import run_match
    from match_evidence import model_identity, runtime_identity

    missing = [n for n, p in PLAYERS if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit(f"missing model files for: {missing}")
    os.makedirs(os.path.join(out, "pairings"), exist_ok=True)
    manifest_path = os.path.join(out, "manifest.json")
    manifest = dict(players=[dict(name=n, path=p, sha256=model_identity(os.path.join(ROOT, p))["sha256"])
                             for n, p in PLAYERS],
                    anchor=ANCHOR, games=args.games, sims=args.sims, workers=args.workers,
                    seed_base=SEED_BASE, seed_stride=SEED_STRIDE, order_seed=ORDER_SEED,
                    runtime=runtime_identity())
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            prior = json.load(fh)
        for key in ("players", "games", "sims", "seed_base", "order_seed"):
            if prior[key] != json.loads(json.dumps(manifest[key])):
                raise SystemExit(f"{manifest_path}: {key} differs from this run; use a new --out")
    else:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    path_of = dict(PLAYERS)
    names = [n for n, _p in PLAYERS]
    # The smoke run only proves every model loads and the files round-trip: a
    # connected chain touches all players in 15 pairings instead of 120.
    pairs = list(zip(names, names[1:])) if args.smoke else schedule(names)
    print(f"{len(names)} players, {len(pairs)} pairings x {args.games} games at {args.sims} sims "
          f"= {len(pairs) * args.games} games -> {out}", flush=True)
    started = time.time()
    played_here = 0
    for i, (a, b) in enumerate(pairs):
        stem = os.path.join(out, "pairings", f"{i:03d}_{a}_vs_{b}")
        if os.path.exists(stem + ".json"):
            continue
        seed = SEED_BASE + i * SEED_STRIDE
        print(f"ELO PAIRING {i + 1}/{len(pairs)}: {a} vs {b} (seed {seed})", flush=True)
        t0 = time.time()
        match = run_match(os.path.join(ROOT, path_of[a]), os.path.join(ROOT, path_of[b]),
                          games=args.games, sims=args.sims, seed=seed, opening_temp_plies=16,
                          workers=args.workers, engine="native", stall_timeout=900.0,
                          game_log=stem + ".jsonl", resume=os.path.exists(stem + ".jsonl"))
        match.update(pair=[a, b], index=i, summary=summarize(match, a, b))
        with open(stem + ".json.tmp", "w", encoding="utf-8") as fh:
            json.dump(match, fh, indent=2)
        os.replace(stem + ".json.tmp", stem + ".json")
        played_here += 1
        done = len(result_files(out))
        per = (time.time() - started) / played_here
        s = match["summary"]
        print(f"ELO RESULT {a} vs {b}: {match['a_score']:.3f} (W/D/L {s['wins']}/{s['draws']}/{s['losses']}, "
              f"{s['mean_plies']} plies) in {(time.time() - t0) / 60:.1f}m; {done}/{len(pairs)} done, "
              f"~{per * (len(pairs) - done) / 3600:.1f}h left at this session's pace", flush=True)
        write_ratings(out, reps=0)   # a cheap running table after every pairing
    print("ELO TOURNAMENT COMPLETE", flush=True)


def result_files(out):
    """Finished pairing reports; the game journals' .jsonl.manifest.json files are not results."""
    folder = os.path.join(out, "pairings")
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.endswith(".json") and not f.endswith(".manifest.json")]


def load_results(out):
    results = []
    for path in result_files(out):
        with open(path, encoding="utf-8") as fh:
            results.append(json.load(fh)["summary"])
    return results


def write_ratings(out, reps=BOOTSTRAP):
    names = [n for n, _p in PLAYERS]
    results = load_results(out)
    if not results:
        return None
    rated = [n for n in names if any(n in (r["a"], r["b"]) for r in results)]
    if ANCHOR[0] not in rated:
        return None
    rep = report(rated, results, reps)
    with open(os.path.join(out, "ratings.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    return rep


def print_table(rep):
    print(f"\n{'player':8} {'Elo':>7} {'95% interval':>17} {'games':>6} {'score':>6} {'White':>6} {'Black':>6}")
    for s in rep["standings"]:
        ci = f"[{s['ci95'][0]:.0f}, {s['ci95'][1]:.0f}]" if s["ci95"] else ""
        print(f"{s['player']:8} {s['elo']:7.0f} {ci:>17} {s['games']:6d} {s['score']:6.3f} "
              f"{s['white_score']:6.3f} {s['black_score']:6.3f}")
    print(f"\nanchor {rep['anchor']['player']} = {rep['anchor']['elo']:.0f}; "
          f"{rep['pairings_played']} pairings; RMS residual {rep['rms_residual_score']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=40, help="games per pairing, half per colour")
    ap.add_argument("--sims", type=int, default=3200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smoke", action="store_true", help="2 games a pairing at 32 sims, separate directory")
    ap.add_argument("--fit-only", action="store_true")
    args = ap.parse_args()
    out = args.out
    if args.smoke:
        args.games, args.sims, args.workers = 2, 32, 2
        out = out + "_smoke" if args.out == OUT else out
    if not args.fit_only:
        play(args, out)
    rep = write_ratings(out, reps=100 if args.smoke else BOOTSTRAP)
    if rep:
        print_table(rep)


if __name__ == "__main__":
    main()
