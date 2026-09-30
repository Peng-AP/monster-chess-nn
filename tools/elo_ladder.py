"""Strength ladder: rate (model, simulations) settings on the round robin's scale.

Two questions, one run:

1. **Search scaling.** How much is a doubling of simulations worth to v29?
   The round robin measured every model at 3,200 only.
2. **Levels a human can play.** The weakest model in the round robin is v17 at
   1198. Settings below that (few simulations, an old network) give the site
   difficulty levels labelled in the same Elo.

Every setting plays the round robin's 16 models, which keep playing at 3,200
simulations. Stage 1: 8 games against each of them (4 per colour) for a
provisional rating. Stage 2: 32 more against the 6 whose ratings are closest,
where a game carries the most information. Ratings come from one joint
Bradley-Terry fit over these games plus all 4,800 round-robin games, anchored
v21 = 1600 exactly as `tools/elo_tournament.py` does.

    py -3 tools/elo_ladder.py --smoke
    py -3 tools/elo_ladder.py               # production (resumable)
    py -3 tools/elo_ladder.py --fit-only
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

V29 = "models/bootstrap_v29/best_value_net.pt"
V17 = "models/fresh_start_v17/best_value_net.pt"
# Cheap, informative settings first; the two deep ones last so a long night can
# stop before them and still keep every level.
LADDER = [
    ("v29@800", V29, 800),
    ("v29@200", V29, 200),
    ("v29@50", V29, 50),
    ("v17@400", V17, 400),
    ("v17@50", V17, 50),
    ("v29@12", V29, 12),
    ("v17@8", V17, 8),
    ("v29@6400", V29, 6400),
    ("v29@12800", V29, 12800),
]
POOL_SIMS = 3200
RR_DIR = os.path.join(ROOT, "benchmarks", "elo_rr_20260929")
OUT = os.path.join(ROOT, "benchmarks", "elo_ladder_20260930")
STAGE1_GAMES = 8
STAGE2_GAMES = 32
STAGE2_OPPONENTS = 6
# Below 2**32 throughout (numpy seeds), clear of the round robin's 3.40e9-3.52e9
# block; match.py uses seed+i and seed+1000+i, so a 10,000 stride never overlaps.
SEED_BASE = 3_600_000_000
SEED_STRIDE = 10_000


def seed_for(cand, opp, stage):
    return SEED_BASE + (cand * 100 + opp * 2 + stage) * SEED_STRIDE


def pool_ratings():
    with open(os.path.join(RR_DIR, "ratings.json"), encoding="utf-8") as fh:
        return {s["player"]: s["elo"] for s in json.load(fh)["standings"]}


def nearest(rating, pool, k=STAGE2_OPPONENTS):
    return sorted(pool, key=lambda n: (abs(pool[n] - rating), n))[:k]


def provisional(name, results, pool):
    """One-player MLE with the pool held at its round-robin ratings."""
    lo, hi = -1000.0, 4000.0
    rows = [r for r in results if r["a"] == name]
    if not rows:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        # Same virtual draw per pairing as the joint fit keeps sweeps finite.
        grad = sum((r["wins"] + r["draws"] / 2 + et.PRIOR_DRAWS / 2)
                   - (r["games"] + et.PRIOR_DRAWS) * et.expected(mid, pool[r["b"]]) for r in rows)
        lo, hi = (mid, hi) if grad > 0 else (lo, mid)
    return (lo + hi) / 2


def result_path(out, cand, stage, opp_name):
    return os.path.join(out, "pairings", f"{cand}_s{stage}_vs_{opp_name}")


def load(out):
    rows = []
    for path in et.result_files(out):
        with open(path, encoding="utf-8") as fh:
            rows.append(json.load(fh)["summary"])
    return rows


def play_one(args, out, ci, cand, path, sims, oi, opp, stage, games):
    from match import run_match
    stem = result_path(out, cand, stage, opp)
    if os.path.exists(stem + ".json"):
        return
    seed = seed_for(ci, oi, stage)
    t0 = time.time()
    match = run_match(os.path.join(ROOT, path), os.path.join(ROOT, dict(et.PLAYERS)[opp]),
                      games=games, sims=sims, sims_b=POOL_SIMS, seed=seed, opening_temp_plies=16,
                      workers=args.workers, engine="native", stall_timeout=1800.0,
                      game_log=stem + ".jsonl", resume=os.path.exists(stem + ".jsonl"))
    match.update(pair=[cand, opp], stage=stage, summary=et.summarize(match, cand, opp))
    with open(stem + ".json.tmp", "w", encoding="utf-8") as fh:
        json.dump(match, fh, indent=2)
    os.replace(stem + ".json.tmp", stem + ".json")
    s = match["summary"]
    print(f"LADDER RESULT {cand} vs {opp} (stage {stage}): {match['a_score']:.3f} "
          f"(W/D/L {s['wins']}/{s['draws']}/{s['losses']}, {s['mean_plies']} plies) "
          f"in {(time.time() - t0) / 60:.1f}m", flush=True)


def play(args, out, ladder):
    from match_evidence import model_identity, runtime_identity
    os.makedirs(os.path.join(out, "pairings"), exist_ok=True)
    pool = pool_ratings()
    manifest = dict(ladder=[dict(name=n, path=p, sims=s, sha256=model_identity(os.path.join(ROOT, p))["sha256"])
                            for n, p, s in ladder],
                    pool_sims=POOL_SIMS, pool_ratings=pool, stage1_games=args.stage1, stage2_games=args.stage2,
                    stage2_opponents=STAGE2_OPPONENTS, seed_base=SEED_BASE, runtime=runtime_identity())
    path = os.path.join(out, "manifest.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            prior = json.load(fh)
        for key in ("ladder", "stage1_games", "stage2_games", "seed_base"):
            if prior[key] != json.loads(json.dumps(manifest[key])):
                raise SystemExit(f"{path}: {key} differs from this run; use a new --out")
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    names = [n for n, _p in et.PLAYERS]
    started = time.time()
    for ci, (cand, cpath, sims) in enumerate(ladder):
        print(f"LADDER SETTING {ci + 1}/{len(ladder)}: {cand} "
              f"({(time.time() - started) / 3600:.2f}h into this session)", flush=True)
        for oi, opp in enumerate(names):
            play_one(args, out, ci, cand, cpath, sims, oi, opp, 1, args.stage1)
        rating = provisional(cand, load(out), pool)
        chosen = nearest(rating, pool)
        print(f"LADDER PROVISIONAL {cand}: {rating:.0f}; stage 2 vs {', '.join(chosen)}", flush=True)
        for opp in chosen:
            play_one(args, out, ci, cand, cpath, sims, names.index(opp), opp, 2, args.stage2)
        rep = write_ratings(out, reps=0)
        row = next(s for s in rep["standings"] if s["player"] == cand)
        print(f"LADDER RATED {cand}: {row['elo']:.0f} over {row['games']} games", flush=True)
    print("LADDER COMPLETE", flush=True)


def write_ratings(out, reps=et.BOOTSTRAP):
    ladder_rows = load(out)
    rr_rows = et.load_results(RR_DIR)
    names = [n for n, _p in et.PLAYERS] + sorted({r["a"] for r in ladder_rows})
    rep = et.report(names, rr_rows + ladder_rows, reps)
    rep["ladder_settings"] = sorted({r["a"] for r in ladder_rows})
    with open(os.path.join(out, "ratings.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smoke", action="store_true",
                    help="two cheap settings, 2 + 2 games, separate directory: plumbing only")
    ap.add_argument("--fit-only", action="store_true")
    args = ap.parse_args()
    args.stage1, args.stage2 = STAGE1_GAMES, STAGE2_GAMES
    out, ladder = args.out, LADDER
    if args.smoke:
        global POOL_SIMS
        POOL_SIMS, args.stage1, args.stage2, args.workers = 16, 2, 2, 2
        ladder = [("v29@12", V29, 12), ("v17@8", V17, 8)]
        out = out + "_smoke" if args.out == OUT else out
    if not args.fit_only:
        play(args, out, ladder)
    rep = write_ratings(out, reps=100 if args.smoke else et.BOOTSTRAP)
    et.print_table(rep)


if __name__ == "__main__":
    main()
