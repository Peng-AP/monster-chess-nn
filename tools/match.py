"""Parallel head-to-head match between two models (or a model and the anchor).

NN-vs-NN games need opening temperature (see benchmark.play_one): at pure
temp 0 both engines are deterministic and every game is identical. Workers
parallelize across games — a match that takes hours sequentially finishes
in minutes.

    py -3 tools/match.py --model-a models/fresh_start_v14/best_value_net.pt \\
        --model-b models/fresh_start_v12/best_value_net.pt --games 20
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

_engines = {}

# Worker default. NOT cpu_count-2: on the 5060 Ti box 14 workers dies with
# "fatal : Memory allocation failure" during CUDA init and leaves 14 orphaned
# ~1.4 GB processes behind. Measured aggregate throughput at 400 sims
# (2026-08-01, scratchpad contention_bench) plateaus well before that --
# 4 workers 5.39 decisions/s, 8 workers 7.11, 12 workers 7.38 -- so 8 buys
# the whole win, keeps VRAM at ~3 GB, and leaves the box usable.
DEFAULT_WORKERS = 8


def resolve_opening_temp_plies(model_b, requested):
    '''Use sampled openings only when both opponents are neural models.'''
    if requested is not None:
        return int(requested)
    return 16 if model_b else 0


def _init_worker(model_a, model_b, sims):
    from benchmark import _build_engine
    _engines["a"], _ = _build_engine(model_a, sims)
    _engines["b"], _ = _build_engine(model_b, sims)


def _play(task):
    """task = (a_is_white, seed, temp_plies) -> (a_result, plies)."""
    from benchmark import play_one
    a_is_white, seed, temp_plies = task
    random.seed(seed)
    white = _engines["a"] if a_is_white else _engines["b"]
    black = _engines["b"] if a_is_white else _engines["a"]
    result, plies, _dec = play_one(white, black, opening_temp_plies=temp_plies)
    return (result if a_is_white else -result), plies, a_is_white


def run_match(model_a, model_b, games, sims, seed, opening_temp_plies=None,
              workers=None):
    """Play a match and return the result dict. The only producer of this schema.

    Callers that need several legs (tools/gate.py) go through here rather than
    re-implementing the pool, so there is exactly one JSON shape to read --
    a_score / a_as_white / a_as_black. benchmark.py's run_benchmark emits a
    *different* shape (candidate_score / white_strength / black_strength);
    confusing the two has cost a whole gate run before (HANDOFF SS10.1).
    """
    opening_temp_plies = resolve_opening_temp_plies(model_b, opening_temp_plies)
    workers = workers or DEFAULT_WORKERS

    n_white = games // 2
    tasks = [(True, seed + i, opening_temp_plies) for i in range(n_white)]
    tasks += [(False, seed + 1000 + i, opening_temp_plies)
              for i in range(games - n_white)]

    t0 = time.time()
    with mp.Pool(workers, initializer=_init_worker,
                 initargs=(model_a, model_b, sims)) as pool:
        results = pool.map(_play, tasks)

    from benchmark import summarize_side
    white_games = [(r, p) for r, p, aw in results if aw]
    black_games = [(r, p) for r, p, aw in results if not aw]
    w = summarize_side(white_games)
    b = summarize_side(black_games)
    score = (w["wins"] + b["wins"] + 0.5 * (w["draws"] + b["draws"])) / games

    name_a = os.path.basename(os.path.dirname(model_a)) or "model-a"
    name_b = (os.path.basename(os.path.dirname(model_b))
              if model_b else "heuristic")
    return {
        "match": f"{name_a} vs {name_b}",
        "name_a": name_a, "name_b": name_b,
        "games": games, "sims": sims, "seed": seed,
        "opening_temp_plies": opening_temp_plies,
        "workers": workers,
        "a_score": round(score, 4),
        "a_as_white": w, "a_as_black": b,
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="candidate model (.pt)")
    ap.add_argument("--model-b", default=None,
                    help="opponent model (.pt); omit for the heuristic anchor")
    ap.add_argument("--games", type=int, default=20, help="total games (half per color)")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260704)
    # Left as None so resolve_opening_temp_plies() can pick the default from the
    # opponent: heuristic tie-breaks already diversify anchor games, so only
    # NN-vs-NN matches need sampled model openings.
    ap.add_argument("--opening-temp-plies", type=int, default=None,
                    help="default: 16 for NN-vs-NN, 0 vs the heuristic anchor")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    out = run_match(args.model_a, args.model_b, args.games, args.sims,
                    args.seed, args.opening_temp_plies, args.workers)
    name_a, name_b = out["name_a"], out["name_b"]

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir,
                        f"match_{name_a}_vs_{name_b}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
