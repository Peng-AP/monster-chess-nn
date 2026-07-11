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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="candidate model (.pt)")
    ap.add_argument("--model-b", default=None,
                    help="opponent model (.pt); omit for the heuristic anchor")
    ap.add_argument("--games", type=int, default=20, help="total games (half per color)")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260704)
    ap.add_argument("--opening-temp-plies", type=int, default=16)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    n_white = args.games // 2
    tasks = [(True, args.seed + i, args.opening_temp_plies) for i in range(n_white)]
    tasks += [(False, args.seed + 1000 + i, args.opening_temp_plies)
              for i in range(args.games - n_white)]

    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.model_a, args.model_b, args.sims)) as pool:
        results = pool.map(_play, tasks)

    from benchmark import summarize_side
    white_games = [(r, p) for r, p, aw in results if aw]
    black_games = [(r, p) for r, p, aw in results if not aw]
    w = summarize_side(white_games)
    b = summarize_side(black_games)
    score = (w["wins"] + b["wins"] + 0.5 * (w["draws"] + b["draws"])) / args.games

    name_a = os.path.basename(os.path.dirname(args.model_a)) or "model-a"
    name_b = (os.path.basename(os.path.dirname(args.model_b))
              if args.model_b else "heuristic")
    out = {
        "match": f"{name_a} vs {name_b}",
        "games": args.games, "sims": args.sims, "seed": args.seed,
        "opening_temp_plies": args.opening_temp_plies,
        "a_score": round(score, 4),
        "a_as_white": w, "a_as_black": b,
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir,
                        f"match_{name_a}_vs_{name_b}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
