"""Does the value head favour one side, relative to what actually happened?

    py -3 tools/value_side_bias.py --model K=models/candidates/v19_K/best_value_net.pt \
        --model B=models/candidates/v19_B/best_value_net.pt \
        --data-dir data/processed/combined_v19_K_r50h60

This exists because the owner looked at v19_B and said its evaluations felt
White-skewed, and "that'll bite it". Measuring it found he was right and that it
was specific to B: the side gap is +0.008 for K against +0.207 for B. That
number decided which arm was promoted to v19, so it needs an artifact rather
than a line in a chat log.

Method. For held-out games only (the processed split's test set), each record's
outcome-grounded ramp target is recomputed exactly as data_processor does, and
compared against the model's prediction. Both are in WHITE's perspective, so:

    bias = predicted - target        positive => model favours White

Reported separately for White-to-move and Black-to-move positions. The *gap*
between them is the quantity of interest -- a model can be uniformly optimistic
without being side-inconsistent, and only the latter gives search targets that
shift depending on whose turn it is.

Held-out matters: on training positions a well-fit model reproduces its targets
and every bias collapses toward zero.
"""
import argparse
import json
import os
import random
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_processor import _discounted_results  # noqa: E402

# The ramp recipe the r50h60 datasets were built with -- NOT config's
# VALUE_TARGET_FLOOR/HORIZON, which are the v17-era 0.97/10 (config.py says so
# in a comment, and this tool used them anyway on its first run: the side gaps
# were unaffected, because a uniform offset cancels in a difference, but every
# absolute bias was wrong by ~0.065). These must match how the dataset was
# processed or the targets are not the targets the model was trained on.
RAMP_FLOOR = 0.5
RAMP_HORIZON = 60


def held_out_samples(raw_dir, data_dir, limit_games, limit_positions, seed,
                     value_floor, value_horizon):
    split_path = os.path.join(data_dir, "split_game_ids.json")
    with open(split_path, encoding="utf-8") as f:
        test_ids = set(json.load(f)["test"])
    paths = []
    for dirpath, _dirs, files in os.walk(raw_dir):
        for name in files:
            if not name.endswith(".jsonl"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), raw_dir).replace("\\", "/")
            if rel in test_ids:
                paths.append(os.path.join(dirpath, name))
    rng = random.Random(seed)
    if limit_games and len(paths) > limit_games:
        paths = rng.sample(paths, limit_games)

    samples = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        if not recs:
            continue
        targets = _discounted_results(recs, horizon=value_horizon,
                                      floor=value_floor)
        for rec, target in zip(recs, targets):
            samples.append((rec["fen"], rec.get("current_player") == "white", target))
    rng.shuffle(samples)
    return samples[:limit_positions], len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--raw-dir", default="data/raw/combined_v19_B")
    ap.add_argument("--data-dir", default="data/processed/combined_v19_B_r50h60")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--positions", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value-floor", type=float, default=RAMP_FLOOR,
                    help="must match how --data-dir was processed (default: ramp 0.5)")
    ap.add_argument("--value-horizon", type=int, default=RAMP_HORIZON,
                    help="must match how --data-dir was processed (default: ramp 60)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    models = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path or not os.path.exists(path):
            ap.error(f"--model expects NAME=PATH with an existing file, got {spec!r}")
        models.append((name, path))

    samples, n_games = held_out_samples(args.raw_dir, args.data_dir,
                                        args.games, args.positions, args.seed,
                                        args.value_floor, args.value_horizon)
    print(f"{len(samples)} held-out positions from {n_games} test-split games "
          f"| targets: floor {args.value_floor} horizon {args.value_horizon}")

    from evaluation import NNEvaluator
    from monster_chess import MonsterChessGame

    rows = {}
    for name, path in models:
        ev = NNEvaluator(path)
        white, black = [], []
        for fen, is_white, target in samples:
            value, _pol = ev.evaluate_with_policy(MonsterChessGame(fen=fen))
            (white if is_white else black).append(value - target)
        rows[name] = {
            "white_to_move_bias": round(st.fmean(white), 4),
            "black_to_move_bias": round(st.fmean(black), 4),
            "side_gap": round(st.fmean(white) - st.fmean(black), 4),
            "n_white": len(white), "n_black": len(black),
        }
        r = rows[name]
        print(f"  {name:10s} White {r['white_to_move_bias']:+.4f}   "
              f"Black {r['black_to_move_bias']:+.4f}   gap {r['side_gap']:+.4f}")

    payload = {
        "positions": len(samples), "test_games": n_games,
        "raw_dir": args.raw_dir, "data_dir": args.data_dir, "seed": args.seed,
        "value_floor": args.value_floor, "value_horizon": args.value_horizon,
        "convention": "bias = predicted - outcome-grounded target, White POV; "
                      "positive means the model favours White",
        "models": rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir,
                        f"value_side_bias_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
