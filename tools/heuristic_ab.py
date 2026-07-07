"""A/B two heuristic evaluators head-to-head, scored per side.

The heuristic is both the training teacher and the benchmark anchor, so a
change is only worth adopting if the new heuristic actually plays stronger.
This pits candidate-MCTS vs baseline-MCTS (same search, different eval),
candidate as White for half the games and Black for the other half, and
reports each side's score separately (via benchmark.summarize_side).

Read plainly: candidate is stronger on a side if its score there is > 0.5.
Keep a change only if it wins or holds on BOTH sides.

    py -3 tools/heuristic_ab.py --geom-scale 0.3 --attack-scale 2.0 --games 20 --sims 200
"""
import argparse
import functools
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mcts import MCTS
from evaluation import evaluate
from benchmark import play_one, summarize_side


def _engine(eval_fn, sims):
    return MCTS(num_simulations=sims, eval_fn=eval_fn, root_noise=False,
                allow_early_stop=True)


def run_ab(geom_scale, attack_scale, games, sims, seed):
    baseline_eval = evaluate  # frozen anchor (both scales default to 1.0)
    candidate_eval = functools.partial(
        evaluate, king_geom_scale=geom_scale, king_attack_scale=attack_scale)

    candidate = _engine(candidate_eval, sims)
    baseline = _engine(baseline_eval, sims)

    n_white = games // 2
    n_black = games - n_white
    white_games, black_games = [], []
    t0 = time.time()

    for i in range(n_white):
        random.seed(seed + i)
        result, plies, _ = play_one(candidate, baseline)   # candidate is White
        white_games.append((result, plies))
    for i in range(n_black):
        random.seed(seed + 1000 + i)
        result, plies, _ = play_one(baseline, candidate)   # candidate is Black
        black_games.append((-result, plies))

    return {
        "king_geom_scale": geom_scale,
        "king_attack_scale": attack_scale,
        "games": games,
        "sims": sims,
        "seed": seed,
        "white_strength": summarize_side(white_games),   # candidate White vs baseline Black
        "black_strength": summarize_side(black_games),   # candidate Black vs baseline White
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser(description="A/B a candidate heuristic vs the frozen baseline")
    p.add_argument("--geom-scale", type=float, default=1.0,
                   help="candidate king_geom_scale (1.0 = baseline)")
    p.add_argument("--attack-scale", type=float, default=1.0,
                   help="candidate king_attack_scale (1.0 = baseline)")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260707)
    args = p.parse_args()

    print(f"A/B: candidate(geom={args.geom_scale}, attack={args.attack_scale}) "
          f"vs baseline(1.0, 1.0) — {args.games} games @{args.sims} sims")
    result = run_ab(args.geom_scale, args.attack_scale, args.games, args.sims, args.seed)
    print(json.dumps(result, indent=2))
    w, b = result["white_strength"], result["black_strength"]
    print("\n--- Candidate strength vs baseline ---")
    for side, s in (("White", w), ("Black", b)):
        verdict = "STRONGER" if (s["score"] or 0) > 0.5 else \
                  "weaker" if (s["score"] or 0) < 0.5 else "even"
        print(f"  As {side}: {s['score']} ({s['wins']}-{s['losses']}-{s['draws']}) [{verdict}]")


if __name__ == "__main__":
    main()
