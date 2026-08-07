"""How much of search's advantage has the policy actually absorbed?

Self-play bootstrapping is distillation: search plays better than the raw
network, and training on search's choices is supposed to move the network
toward them. Measured 2026-08-07, search at 700 sims beats the policy alone by
**317 Elo** (0.8612 over 400 games), so the signal is there in quantity.

If distillation were working, the policy's top move should increasingly match
what search picks. Reanalysis has reported action-change rates of 86-92% for
five generations running -- but those rows are *selected* for disagreement, so
they cannot answer the question. This samples positions without that bias and
asks plainly: how often does the policy already agree with search, and where
does it disagree?

    py -3 tools/policy_search_agreement.py --model models/fresh_start_v21/best_value_net.pt
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "native"))

import chess  # noqa: E402
from evaluation import NNEvaluator  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from native_mcts import NativeMCTS  # noqa: E402


def sample_positions(model_path, n, sims, seed):
    """Positions from self-play at the generation operating point.

    Sampling from real self-play rather than from the stored corpus keeps this
    honest about the distribution the loop actually trains on.
    """
    nn = NNEvaluator(model_path)
    engine = NativeMCTS(num_simulations=sims, eval_fn=nn, batch_size=16,
                        seed=seed, allow_early_stop=True)
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        game = MonsterChessGame()
        plies = 0
        while not game.is_terminal() and plies < 400 and len(out) < n:
            if rng.random() < 0.25:          # thin the stream, decorrelate
                out.append((game.fen(), bool(game.white_half_pending)))
            action, _p, _v = engine.get_best_action(game, temperature=0.0)
            if action is None:
                break
            game.apply_search_action(action)
            plies += 1
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sims", type=int, default=700,
                    help="teacher strength; 700 is the generation setting")
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = os.path.join(ROOT, args.model) if not os.path.isabs(args.model) else args.model
    started = time.time()
    print(f"sampling {args.positions} positions from self-play...", flush=True)
    positions = sample_positions(model, args.positions, args.sims, args.seed)

    nn = NNEvaluator(model)
    teacher = NativeMCTS(num_simulations=args.sims, eval_fn=nn, batch_size=16,
                         seed=args.seed + 1, allow_early_stop=False)
    # sims=1 is degenerate (root expanded, no informative visits); sims=2 visits
    # exactly one child, chosen by PUCT with all Q=0 -- i.e. the policy's own
    # top legal move. That is the "no search" player.
    policy = NativeMCTS(num_simulations=2, eval_fn=nn, batch_size=16,
                        seed=args.seed + 1, allow_early_stop=False)

    agree = total = 0
    by_side = {"white": [0, 0], "black": [0, 0]}
    for i, (fen, half) in enumerate(positions):
        g1 = MonsterChessGame(fen); g1.white_half_pending = half
        g2 = MonsterChessGame(fen); g2.white_half_pending = half
        t_mv, _p, _v = teacher.get_best_action(g1, temperature=0.0)
        p_mv, _p, _v = policy.get_best_action(g2, temperature=0.0)
        if t_mv is None or p_mv is None:
            continue
        side = "white" if g1.is_white_turn else "black"
        total += 1
        by_side[side][1] += 1
        if t_mv.uci() == p_mv.uci():
            agree += 1
            by_side[side][0] += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(positions)}] agreement so far "
                  f"{agree/max(1,total):.1%}", flush=True)

    print(f"\npositions compared : {total}")
    print(f"teacher            : {args.sims} sims")
    print(f"\nPOLICY AGREES WITH SEARCH: {agree/total:.1%}" if total else "no data")
    for side in ("white", "black"):
        a, t = by_side[side]
        if t:
            print(f"   as {side:5} {a/t:.1%}   ({a}/{t})")
    print(f"\nelapsed {(time.time()-started)/60:.1f} min")
    print("\nA low number means search is finding moves the policy has not")
    print("absorbed -- distillation headroom. A high number means the policy")
    print("already plays search's moves and the 317 Elo comes from elsewhere")
    print("(value estimates, tactics at depth), which distillation cannot capture.")

    out = args.out or os.path.join(
        ROOT, "benchmarks",
        f"policy_search_agreement_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "sims": args.sims,
                   "positions": total, "agreement": round(agree / total, 4) if total else None,
                   "by_side": {k: {"agree": v[0], "total": v[1]} for k, v in by_side.items()},
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
