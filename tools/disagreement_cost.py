"""When search overrules the policy, how much is the correction worth?

Measured 2026-08-07: search at 700 sims beats the raw policy by 317 Elo, yet
the policy already plays search's move 75% of the time. So the entire advantage
sits in the 25% of positions where they differ -- plus search's in-play value
estimates, which policy distillation cannot transfer at all.

That leaves one question the bootstrap loop turns on. If the disagreements are
**near-ties** -- the policy's move nearly as good as search's -- then training
on those corrections teaches almost nothing, which would explain five
generations of reanalysis producing null candidates. If they are **real
blunders**, the corrections carry value and the loop is failing for some other
reason.

Method: at each disagreement, play search's move and the policy's move, search
each resulting position, and compare the two values in the MOVER's frame. The
difference is what choosing the policy's move actually costs.

    py -3 tools/disagreement_cost.py --model models/fresh_start_v21/best_value_net.pt
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "native"))

from evaluation import NNEvaluator  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from native_mcts import NativeMCTS  # noqa: E402


def value_after(engine, game, move):
    """Search the position after `move`, returned in the MOVER's frame.

    White moves twice per turn, so applying one half-move does not always flip
    the side. Comparing raw root values without checking would silently negate
    half the samples.
    """
    mover_is_white = game.is_white_turn
    probe = game.clone()
    probe.apply_search_action(move)
    if probe.is_terminal():
        result = probe.get_result()          # +1 White, -1 Black
        return result if mover_is_white else -result
    _mv, _probs, value = engine.get_best_action(probe, temperature=0.0)
    # get_best_action reports the root value in the side-to-move frame
    return value if probe.is_white_turn == mover_is_white else -value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sims", type=int, default=700)
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=8080)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = os.path.join(ROOT, args.model) if not os.path.isabs(args.model) else args.model
    nn = NNEvaluator(model)
    search = NativeMCTS(num_simulations=args.sims, eval_fn=nn, batch_size=16,
                        seed=args.seed, allow_early_stop=False)
    policy = NativeMCTS(num_simulations=2, eval_fn=nn, batch_size=16,
                        seed=args.seed, allow_early_stop=False)
    rng = random.Random(args.seed)

    costs, by_side = [], {"white": [], "black": []}
    seen = disagreements = 0
    started = time.time()
    while seen < args.positions:
        game = MonsterChessGame()
        plies = 0
        while not game.is_terminal() and plies < 400 and seen < args.positions:
            if rng.random() < 0.3:
                seen += 1
                s_mv, _p, _v = search.get_best_action(game, temperature=0.0)
                p_mv, _p, _v = policy.get_best_action(game, temperature=0.0)
                if s_mv is not None and p_mv is not None and s_mv.uci() != p_mv.uci():
                    disagreements += 1
                    side = "white" if game.is_white_turn else "black"
                    cost = value_after(search, game, s_mv) - value_after(search, game, p_mv)
                    costs.append(cost)
                    by_side[side].append(cost)
                if seen % 50 == 0:
                    print(f"  [{seen}/{args.positions}] disagreements "
                          f"{disagreements} | mean cost "
                          f"{statistics.mean(costs) if costs else 0:+.4f}", flush=True)
            action, _p, _v = search.get_best_action(game, temperature=0.0)
            if action is None:
                break
            game.apply_search_action(action)
            plies += 1

    print(f"\npositions sampled : {seen}")
    print(f"disagreements     : {disagreements}  ({disagreements/seen:.1%})")
    if not costs:
        raise SystemExit("no disagreements found")
    mean = statistics.mean(costs)
    med = statistics.median(costs)
    print(f"\ncost of playing the POLICY's move instead of SEARCH's:")
    print(f"   mean   {mean:+.4f}")
    print(f"   median {med:+.4f}")
    print(f"   sd     {statistics.pstdev(costs):.4f}")
    for side in ("white", "black"):
        if by_side[side]:
            print(f"   as {side:5} mean {statistics.mean(by_side[side]):+.4f} "
                  f"(n={len(by_side[side])})")
    big = sum(1 for c in costs if c > 0.20)
    tiny = sum(1 for c in costs if abs(c) < 0.05)
    print(f"\n   near-ties (|cost| < 0.05) : {tiny}/{len(costs)}  {tiny/len(costs):.0%}")
    print(f"   real corrections (> 0.20) : {big}/{len(costs)}  {big/len(costs):.0%}")
    print("\nMostly near-ties => distilling these corrections teaches little,")
    print("and the 317 Elo lives in search's evaluation, not in its move choice.")
    print(f"\nelapsed {(time.time()-started)/60:.1f} min")

    out = args.out or os.path.join(
        ROOT, "benchmarks",
        f"disagreement_cost_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "sims": args.sims,
                   "positions": seen, "disagreements": disagreements,
                   "mean_cost": round(mean, 4), "median_cost": round(med, 4),
                   "near_tie_fraction": round(tiny / len(costs), 4),
                   "big_correction_fraction": round(big / len(costs), 4),
                   "by_side": {k: round(statistics.mean(v), 4)
                               for k, v in by_side.items() if v},
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
