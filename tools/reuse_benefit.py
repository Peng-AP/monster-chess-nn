"""How much does tree reuse across moves actually buy? (LC0 reference, E6)

Two numbers, because they answer different questions:

**Carry-over** — how many nodes a search starts with instead of zero. This is
the mechanism, and it is free: those nodes were paid for by earlier searches.
The Python engine reuses only across White's first -> second half-move, so it
throws the tree away on every Black move; LC0 keeps it for the whole game.

**Strength** — a head-to-head match, reuse on against reuse off, same model and
same nominal sims. This is the number that matters. Carry-over could be large
and still buy nothing if the retained statistics are stale or mis-framed, and
a rebasing error would show up here as a *loss* rather than as a crash.

    py -3 tools/reuse_benefit.py --games 40 --sims 200
"""
import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "native"))

from evaluation import NNEvaluator  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from native_mcts import NativeMCTS  # noqa: E402

DECK = os.path.join(ROOT, "data", "start_fens", "promotion_defense_deck_v1.jsonl")


def load_starts(limit):
    fens = []
    with open(DECK, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    fens.append(json.loads(line)["fen"])
                except Exception:
                    pass
            if len(fens) >= limit:
                break
    return fens


def carryover(model, sims, plies, seed):
    """Nodes a search inherits, with reuse across moves and without."""
    out = {}
    evaluator = NNEvaluator(model)
    for across in (False, True):
        random.seed(seed)
        engine = NativeMCTS(num_simulations=sims, eval_fn=evaluator,
                            allow_early_stop=False, reuse_across_moves=across,
                            seed=seed)
        state = MonsterChessGame(load_starts(1)[0])
        inherited, searches = [], 0
        for _ in range(plies):
            if state.is_terminal():
                break
            before = (engine._reuse_tree.node_count()
                      if engine._reuse_tree is not None else 0)
            inherited.append(before)
            searches += 1
            action, _p, _v = engine.get_best_action(state, temperature=0.0)
            if action is None:
                break
            state.apply_search_action(action)
        out[across] = (sum(inherited) / max(1, searches), searches)
    return out


def match(model, sims, games, seed, on_sims=None):
    """Reuse-on vs reuse-off, colours alternated.

    `on_sims` lets the reuse side be given the extra simulations its wall-clock
    saving pays for. Equal-sims is the wrong comparison for a change whose
    benefit is speed: reuse banks the previous search's visit lead, so
    `_should_stop_early` fires sooner and the search terminates earlier rather
    than going deeper. Measured 2026-08-04: 233 vs 301 ms per decision, so at
    equal time the reuse side affords ~1.29x the simulations.
    """
    evaluator = NNEvaluator(model)
    starts = load_starts(max(1, games // 2))
    points = 0.0
    played = 0
    began = time.time()
    for i in range(games):
        random.seed(seed + i)
        on = NativeMCTS(num_simulations=on_sims or sims, eval_fn=evaluator,
                        allow_early_stop=True, reuse_across_moves=True)
        off = NativeMCTS(num_simulations=sims, eval_fn=evaluator,
                         allow_early_stop=True, reuse_across_moves=False)
        on_is_white = (i % 2 == 0)
        state = MonsterChessGame(starts[i % len(starts)])
        while not state.is_terminal():
            actions = state.get_search_actions()
            if not actions:
                break
            engine = on if (state.is_white_turn == on_is_white) else off
            action, _p, _v = engine.get_best_action(state, temperature=0.0)
            if action is None:
                break
            state.apply_search_action(action)
        result = state.get_result()
        played += 1
        if result is not None and result >= 1:
            points += 1.0 if on_is_white else 0.0
        elif result is not None and result <= -1:
            points += 0.0 if on_is_white else 1.0
        else:
            points += 0.5
        # Progress every few games: a match that reports only at the end is
        # indistinguishable from a hung one, which is the whole reason
        # tools/runs.py exists.
        if played % max(1, games // 20) == 0 or played == games:
            elapsed = time.time() - began
            rate = played / elapsed if elapsed else 0.0
            print(f"  [{played:3d}/{games}] reuse-on {points / played:.3f}  "
                  f"{elapsed / 60:5.1f}m elapsed, "
                  f"~{(games - played) / rate / 60 if rate else 0:5.1f}m left",
                  flush=True)
    return points / played, played


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fresh_start_v19/best_value_net.pt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--plies", type=int, default=30)
    ap.add_argument("--on-sims", type=int, default=None,
                    help="simulations for the reuse side; use the wall-clock "
                         "ratio to compare at equal TIME rather than equal nodes")
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    print("measuring carry-over...", flush=True)
    carry = carryover(args.model, args.sims, args.plies, args.seed)
    for across, (mean_nodes, searches) in sorted(carry.items()):
        label = "across moves" if across else "half-pair only (python scope)"
        print(f"  {label:32s} mean inherited nodes/search {mean_nodes:8.1f} "
              f"over {searches} searches", flush=True)

    print("\nmeasuring strength...", flush=True)
    started = time.time()
    score, played = match(args.model, args.sims, args.games, args.seed,
                          on_sims=args.on_sims)
    se = math.sqrt(0.25 / played)
    print(f"  reuse-on score {score:.4f} +- {se:.4f} over {played} games "
          f"({(score - 0.5) / se:+.2f} SE)  [{time.time() - started:.0f}s]")

    summary = {
        "sims": args.sims,
        "on_sims": args.on_sims or args.sims,
        "games": played,
        "carryover_half_pair_only": carry[False][0],
        "carryover_across_moves": carry[True][0],
        "reuse_on_score": round(score, 4),
        "se": round(se, 4),
        "deviation_in_se": round((score - 0.5) / se, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = os.path.join(ROOT, "benchmarks",
                        f"reuse_benefit_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
