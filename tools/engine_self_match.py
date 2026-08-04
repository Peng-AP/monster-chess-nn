"""E3 gate (b): old engine vs new engine at equal sims.

Same model, same simulation count, colours alternated. If the port is faithful
the two are the same player, so the score must sit within 2 SE of 0.50 — a
result far from 0.50 means the ports differ in strength even where gate (a)
found them selecting identical moves, which would point at the parts a
single-decision comparison cannot see (tree reuse, history, cap handling).

The native side is driven with the **whole** state each ply — FEN, pending
flag, turn count and move history — because a FEN under-determines a Monster
Chess position in three separate ways (DIRECTIVE E1/E2 notes).

    py -3 tools/engine_self_match.py --games 200 --sims 200
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
sys.path.insert(0, os.path.join(ROOT, "tools"))

import monster_native as mn  # noqa: E402
from evaluation import NNEvaluator  # noqa: E402
from mcts import MCTS  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from nn_bridge import make_eval_fn  # noqa: E402

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
                    continue
            if len(fens) >= limit:
                break
    return fens


_DECISIONS = {"n": 0}


def native_move(state, history, eval_fn, channels, sims, batch, noise=False,
                temperature=0.0):
    # Seed advances per decision: a constant re-seeds an identical stream and
    # freezes both noise and temperature sampling.
    _DECISIONS["n"] += 1
    seed = 20260804 + _DECISIONS["n"]
    tree = mn.Tree(state.fen(), state.white_half_pending, state.turn_count, history)
    tree.run_batched_puct(sims, eval_fn, batch_size=batch, channels=channels,
                          allow_early_stop=True, root_noise=noise, seed=seed)
    action, _probs, _value = tree.best_action(temperature=temperature, seed=seed)
    return action


def python_move(state, evaluator, sims, batch, noise=False, temperature=0.0):
    search = MCTS(num_simulations=sims, eval_fn=evaluator, root_noise=noise,
                  allow_early_stop=True, batch_size=batch)
    action, _probs, _value = search.get_best_action(state, temperature=temperature)
    return action.uci() if action is not None else None


def play(start_fen, native_is_white, eval_fn, channels, evaluator, sims, batch,
         noise=False, temperature=0.0):
    state = MonsterChessGame(start_fen)
    history = []
    while not state.is_terminal():
        actions = state.get_search_actions()
        if not actions:
            break
        native_turn = (state.is_white_turn == native_is_white)
        uci = (native_move(state, history, eval_fn, channels, sims, batch,
                           noise, temperature)
               if native_turn
               else python_move(state, evaluator, sims, batch, noise, temperature))
        if uci is None:
            break
        move = next((m for m in actions if m.uci() == uci), None)
        if move is None:  # an engine proposed something unplayable: record it
            return None, "illegal:" + str(uci)
        history.append(uci)
        state.apply_search_action(move)
    return state.get_result(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="models/fresh_start_v19/best_value_net.pt")
    ap.add_argument("--root-noise", action="store_true",
                    help="the generation regime: gates (a)/(b) run without it")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    starts = load_starts(max(1, args.games // 2))
    eval_fn, channels = make_eval_fn(args.model)
    evaluator = NNEvaluator(args.model)
    mn.Tree(starts[0]).run_batched_puct(32, eval_fn, batch_size=args.batch,
                                        channels=channels)  # warm the GPU

    points = 0.0
    played = 0
    errors = []
    started = time.time()
    for i in range(args.games):
        start = starts[i % len(starts)]
        native_is_white = (i % 2 == 0)
        result, err = play(start, native_is_white, eval_fn, channels, evaluator,
                           args.sims, args.batch, args.root_noise,
                           args.temperature)
        if err:
            errors.append({"game": i, "error": err})
            continue
        played += 1
        # Only a king capture is a win (owner, 2026-08-03); the cap is a draw.
        if result is not None and result >= 1:
            points += 1.0 if native_is_white else 0.0
        elif result is not None and result <= -1:
            points += 0.0 if native_is_white else 1.0
        else:
            points += 0.5

    elapsed = time.time() - started
    score = points / played if played else None
    se = math.sqrt(0.25 / played) if played else None
    summary = {
        "games_played": played,
        "sims": args.sims,
        "root_noise": args.root_noise,
        "temperature": args.temperature,
        "native_score": round(score, 4) if score is not None else None,
        "se": round(se, 4) if se else None,
        "deviation_in_se": round(abs(score - 0.5) / se, 2) if score and se else None,
        "passed": (abs(score - 0.5) <= 2 * se) if score and se else False,
        "errors": errors[:10],
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"engine_self_match_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
