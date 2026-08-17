"""Compare moves-left search utility on/off at real conversion positions.

The probe replays complete exhibition games so the state retains the recent
move history needed by oscillation handling. For each capped draw it selects
the last non-terminal Black-to-move state, optionally resets the move-limit
clock, and runs the same checkpoint with only moves-left utility changed.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import (MOVES_LEFT_MAX_EFFECT, MOVES_LEFT_SLOPE,  # noqa: E402
                    MOVES_LEFT_THRESHOLD)
from evaluation import NNEvaluator  # noqa: E402
from mcts import _is_reversal, _own_previous_moves  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from native_mcts import NativeMCTS  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_black_state(game):
    frames = game.get("frames") or []
    if not frames:
        return None
    state = MonsterChessGame(frames[0]["fen"])
    last = None
    for frame in frames[1:]:
        move = frame.get("move")
        if move:
            state.apply_search_action(chess.Move.from_uci(move))
        if not state.is_white_turn and not state.is_terminal():
            last = state.clone()
    return last


def run(args):
    with open(args.source, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    games = []
    for matchup in document.get("matchups", []):
        for game in matchup.get("games", []):
            if game.get("category") == "draw":
                games.append(game)

    evaluator_off = NNEvaluator(args.model)
    evaluator_on = NNEvaluator(args.model)
    common = dict(
        num_simulations=args.sims,
        batch_size=args.batch_size,
        root_noise=False,
        allow_early_stop=False,
    )
    search_off = NativeMCTS(eval_fn=evaluator_off, **common)
    search_on = NativeMCTS(
        eval_fn=evaluator_on,
        moves_left_utility=True,
        moves_left_max_effect=args.max_effect,
        moves_left_threshold=args.threshold,
        moves_left_slope=args.slope,
        **common,
    )

    rows = []
    started = time.time()
    for index, game in enumerate(games):
        state = _last_black_state(game)
        if state is None:
            continue
        original_turn_count = int(state.turn_count)
        if args.reset_turn_count:
            state.turn_count = 0
            state._terminal = False
            state._result = None
        _value, _policy, predicted_moves_left = (
            evaluator_on.evaluate_with_policy_and_moves_left(state))
        action_off, probs_off, value_off = search_off.get_best_action(
            state, temperature=0.0)
        action_on, probs_on, value_on = search_on.get_best_action(
            state, temperature=0.0)
        previous = _own_previous_moves(state)
        row = {
            "index": index,
            "seed": game.get("seed"),
            "fen": state.fen(),
            "original_turn_count": original_turn_count,
            "searched_turn_count": int(state.turn_count),
            "predicted_moves_left": predicted_moves_left,
            "off_action": action_off.uci(),
            "on_action": action_on.uci(),
            "changed": action_off != action_on,
            "off_reversal": _is_reversal(action_off, previous),
            "on_reversal": _is_reversal(action_on, previous),
            "off_value": value_off,
            "on_value": value_on,
            "off_selected_visit_share": probs_off.get(action_off.uci(), 0.0),
            "on_selected_visit_share": probs_on.get(action_on.uci(), 0.0),
        }
        rows.append(row)
        print(
            f"[{len(rows):02}/{len(games):02}] "
            f"{row['off_action']} -> {row['on_action']} "
            f"changed={row['changed']} reversal="
            f"{int(row['off_reversal'])}->{int(row['on_reversal'])}",
            flush=True,
        )

    summary = {
        "positions": len(rows),
        "changed_actions": sum(row["changed"] for row in rows),
        "reversals_off": sum(row["off_reversal"] for row in rows),
        "reversals_on": sum(row["on_reversal"] for row in rows),
        "reversals_removed": sum(
            row["off_reversal"] and not row["on_reversal"] for row in rows),
        "reversals_added": sum(
            not row["off_reversal"] and row["on_reversal"] for row in rows),
        "mean_predicted_moves_left": (
            statistics.mean(row["predicted_moves_left"] for row in rows)
            if rows else None),
    }
    result = {
        "source": args.source,
        "model": args.model,
        "model_sha256": _sha256(args.model),
        "sims": args.sims,
        "batch_size": args.batch_size,
        "reset_turn_count": args.reset_turn_count,
        "moves_left_utility": {
            "max_effect": args.max_effect,
            "threshold": args.threshold,
            "slope": args.slope,
        },
        "summary": summary,
        "positions": rows,
        "elapsed_sec": round(time.time() - started, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="complete-game exhibition JSON")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sims", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-effect", type=float,
                        default=MOVES_LEFT_MAX_EFFECT)
    parser.add_argument("--threshold", type=float,
                        default=MOVES_LEFT_THRESHOLD)
    parser.add_argument("--slope", type=float, default=MOVES_LEFT_SLOPE)
    parser.add_argument("--reset-turn-count", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
