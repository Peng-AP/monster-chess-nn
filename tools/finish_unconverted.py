"""Finish the games the old engine could not: turn -0.5 into a real outcome.

A `game_result` of -0.5 means the move limit arrived with Black decisively
ahead. Measured 2026-08-03, **29% of those positions held a forced king capture
Black simply walked past** — so a large share of the label is not "drawn", it is
"won and unfinished". The corpus teaches the value head that those positions are
worth -0.5 when the truth is -1.

The honest way to correct that is **not** to rewrite the label. A relabel
asserts an outcome the game never played, and the ramp value target is a
function of `plies_to_end`, so an asserted -1 with no plies attached is a
different kind of wrong. Instead this **resumes the game** from its final
position with the tools the original engine did not have — the repaired
scripted oracle (material guard + forced-capture preflight) and the native
search — and keeps the continuation only if Black actually captures the king.
What lands in the corpus is a played conversion with real plies.

Games that still cannot be converted are left exactly as they are. That number
is worth as much as the conversions: it is the honest size of the finishing
problem after every fix currently available.

    py -3 tools/finish_unconverted.py --source data/raw/combined_v19_K --limit 40
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "native"))

import chess  # noqa: E402
from evaluation import NNEvaluator, evaluate  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from native_mcts import NativeMCTS  # noqa: E402
from scripted_mate import ScriptedMate, mate_algo_applicable  # noqa: E402

DOMINANT_UNFINISHED = -0.5


def load_game(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def final_state(records):
    """Rebuild the state the game ended in, with its history and turn count.

    A FEN alone under-determines a Monster Chess position (pending flag, turn
    count, move history), so the state is rebuilt by replaying rather than
    parsed from the last record.
    """
    game = MonsterChessGame(records[0]["fen"])
    for i in range(len(records) - 1):
        want = " ".join(records[i + 1]["fen"].split()[:4])
        played = None
        for action in game.get_search_actions():
            probe = game.clone()
            probe.apply_search_action(action)
            if " ".join(probe.fen().split()[:4]) == want:
                played = action
                break
        if played is None:
            return None
        game.apply_search_action(played)
    return game


def resume(game, white_engine, black_sims, extra_turns, model_path):
    """Play on from `game`. Returns (converted, new_records, plies)."""
    records = []
    bot = ScriptedMate()          # repaired: material guard + preflight, both on
    nn = NNEvaluator(model_path)
    black_engine = NativeMCTS(num_simulations=black_sims, eval_fn=nn,
                              allow_early_stop=True)
    start_turn = game.turn_count
    while game.turn_count - start_turn < extra_turns:
        if game.board.king(chess.WHITE) is None:
            return True, records, len(records)
        actions = game.get_search_actions()
        if not actions:
            break
        if game.is_white_turn:
            action, _p, _v = white_engine.get_best_action(game, temperature=0.0)
            if action is None:
                break
            game.apply_search_action(action)
            continue
        # Black: the repaired oracle where it applies, else the native search.
        if mate_algo_applicable(game):
            move = bot.select_move(game)
            if move is None or move not in actions:
                move = None
        else:
            move = None
        if move is None:
            move, _p, _v = black_engine.get_best_action(game, temperature=0.0)
        if move is None:
            break
        records.append({
            "fen": game.fen(),
            "mcts_value": round(-evaluate(game), 4),
            "policy": {move.uci(): 1.0},
            "current_player": "black",
            "half": 0,
            "resumed": True,
        })
        game.apply_search_action(move)
    return game.board.king(chess.WHITE) is None, records, len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--out-dir", default=None,
                    help="write corrected games here (default: alongside, "
                         "suffix _finished)")
    ap.add_argument("--model", default="models/fresh_start_v19/best_value_net.pt")
    ap.add_argument("--black-sims", type=int, default=1600)
    ap.add_argument("--white-sims", type=int, default=400)
    ap.add_argument("--extra-turns", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sources = args.source or [os.path.join(ROOT, "data", "raw", "combined_v19_K")]
    targets = []
    for source in sources:
        for dirpath, _d, names in os.walk(source):
            for name in sorted(names):
                if not name.endswith(".jsonl"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    records = load_game(path)
                except Exception:
                    continue
                if records and records[0].get("game_result") == DOMINANT_UNFINISHED:
                    targets.append((path, records))
                if args.limit and len(targets) >= args.limit:
                    break
            if args.limit and len(targets) >= args.limit:
                break

    print(f"{len(targets)} unconverted (-0.5) games to attempt", flush=True)
    if not targets:
        return

    nn = NNEvaluator(args.model)
    white_engine = NativeMCTS(num_simulations=args.white_sims, eval_fn=nn,
                              allow_early_stop=True)
    out_dir = args.out_dir or os.path.join(ROOT, "data", "raw", "finished_conversions")
    os.makedirs(out_dir, exist_ok=True)

    converted = unreachable = failed = 0
    extra_plies = []
    started = time.time()
    for path, records in targets:
        game = final_state(records)
        if game is None:
            unreachable += 1
            continue
        ok, new_records, plies = resume(game, white_engine, args.black_sims,
                                        args.extra_turns, args.model)
        if not ok:
            failed += 1
            continue
        converted += 1
        extra_plies.append(plies)
        merged = [dict(r) for r in records] + new_records
        for r in merged:
            r["game_result"] = -1        # a real capture, actually played
        name = os.path.basename(path).replace(".jsonl", "_finished.jsonl")
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            for r in merged:
                fh.write(json.dumps(r) + "\n")

    elapsed = time.time() - started
    summary = {
        "attempted": len(targets),
        "converted": converted,
        "conversion_rate": round(converted / len(targets), 4) if targets else None,
        "still_unconverted": failed,
        "unreplayable": unreachable,
        "mean_extra_black_moves": (round(sum(extra_plies) / len(extra_plies), 1)
                                   if extra_plies else None),
        "black_sims": args.black_sims,
        "extra_turns": args.extra_turns,
        "out_dir": os.path.relpath(out_dir, ROOT),
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = os.path.join(ROOT, "benchmarks",
                        f"finish_unconverted_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
