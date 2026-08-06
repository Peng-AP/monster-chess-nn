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
import hashlib
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


def resume_state(records):
    """Rebuild the position the game stopped in, ready to play on.

    Replaying the game forward is not possible: generation retains records
    side-selectively (`data_generation._should_skip_record`), so consecutive
    records are not consecutive plies and no single action connects them.

    The final record is enough on its own. `fen` plus `half` pins the position
    exactly, because the state machine keeps `board.turn` pointing at the side
    genuinely to move even between White's two half-moves.

    Two deliberate losses. `turn_count` restarts at 0, because these games
    stopped *at* MAX_GAME_TURNS and would otherwise be terminal on arrival —
    playing past the cap is the entire point. And the move stack starts empty,
    so the oscillation penalty is inactive for the first few plies.
    """
    last = records[-1]
    game = MonsterChessGame(last["fen"])
    game.white_half_pending = bool(last.get("half"))
    return game


def resume(game, white_engine, black_engine, bot, extra_turns):
    """Play on from `game`. Returns (converted, new_records, applied_plies)."""
    records = []
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
    ap.add_argument("--model", default="models/fresh_start_v20/best_value_net.pt")
    ap.add_argument("--black-sims", type=int, default=1600)
    ap.add_argument("--white-sims", type=int, default=400)
    ap.add_argument("--extra-turns", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sources = args.source or [os.path.join(ROOT, "data", "raw", "combined_v19_K")]
    targets = []
    seen = set()
    duplicates = 0
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
                if not records or records[0].get(
                        "game_result") != DOMINANT_UNFINISHED:
                    continue
                # The corpus variants are largely copies of one another: 631
                # matching files hold 208 distinct games. Without this, most of
                # the run re-finishes games it already finished, and the
                # conversion rate is weighted by how often a game was copied.
                digest = hashlib.sha1(
                    json.dumps(records, sort_keys=True).encode()).hexdigest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                targets.append((path, records))
                if args.limit and len(targets) >= args.limit:
                    break
            if args.limit and len(targets) >= args.limit:
                break

    print(f"{len(targets)} unconverted (-0.5) games to attempt "
          f"({duplicates} duplicate copies skipped)", flush=True)
    if not targets:
        return

    # One model, one set of engines, reused across every game. Building these
    # per game reloads the net from disk each time and dominates the run.
    nn = NNEvaluator(args.model)
    white_engine = NativeMCTS(num_simulations=args.white_sims, eval_fn=nn,
                              allow_early_stop=True)
    black_engine = NativeMCTS(num_simulations=args.black_sims, eval_fn=nn,
                              allow_early_stop=True)
    bot = ScriptedMate()          # repaired: material guard + preflight, both on
    out_dir = args.out_dir or os.path.join(ROOT, "data", "raw", "finished_conversions")
    os.makedirs(out_dir, exist_ok=True)

    converted = unreachable = failed = 0
    extra_plies = []
    started = time.time()
    for index, (path, records) in enumerate(targets, 1):
        game = resume_state(records)
        if game is None:
            unreachable += 1
            continue
        ok, new_records, plies = resume(game, white_engine, black_engine, bot,
                                        args.extra_turns)
        rate = converted / max(1, index - 1) if index > 1 else 0.0
        print(f"[{index}/{len(targets)}] {'CONVERTED' if ok else 'no'} "
              f"({plies} black moves) running rate {rate:.1%} "
              f"elapsed {(time.time() - started) / 60:.1f}m", flush=True)
        if not ok:
            failed += 1
            continue
        converted += 1
        extra_plies.append(plies)
        merged = [dict(r) for r in records] + new_records
        for r in merged:
            r["game_result"] = -1        # a real capture, actually played
        # The ramp target is a function of plies_to_end, and extending a game
        # invalidates every original one. Recompute over the merged list in the
        # same record-index units the generator uses.
        for i, r in enumerate(merged):
            r["plies_to_end"] = len(merged) - 1 - i
        name = os.path.basename(path).replace(".jsonl", "_finished.jsonl")
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            for r in merged:
                fh.write(json.dumps(r) + "\n")

    elapsed = time.time() - started
    summary = {
        "sources": [os.path.relpath(s, ROOT) for s in sources],
        "attempted": len(targets),
        "duplicate_copies_skipped": duplicates,
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
