"""E0(b) — did MCTS walk past forced king captures? (DIRECTIVE 2026-08-03 §3)

Scans dominant-but-unfinished games (`game_result == -0.5`: Black decisively
ahead when the move limit ran out) and asks, at every late Black-to-move
position, whether an exact AND/OR search finds a **forced** king capture the
game did not play.

Every hit is a game that was won and not finished. The count sizes purchase #2
of the rewrite — the finisher search — and it does so before any Rust exists,
which is why the directive promotes this from a sizing note to a go/no-go.

**Read the caveat with the number.** Corpus games were generated at assorted
sim counts, mostly 400-700, so a hit here says "the finisher would have helped
*these* games", not "it helps v19 at 1600 sims". The targeted 1600-sim replay
is the confirmation; this is the cheap first look, and it is free.

    py -3 tools/forced_capture_probe.py --source data/raw/combined_v19_K \
        --depth 2 --late-plies 20 --workers 6
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from forced_capture import try_forced_capture_depth  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

DOMINANT_UNFINISHED = -0.5


def iter_games(source):
    """Yield (path, records) for every .jsonl game under `source`."""
    for dirpath, _dirnames, filenames in os.walk(source):
        for name in sorted(filenames):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(dirpath, name)
            records = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if records:
                yield path, records


def select_positions(records, late_plies):
    """Late Black-to-move positions, oldest first, with their record index."""
    black = [(i, r) for i, r in enumerate(records)
             if r.get("current_player") == "black"]
    return black[-late_plies:] if late_plies else black


def _probe_game(task):
    """Probe one game's late Black-to-move positions.

    `positions` carries `moves_left` — how many Black moves remain in the game
    from that point, counting the position itself. It is the discriminator
    between the two very different things a hit can mean:

    - **walked past** (`moves_left >= depth`): Black had a forced capture *and*
      the moves to play it, and the game still ended without one. This is the
      finisher's target.
    - **clock cutoff** (`moves_left < depth`): the forced win existed but the
      move limit landed first. Verified by hand on
      `combined_v19_K/game_00157` record 224, where the solver's winning move
      `h5h1` is exactly what the game played on its final ply — a conversion
      interrupted, not missed.

    Counting cutoffs as misses would inflate the finisher's value with games it
    could not have saved, which is the number this spike exists to get right.
    """
    path, positions, depth, budget = task
    hits, exhausted, nodes_total = [], 0, 0
    for idx, fen, moves_left in positions:
        try:
            state = MonsterChessGame(fen)
        except Exception:
            continue
        if state.is_white_turn:
            continue
        d, nodes, budget_hit = try_forced_capture_depth(
            state, max_black_moves=depth, node_budget=budget)
        nodes_total += nodes
        if budget_hit:
            exhausted += 1
        elif d is not None:
            hits.append({
                "record_index": idx,
                "depth": d,
                "moves_left": moves_left,
                "verdict": "walked_past" if moves_left >= d else "clock_cutoff",
                "fen": fen,
            })
    return {
        "path": os.path.relpath(path, ROOT),
        "hits": hits,
        "positions_probed": len(positions),
        "budget_exhausted": exhausted,
        "nodes": nodes_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[],
                    help="corpus dir to scan (repeatable)")
    ap.add_argument("--depth", type=int, default=2,
                    help="max Black moves in the forced line")
    ap.add_argument("--late-plies", type=int, default=20,
                    help="probe only the last N Black-to-move positions")
    ap.add_argument("--budget", type=int, default=400_000,
                    help="node budget per position")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--label", default="corpus")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    sources = args.source or [os.path.join(ROOT, "data", "raw", "combined_v19_K")]

    tasks, scanned, dominant = [], 0, 0
    for source in sources:
        for path, records in iter_games(source):
            scanned += 1
            if records[0].get("game_result") != DOMINANT_UNFINISHED:
                continue
            dominant += 1
            black_idx = [i for i, r in enumerate(records)
                         if r.get("current_player") == "black"]
            positions = []
            for i, r in select_positions(records, args.late_plies):
                moves_left = sum(1 for j in black_idx if j >= i)
                positions.append((i, r["fen"], moves_left))
            if positions:
                tasks.append((path, positions, args.depth, args.budget))
            if args.max_games and len(tasks) >= args.max_games:
                break
        if args.max_games and len(tasks) >= args.max_games:
            break

    print(f"scanned {scanned} games, {dominant} dominant-unfinished, "
          f"probing {len(tasks)} of them at depth<={args.depth} "
          f"({args.late_plies} late plies each)", flush=True)
    if not tasks:
        print("nothing to probe")
        return

    started = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_probe_game, tasks)
    else:
        results = [_probe_game(t) for t in tasks]
    elapsed = time.time() - started

    probed = sum(r["positions_probed"] for r in results)
    all_hits = [h for r in results for h in r["hits"]]
    walked = [h for h in all_hits if h["verdict"] == "walked_past"]
    cutoff = [h for h in all_hits if h["verdict"] == "clock_cutoff"]
    games_with_hits = [r for r in results if r["hits"]]
    games_walked = [r for r in results
                    if any(h["verdict"] == "walked_past" for h in r["hits"])]
    by_depth = {}
    for h in walked:
        by_depth[h["depth"]] = by_depth.get(h["depth"], 0) + 1

    # The headline: of games that were dominant and unfinished, how many held a
    # position from which a forced capture existed?
    summary = {
        "label": args.label,
        "sources": sources,
        "depth": args.depth,
        "late_plies": args.late_plies,
        "node_budget": args.budget,
        "games_scanned": scanned,
        "games_dominant_unfinished": dominant,
        "games_probed": len(tasks),
        "positions_probed": probed,
        "games_with_forced_capture": len(games_with_hits),
        "games_walked_past_a_win": len(games_walked),
        "walked_past_rate": round(len(games_walked) / len(tasks), 4),
        "positions_with_forced_capture": len(all_hits),
        "positions_walked_past": len(walked),
        "positions_clock_cutoff": len(cutoff),
        "walked_past_hits_by_depth": {str(k): v for k, v in sorted(by_depth.items())},
        "positions_budget_exhausted": sum(r["budget_exhausted"] for r in results),
        "nodes_total": sum(r["nodes"] for r in results),
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "games": [{"path": r["path"], "hits": r["hits"]} for r in games_with_hits],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out_dir, f"forced_capture_{args.label}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "games"}, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
