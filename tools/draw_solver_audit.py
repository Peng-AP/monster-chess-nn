"""Are strong-model draws holds by White, or wins Black failed to convert? (2026-10-12)

Every strong-vs-strong draw on the repetition-aware engine (gen55's run)
ends with Black at least a rook ahead. A material lead is not a proof against a
double-moving king, so this asks the exact forced-capture solver
(src/forced_capture.py), CPU only, about the last Black-to-move position of
each drawn game:

    proven win     a forced king capture within the depth: Black had a win
    no win         the search finished: no forced capture within the depth
    unknown        node budget exhausted: no answer

Depth 6 Black moves, 20M nodes. Writes benchmarks/game_value_20261012/draw_solver_audit.json.
"""
import argparse
import collections
import glob
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "web")]
STRONG = ("gen_0053", "gen_0054", "gen_0055", "bootstrap_v29")
VAL = dict(q=9, r=5, b=3, n=3, p=1, k=0)


def norm(p):
    return str(p).replace(os.sep, "/")


def draws(limit_per_depth):
    out = collections.defaultdict(list)
    files = [f for f in glob.glob(str(ROOT / "benchmarks/gen55_program/gen55_20261007/production/**/*.jsonl"),
                                  recursive=True)
             if os.path.exists(f + ".manifest.json") and "/continuations/" not in norm(f)]
    for f in sorted(files):
        s = json.load(open(f + ".manifest.json"))["settings"]
        a, b = norm(s["model_a"]["path"]), norm(s["model_b"]["path"])
        if not (any(x in a for x in STRONG) and any(x in b for x in STRONG)):
            continue
        for line in open(f):
            r = json.loads(line)
            if "game" in r and r["white_score"] not in (0, 1) and len(out[s["sims"]]) < limit_per_depth:
                out[s["sims"]].append(dict(file=norm(os.path.relpath(f, ROOT)), seed=r["seed"],
                                           termination=r["game"]["termination"], trajectory=r["game"]["trajectory"]))
    return out


def last_black_to_move(trajectory):
    from monster_chess import MonsterChessGame
    for state in reversed(trajectory):
        if state["fen"].split()[1] == "b" and not state.get("half"):
            g = MonsterChessGame(state["fen"])
            g.turn_count = state.get("turn_count", 0)
            return g, state["fen"]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-depth", type=int, default=150)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=20_000_000)
    args = ap.parse_args()
    from forced_capture import try_forced_capture_move
    results, summary = [], collections.Counter()
    for sims, games in sorted(draws(args.per_depth).items()):
        for d in games:
            g, fen = last_black_to_move(d["trajectory"])
            if g is None:
                continue
            fen_board = fen.split()[0]
            white_pawns = sum(1 for c in fen_board if c.isupper() and c != "K")
            t = time.time()
            move, depth, exhausted = try_forced_capture_move(g, max_black_moves=args.depth, node_budget=args.nodes)
            verdict = "proven win" if move else "unknown" if exhausted else "no win"
            summary[(sims, "bare king" if white_pawns == 0 else "White has pawns", verdict)] += 1
            results.append(dict(sims=sims, file=d["file"], seed=d["seed"], termination=d["termination"], fen=fen,
                                white_pawns=white_pawns, verdict=verdict, depth=depth,
                                seconds=round(time.time() - t, 2)))
            print(f"{sims} {verdict:10s} depth={depth} pawns={white_pawns} {time.time()-t:5.1f}s", flush=True)
    out = ROOT / "benchmarks/game_value_20261012/draw_solver_audit.json"
    out.write_text(json.dumps(dict(depth=args.depth, nodes=args.nodes,
                                   summary={" | ".join(map(str, k)): v for k, v in sorted(summary.items())},
                                   positions=results), indent=1))
    for k, v in sorted(summary.items()):
        print("SUMMARY", k, v, flush=True)
    print("DRAW AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
