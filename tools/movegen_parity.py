"""E1: differential the native pseudo-legal movegen against python-chess.

python-chess is the reference the Python engine is built on, so it — not the
rules of chess — is what the native core must reproduce. That includes the
awkward parts: check-aware castling inside a "pseudo-legal" generator, ep only
when the FEN carries it, and king captures as ordinary moves.

Positions come from random walks of real games so they are reachable, plus the
FENs of every recorded game on disk so the distribution includes real openings
and endgames rather than only whatever random play wanders into.

    py -3 tools/movegen_parity.py --positions 50000
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
import monster_native as mn  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def reference(fen):
    return {m.uci() for m in chess.Board(fen).pseudo_legal_moves}


def native(fen):
    return set(mn.pseudo_legal_uci(fen))


def walk_positions(limit, seed):
    """Random-walk real games, yielding FENs."""
    rng = random.Random(seed)
    seen = 0
    while seen < limit:
        game = MonsterChessGame(START_FEN)
        steps = 0
        while not game.is_terminal() and steps < 240 and seen < limit:
            yield game.fen()
            seen += 1
            steps += 1
            actions = game.get_search_actions()
            if not actions:
                break
            game.apply_search_action(rng.choice(actions))


def corpus_positions(limit):
    """FENs from recorded games — real openings and endgames."""
    count = 0
    for source in ("ps_monster", "human_games"):
        base = os.path.join(ROOT, "data", "raw", source)
        for dirpath, _dirs, names in os.walk(base):
            for name in sorted(names):
                if not name.endswith(".jsonl"):
                    continue
                with open(os.path.join(dirpath, name), "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)["fen"]
                        except Exception:
                            continue
                        count += 1
                        if count >= limit:
                            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    half = args.positions // 2
    sources = [("random_walk", walk_positions(half, args.seed)),
               ("corpus", corpus_positions(args.positions - half))]

    checked = 0
    mismatches = []
    started = time.time()
    for label, gen in sources:
        for fen in gen:
            checked += 1
            try:
                got, want = native(fen), reference(fen)
            except Exception as exc:
                mismatches.append({"source": label, "fen": fen, "error": repr(exc)})
                continue
            if got != want:
                mismatches.append({
                    "source": label,
                    "fen": fen,
                    "native_only": sorted(got - want)[:12],
                    "reference_only": sorted(want - got)[:12],
                })
    elapsed = time.time() - started

    summary = {
        "positions_checked": checked,
        "mismatches": len(mismatches),
        "match_rate": round(1 - len(mismatches) / checked, 8) if checked else None,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "examples": mismatches[:25],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"movegen_parity_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
    if mismatches:
        print(f"\nfirst {min(6, len(mismatches))} mismatches:")
        for m in mismatches[:6]:
            print(" ", json.dumps(m)[:300])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
