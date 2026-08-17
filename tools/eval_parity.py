"""E2 exit gate: the native heuristic must match `evaluation.py` to 1e-9.

The gate is 1e-9 because that is what the directive demands, but the honest
target is *bit-identical*: every operation in the heuristic is +, -, *, / on
f64 plus an integer power, so there is no libm call to diverge on. The only
real hazard is accumulation order, and an ordering mistake produces a delta far
larger than 1e-9 — so a run reporting exactly 0.0 is evidence the term order
matches, not just that the arithmetic is close.

Positions come from random walks (reachable states, all phases) and from every
recorded game on disk (real openings and endgames), plus the pending-aware
White-threat case, which is the one branch a random walk under-samples.

    py -3 tools/eval_parity.py --positions 200000
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

import monster_native as mn  # noqa: E402
from evaluation import evaluate  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def walk_states(limit, seed):
    rng = random.Random(seed)
    seen = 0
    while seen < limit:
        game = MonsterChessGame(START_FEN)
        for _ in range(240):
            if game.is_terminal() or seen >= limit:
                break
            yield game
            seen += 1
            actions = game.get_search_actions()
            if not actions:
                break
            game.apply_search_action(rng.choice(actions))


def corpus_states(limit):
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
                            rec = json.loads(line)
                            game = MonsterChessGame(rec["fen"])
                        except Exception:
                            continue
                        # Recorded White records carry the half; replay it so the
                        # pending-aware threat scan is exercised from real data.
                        if rec.get("half") == 1:
                            game.white_half_pending = True
                        yield game
                        count += 1
                        if count >= limit:
                            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--tolerance", type=float, default=1e-9)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    half = args.positions // 2
    checked = 0
    beyond = 0
    exact = 0
    worst = 0.0
    worst_case = None
    pending_seen = 0
    fen_lossy = 0
    started = time.time()

    for label, states in (("random_walk", walk_states(half, args.seed)),
                          ("corpus", corpus_states(args.positions - half))):
        for game in states:
            py = evaluate(game)
            fen = game.fen()
            rs = mn.evaluate_fen(fen, game.is_white_turn, game.white_half_pending)

            # A FEN cannot carry an ep square that has no legal capture
            # (python-chess writes `en_passant="legal"`), so a live game object
            # holds state its own FEN drops. Evaluating the round-trip tells us
            # whether a disagreement is the port's fault or the format's.
            mirror = MonsterChessGame(fen)
            mirror.white_half_pending = game.white_half_pending
            mirror.is_white_turn = game.is_white_turn
            py_fen = evaluate(mirror)
            if py != py_fen:
                fen_lossy += 1
                py = py_fen  # judge the port against what the FEN can express

            delta = abs(py - rs)
            checked += 1
            if game.white_half_pending:
                pending_seen += 1
            if delta == 0.0:
                exact += 1
            if delta > worst:
                worst = delta
                worst_case = {"source": label, "fen": game.fen(),
                              "is_white_turn": game.is_white_turn,
                              "white_half_pending": game.white_half_pending,
                              "python": py, "native": rs}
            if delta > args.tolerance:
                beyond += 1
    elapsed = time.time() - started

    summary = {
        "positions_checked": checked,
        "beyond_tolerance": beyond,
        "tolerance": args.tolerance,
        "bit_identical": exact,
        "bit_identical_rate": round(exact / checked, 6) if checked else None,
        "worst_delta": worst,
        "pending_positions_seen": pending_seen,
        "fen_lossy_positions": fen_lossy,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "worst_case": worst_case,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"eval_parity_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
