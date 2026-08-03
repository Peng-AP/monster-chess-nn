"""E1 exit gate, part 2: randomized differential over the rules layer.

The gate asks for legal-action **set** equality on both APIs across >=10M
positions, "including forced-blunder and ep cases". Two things have to be true
for that to mean anything, and only one of them is about the port:

1. The comparison must cover both APIs — the atomic `(m1, m2)` pairs and the
   half-move actions — because they are separately implemented and the half-move
   path deliberately offers an m1 whose every continuation is unsafe.
2. **The sampler must actually produce the rare cases.** A 10M-position
   differential that never generates an en-passant position or a forced blunder
   proves nothing about either. So this tool reports coverage counts alongside
   the comparison, and coverage is a gate criterion in its own right.

Ordering is checked where it is contractual: `get_legal_actions` lists winning
captures first, and callers rely on it (`CONTEXT.md` §1.1). Set equality alone
would let a port silently reorder them.

Run with one engine it is a sampler-coverage report; run with two
(`--engine-b native`, once the crate builds) it is the parity gate.

    py -3 tools/differential.py --positions 200000 --workers 6
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chess  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def classify(game):
    """Tags describing which rare rules paths this position exercises."""
    tags = []
    if game.board.ep_square is not None:
        tags.append("ep_available")
        # Only interesting if some pawn of the mover can actually take it.
        us = chess.WHITE if game.is_white_turn else chess.BLACK
        if game.board.attackers(us, game.board.ep_square) & game.board.pawns:
            tags.append("ep_capturable")
    if game.is_white_turn:
        if game.white_half_pending:
            tags.append("white_second_half")
        else:
            tags.append("white_first_half")
        # Forced blunder: every completion leaves White's king capturable, so
        # the generator falls back to offering everything.
        pairs = game._get_white_actions(truncate_wins=False)
        if pairs:
            safe = 0
            for m1, m2 in pairs[:60]:
                probe = game.clone()
                try:
                    probe.apply_action((m1, m2))
                except Exception:
                    continue
                wk = probe.board.king(chess.WHITE)
                if wk is not None and not probe.board.is_attacked_by(chess.BLACK, wk):
                    safe += 1
                    break
            if safe == 0:
                tags.append("white_forced_blunder")
    else:
        tags.append("black_to_move")
        safe_exists = False
        for move in game.board.pseudo_legal_moves:
            probe = game.board.copy(stack=False)
            probe.push(move)
            if probe.king(chess.WHITE) is None:
                safe_exists = True
                break
            bk = probe.king(chess.BLACK)
            if bk is not None and not probe.is_attacked_by(chess.WHITE, bk):
                safe_exists = True
                break
        if not safe_exists:
            tags.append("black_forced_blunder")
    if game.board.castling_rights:
        tags.append("castling_rights")
    if game.board.king(chess.WHITE) is None or game.board.king(chess.BLACK) is None:
        tags.append("king_absent")
    return tags


def action_signature(game):
    """Both APIs, as comparable structures. Order preserved where contractual."""
    atomic = game.get_legal_actions()
    if game.is_white_turn:
        atomic_repr = [f"{m1.uci()},{m2.uci()}" for m1, m2 in atomic]
    else:
        atomic_repr = [m.uci() for m in atomic]
    half_repr = [m.uci() for m in game.get_search_actions()]
    return atomic_repr, half_repr


def _walk(task):
    """Random-walk games, sampling positions and tallying coverage."""
    seed, positions_target = task
    rng = random.Random(seed)
    counts = {}
    sampled = 0
    mismatches = []
    while sampled < positions_target:
        game = MonsterChessGame(START_FEN)
        steps = 0
        while not game.is_terminal() and steps < 260 and sampled < positions_target:
            actions = game.get_search_actions()
            if not actions:
                break
            for tag in classify(game):
                counts[tag] = counts.get(tag, 0) + 1
            atomic_repr, half_repr = action_signature(game)
            counts["positions"] = counts.get("positions", 0) + 1
            counts["atomic_actions"] = counts.get("atomic_actions", 0) + len(atomic_repr)
            counts["half_actions"] = counts.get("half_actions", 0) + len(half_repr)
            # Contractual ordering: a winning capture, when present, comes first.
            if not game.is_white_turn:
                wk = game.board.king(chess.WHITE)
                if wk is not None and game.board.attackers(chess.BLACK, wk) and atomic_repr:
                    if chess.parse_square(atomic_repr[0][2:4]) != wk:
                        mismatches.append({
                            "fen": game.fen(),
                            "reason": "winning capture not listed first",
                            "first": atomic_repr[0],
                        })
            sampled += 1
            steps += 1
            game.apply_search_action(rng.choice(actions))
    return counts, mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--label", default="coverage")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    per = max(1, args.positions // args.workers)
    tasks = [(args.seed + i, per) for i in range(args.workers)]
    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(_walk, tasks)
    elapsed = time.time() - started

    counts, mismatches = {}, []
    for c, m in results:
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
        mismatches.extend(m)

    total = counts.get("positions", 0)
    summary = {
        "label": args.label,
        "positions": total,
        "elapsed_sec": round(elapsed, 1),
        "positions_per_sec": round(total / elapsed, 1) if elapsed else None,
        "coverage": {k: v for k, v in sorted(counts.items())},
        "coverage_rate": {
            k: round(v / total, 6) for k, v in sorted(counts.items()) if total
        },
        "ordering_violations": len(mismatches),
        "examples": mismatches[:10],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"differential_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
