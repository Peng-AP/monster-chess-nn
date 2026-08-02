"""Harvest promotion-defense positions: Black to move, White pawn one square
from queening, and a legal Black capture of it available.

This is the deck behind M2/M3. The failure it exists to measure is HANDOFF
SS4.4: dup1, defending, declined to capture a pawn about to promote and rated
the blunder above the capture. A static probe over 400 such positions found no
population-wide *prior* deficit, but the failure lives in Q, not the priors --
so the deck has to be re-runnable through search, and therefore committed
rather than left in a session scratchpad (the earlier one was, and is gone).

Selection is deliberately narrow: every position here has a concrete, legal,
immediately available answer to the threat. A model that declines is not
choosing between subtleties.

    py -3 tools/make_promotion_defense_deck.py \
        --raw-dir data/raw/combined_v19_base --raw-dir data/raw/ps_monster \
        --count 400 --output data/start_fens/promotion_defense_deck_v1.jsonl
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chess  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

SEVENTH = 6  # 0-indexed rank a White pawn queens from


def promoting_pawns(board):
    return [sq for sq in board.pieces(chess.PAWN, chess.WHITE)
            if chess.square_rank(sq) == SEVENTH]


def capture_moves(game, targets):
    """Legal Black moves capturing one of `targets`."""
    out = []
    for action in game.get_legal_actions():
        move = action if isinstance(action, chess.Move) else None
        if move is not None and move.to_square in targets:
            out.append(move)
    return out


def qualifies(fen):
    """-> (game, targets, captures) when the position is a probe position."""
    try:
        game = MonsterChessGame(fen=fen)
    except ValueError:
        return None
    if game.is_white_turn or game.is_terminal():
        return None
    targets = set(promoting_pawns(game.board))
    if not targets:
        return None
    caps = capture_moves(game, targets)
    if not caps:
        return None
    return game, targets, caps


def scan(raw_dir):
    for dirpath, _dirs, files in os.walk(raw_dir):
        for name in sorted(files):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line).get("fen"), path
                    except json.JSONDecodeError:
                        continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", action="append", required=True)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    found = {}
    scanned = 0
    for raw in args.raw_dir:
        for fen, path in scan(raw):
            scanned += 1
            if not fen or fen in found:
                continue
            got = qualifies(fen)
            if got is None:
                continue
            _game, targets, caps = got
            found[fen] = {
                "fen": fen,
                "current_player": "black",
                "source": os.path.basename(os.path.dirname(path)) or os.path.basename(raw),
                "promoting_pawns": sorted(chess.square_name(s) for s in targets),
                "captures": sorted(m.uci() for m in caps),
            }
        print(f"{raw}: {scanned} records scanned, {len(found)} qualifying so far",
              flush=True)

    positions = list(found.values())
    random.Random(args.seed).shuffle(positions)
    positions = positions[:args.count]
    # Stable order in the file; the sampling above is what the seed controls.
    positions.sort(key=lambda p: p["fen"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in positions:
            f.write(json.dumps(p) + "\n")

    multi = sum(1 for p in positions if len(p["promoting_pawns"]) > 1)
    print(f"\nwrote {len(positions)} positions to {args.output}")
    print(f"  {multi} have more than one pawn on the seventh")
    print(f"  mean legal captures available: "
          f"{sum(len(p['captures']) for p in positions) / max(len(positions), 1):.2f}")


if __name__ == "__main__":
    main()
