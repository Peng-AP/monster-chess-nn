"""Harvest cliff starts: pawn-phase positions Black has to convert.

The pawn-phase cliff is law 1 -- Black converts 91-100% once White's pawns are
gone and 7-14% from the 3-4 pawn positions the owner wins 100% of. Everything
in Phase 2 needs starts from *inside* that phase, and they have to come from
games where conversion was actually demonstrated, not from wishful sampling.

Sources, in the order D3 specifies:
  * the owner's won-as-Black games -- his conversions are the existence proof
    that the technique is real;
  * ps_monster games Black won -- 41.2% pawn phase against this corpus's 9.4%,
    and the only opening variety the project has (law 3.1);
  * anything else passed with --raw-dir.

Selection: Black to move, White has >= --min-white-pawns pawns on the board,
the game was won by Black, and the position is not already terminal. Dedup is
by placement + side to move, so the same structure reached by different move
orders counts once.

    py -3 tools/make_cliff_deck.py \\
        --raw-dir data/raw/combined_v19_base/human_games --require-black-win \\
        --raw-dir data/raw/ps_monster --require-black-win \\
        --count 300 --output data/start_fens/cliff_starts_v2.jsonl
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from monster_chess import MonsterChessGame  # noqa: E402


def white_pawns(fen):
    return sum(1 for c in fen.split()[0] if c == "P")


def black_to_move(fen):
    parts = fen.split()
    return len(parts) > 1 and parts[1] == "b"


def dedup_key(fen):
    """Placement + side to move: same structure, different move order = one."""
    return " ".join(fen.split()[:2])


def game_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def harvest(raw_dir, min_white_pawns, require_black_win, offset_from_end):
    """Yield candidate entries from every game under raw_dir."""
    for dirpath, _dirs, files in os.walk(raw_dir):
        for name in sorted(files):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(dirpath, name)
            try:
                recs = game_records(path)
            except (json.JSONDecodeError, OSError):
                continue
            if not recs:
                continue
            result = recs[-1].get("game_result", 0)
            if require_black_win and not (result is not None and result < 0):
                continue
            # Skip the last few plies: a position two moves from a king capture
            # teaches nothing about converting.
            usable = recs[:-offset_from_end] if offset_from_end else recs
            for rec in usable:
                fen = rec.get("fen")
                if not fen or not black_to_move(fen):
                    continue
                if white_pawns(fen) < min_white_pawns:
                    continue
                yield {
                    "fen": fen,
                    "current_player": "black",
                    "source": os.path.basename(dirpath) or os.path.basename(raw_dir),
                    "white_pawns": white_pawns(fen),
                    "game_result": result,
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", action="append", required=True)
    ap.add_argument("--require-black-win", action="store_true",
                    help="only harvest from games Black actually won")
    ap.add_argument("--min-white-pawns", type=int, default=3,
                    help="the cliff is wP>=3 (default: 3)")
    ap.add_argument("--offset-from-end", type=int, default=6,
                    help="drop the last N records of each game (default: 6)")
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--cap-per-source", type=int, default=None,
                    help="at most N starts from any one source directory. "
                         "Without it a large source swamps the deck: "
                         "ps_monster alone yields 1753 candidates against the "
                         "owner's 212, and an unweighted sample of 300 came "
                         "out 89%% ps -- burying the conversions that are the "
                         "whole reason the owner's games are in here.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = {}
    scanned = 0
    for raw in args.raw_dir:
        if not os.path.isdir(raw):
            print(f"  skip missing {raw}")
            continue
        before = len(seen)
        for entry in harvest(raw, args.min_white_pawns,
                             args.require_black_win, args.offset_from_end):
            scanned += 1
            key = dedup_key(entry["fen"])
            if key in seen:
                continue
            # Validate against the live engine: a start nothing can be played
            # from is worse than no start at all.
            try:
                game = MonsterChessGame(fen=entry["fen"])
            except ValueError:
                continue
            if game.is_terminal() or not game.get_legal_actions():
                continue
            seen[key] = entry
        print(f"  {raw}: +{len(seen) - before} distinct (running total {len(seen)})")

    entries = list(seen.values())
    rng.shuffle(entries)
    if args.cap_per_source:
        kept, per_source = [], Counter()
        for e in entries:
            if per_source[e["source"]] >= args.cap_per_source:
                continue
            per_source[e["source"]] += 1
            kept.append(e)
        entries = kept
    entries = entries[:args.count]
    entries.sort(key=lambda e: e["fen"])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    pawns = Counter(e["white_pawns"] for e in entries)
    srcs = Counter(e["source"] for e in entries)
    print(f"\nwrote {len(entries)} starts to {args.output}")
    print(f"  white pawns: {dict(sorted(pawns.items()))}")
    print(f"  sources: {dict(srcs.most_common(8))}")
    print(f"  candidate positions scanned: {scanned}")


if __name__ == "__main__":
    main()
