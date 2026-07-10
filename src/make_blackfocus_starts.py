"""Extract backward-chained focus start FENs from won games.

For every game won by --side (default black: game_result == -1) in the input
dirs, take the position N plies before the end for each N in --offsets,
keeping only records where the winning side is to move (no White half-pending
starts). Larger offsets chain the curriculum backwards: the generator learns
to convert from positions progressively further from the win.

Output: JSONL of {"fen": ..., "source": ..., "offset": N} records, deduped
by FEN, consumable by data_generation.py --start-fen-file.
"""
import argparse
import glob
import json
import os


def _iter_games(input_dirs):
    for d in input_dirs:
        for path in sorted(glob.glob(os.path.join(d, "game_*.jsonl"))):
            with open(path, "r", encoding="utf-8") as f:
                records = [json.loads(ln) for ln in f if ln.strip()]
            if records:
                yield path, records


def _pick_side_record(records, idx, side, tolerance=2):
    """Nearest side-to-move, non-half-pending record at or below idx."""
    for j in range(idx, max(idx - tolerance, 0) - 1, -1):
        rec = records[j]
        if str(rec.get("current_player", "")).lower() == side and not rec.get("half"):
            return rec
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", action="append", required=True,
                    help="raw game dir; repeatable")
    ap.add_argument("--offsets", type=str, default="6,12,20,30",
                    help="comma-separated plies-from-end (record indices)")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--min-plies", type=int, default=8,
                    help="skip games shorter than this many records")
    ap.add_argument("--side", choices=("black", "white"), default="black",
                    help="winning side to chain from (black: result==-1, white: result==+1)")
    args = ap.parse_args()

    offsets = sorted({int(x) for x in args.offsets.split(",") if x.strip()})
    want_result = -1 if args.side == "black" else 1
    seen = set()
    out_records = []
    games_used = 0
    for path, records in _iter_games(args.input_dir):
        if records[-1].get("game_result", 0) != want_result:
            continue
        if len(records) < args.min_plies:
            continue
        games_used += 1
        for off in offsets:
            idx = len(records) - 1 - off
            if idx < 1:
                continue
            rec = _pick_side_record(records, idx, args.side)
            if rec is None:
                continue
            fen = rec.get("fen")
            if not fen or fen in seen:
                continue
            seen.add(fen)
            out_records.append({
                "fen": fen,
                "current_player": args.side,
                "source": os.path.basename(os.path.dirname(path)),
                "offset": off,
            })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(out_records)} unique {args.side}-to-move start FENs "
          f"from {games_used} {args.side}-won games to {args.output}")


if __name__ == "__main__":
    main()
