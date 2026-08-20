"""Convert an opening book into a start-position JSONL for data_generation.

WHY. Ordinary self-play starts from the true opening and diversifies with 16
plies of temperature sampling from the engine's own visit counts. Measured on
gen23 (2026-08-19): effective unique openings 0.61 at 1600 sims and 0.44 at
3200, with three openings covering 113 of 300 games. Under a book it is 1.0.
So the corpus is fed by whatever narrow distribution the engine currently
prefers, and the narrowing gets worse as the engine gets stronger. Seeding part
of generation from a book widens the training distribution without replacing
it.

WHAT IS AND IS NOT PRESERVED. A book position is FEN + white_half_pending +
turn_count (CONTEXT section 8), and a start-FEN file carries only the FEN. Two
consequences, both handled here:

  * Entries with a pending White half-move CANNOT be reconstructed from a FEN
    alone -- board.turn stays WHITE across White's pending half, so the FEN
    cannot say which half is next. Those entries are DROPPED, not guessed.
    (Books built by tools/make_book.py at whole-turn depths have none.)
  * turn_count restarts at 0, so a seeded game gets the full MAX_GAME_TURNS
    from its start rather than the remainder. For generation that is harmless
    and arguably right -- the game is played out from the position. For MATCH
    play it would not be, which is why match.py takes the book directly.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_entries(path):
    with open(path, "r", encoding="utf-8") as fh:
        book = json.load(fh)
    entries = book.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{path} has no entries")
    return book, entries


def main():
    ap = argparse.ArgumentParser(
        description="Book -> JSONL start positions for data_generation.py")
    ap.add_argument("--book", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="keep at most N entries (0 = all)")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N entries, so a generation batch can "
                         "use a block disjoint from the gate's")
    args = ap.parse_args()

    book, entries = load_entries(args.book)
    total = len(entries)
    entries = entries[args.offset:]
    if args.limit:
        entries = entries[:args.limit]

    kept, dropped_half, dropped_nofen = 0, 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for e in entries:
            fen = e.get("fen")
            if not fen:
                dropped_nofen += 1
                continue
            # make_book writes "half"; be liberal about the other spelling.
            if e.get("half") or e.get("white_half_pending"):
                dropped_half += 1
                continue
            fh.write(json.dumps({
                "fen": fen,
                "current_player": "white" if fen.split()[1] == "w" else "black",
                "source": "book",
                "book": os.path.basename(args.book),
                "book_turn_count": e.get("turn_count"),
            }) + "\n")
            kept += 1

    print(f"book        : {args.book} ({total} entries)")
    print(f"window      : offset {args.offset}, limit {args.limit or 'all'}")
    print(f"kept        : {kept}")
    print(f"dropped     : {dropped_half} with a pending White half-move, "
          f"{dropped_nofen} with no fen")
    print(f"wrote       : {args.out}")
    if not kept:
        sys.exit("no usable start positions")


if __name__ == "__main__":
    main()
