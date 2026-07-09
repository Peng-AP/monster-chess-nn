"""Build a deck of start positions for human curriculum play (play.ipynb).

The human is the only agent that knows either side's winning plan; this deck
serves them positions from the corpus slices nobody else can teach:

  - pawn-phase, Black to move  (wP >= 2, Black army >= 12): the cliff —
    restarts by engines from here are ~7-33% Black; the human demonstrates.
  - pawn-phase, White to move  (same filter): the White plan void — the NN
    White freezes because no data shows White activity succeeding.
  - own-game offsets at/before the cliff (>= 20 plies back from the end of
    the human's Black wins): positions amplification cannot amplify.
  - move-limit "fortress" endings (optional --endings-jsonl from a benchmark
    replay): eval says Black is winning; the human proves or refutes it.

Output: JSONL {"fen", "current_player", "source"} consumable by the
curriculum cell in play.ipynb (human plays the side to move).

    py -3 tools/make_human_deck.py --corpus-dir data/raw/heuristic_v6 \
        --own-games-dir data/raw/human_games/black_2026_07 \
        --endings-jsonl <replay_results.jsonl> \
        --output data/start_fens/human_deck_v1.jsonl
"""
import argparse
import glob
import json
import os
import random


def _wp(fen):
    return fen.split()[0].count("P")


def _bm(fen):
    return sum(1 for c in fen.split()[0] if c.islower() and c != "k")


def _side(fen):
    return "white" if fen.split()[1] == "w" else "black"


def _iter_records(dirs):
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "game_*.jsonl"))):
            with open(path, encoding="utf-8") as f:
                records = [json.loads(ln) for ln in f if ln.strip()]
            if records:
                yield path, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", action="append", required=True,
                    help="raw game dir to sample pawn-phase positions from; repeatable")
    ap.add_argument("--own-games-dir", action="append", default=[],
                    help="human Black-win dir for at/before-cliff offsets; repeatable")
    ap.add_argument("--endings-jsonl", default=None,
                    help="replay results JSONL with final_fen fields (fortress endings)")
    ap.add_argument("--per-side", type=int, default=60,
                    help="pawn-phase positions per side to move")
    ap.add_argument("--own-offsets", default="20,26,32,38,44")
    ap.add_argument("--min-wp", type=int, default=2)
    ap.add_argument("--min-bm", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    deck = []

    def add(fen, source):
        key = " ".join(fen.split()[:2])
        if key in seen:
            return
        seen.add(key)
        deck.append({"fen": fen, "current_player": _side(fen), "source": source})

    # --- pawn-phase corpus positions (skip half-pending White records:
    # FEN cannot represent a spent first half-move) ---
    pool = {"white": [], "black": []}
    for _path, records in _iter_records(args.corpus_dir):
        for rec in records:
            if rec.get("half"):
                continue
            fen = rec.get("fen")
            if not fen or _wp(fen) < args.min_wp or _bm(fen) < args.min_bm:
                continue
            pool[_side(fen)].append(fen)
    for side in ("black", "white"):
        rng.shuffle(pool[side])
        for fen in pool[side][:args.per_side]:
            add(fen, f"pawnphase_{side}")

    # --- own-game offsets at/before the cliff (Black-won games only) ---
    offsets = sorted({int(x) for x in args.own_offsets.split(",") if x.strip()})
    for _path, records in _iter_records(args.own_games_dir):
        if records[-1].get("game_result", 0) != -1:
            continue
        for off in offsets:
            idx = len(records) - 1 - off
            if idx < 1:
                continue
            rec = records[idx]
            if rec.get("half") or not rec.get("fen"):
                continue
            add(rec["fen"], f"own_offset_{off}")

    # --- fortress endings from a benchmark replay ---
    if args.endings_jsonl and os.path.isfile(args.endings_jsonl):
        with open(args.endings_jsonl, encoding="utf-8") as f:
            for ln in f:
                rec = json.loads(ln)
                if rec.get("result") in (-0.5, 0.5, 0) and rec.get("final_fen"):
                    add(rec["final_fen"], "fortress_ending")

    rng.shuffle(deck)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in deck:
            f.write(json.dumps(entry) + "\n")

    from collections import Counter
    counts = Counter(e["source"].rsplit("_", 1)[0] if e["source"].startswith("own_offset")
                     else e["source"] for e in deck)
    print(f"Wrote {len(deck)} positions to {args.output}")
    for src, n in sorted(counts.items()):
        print(f"  {src}: {n}")


if __name__ == "__main__":
    main()
