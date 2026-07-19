"""Extract take-into-promotion motifs from the owner's games into a deck.

The models' promotion blindness is not a runner-race failure: the owner's
promotions arrive through capture sequences that land on or clear the
promotion square, embedded in ordinary positions (owner, 2026-07-19). Those
motifs exist nowhere but his recorded games, with exact ply indices. This
tool walks his games, finds every promotion he played (any move of a White
m1,m2 pair or a Black move carrying a promotion suffix), and emits start
positions a few decisions BEFORE the tactic, at side-to-move = the promoting
(human) side.

Uses: (1) owner sessions — he re-executes the motif against a stronger
defender (deck cell in play.ipynb); (2) engine restarts at high sims for
contrast data — the defender either finds the prevention or the outcome
label carries the punishment.

Output JSONL {"fen", "current_player", "source"} — same shape as
make_human_deck.py decks.

    py -3 tools/make_promo_deck.py \
        --games-dir data/raw/human_games/white_2026_07 \
        --games-dir data/raw/human_games/black_2026_07 \
        --games-dir data/raw/human_games/curriculum_2026_07 \
        --output data/start_fens/promo_motif_deck_v1.jsonl
"""
import argparse
import glob
import json
import os


def _moves_of(record):
    """UCI half-moves of the decision actually played (max-prob policy key)."""
    policy = record.get("policy") or {}
    if not policy:
        return []
    key = max(policy, key=policy.get)
    return key.split(",")


def _is_promotion(uci):
    return len(uci) == 5 and uci[4] in "qrbn"


def _side(fen):
    return "white" if fen.split()[1] == "w" else "black"


def find_promotions(records):
    """Indices of records where the human played a promotion."""
    hits = []
    for i, rec in enumerate(records):
        if rec.get("actor") != "human" or rec.get("half") or not rec.get("fen"):
            continue
        if any(_is_promotion(m) for m in _moves_of(rec)):
            hits.append(i)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dir", action="append", required=True,
                    help="human game dir to scan; repeatable")
    ap.add_argument("--offsets", default="3,6,9",
                    help="decisions before the promotion to emit (same-side records)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    offsets = sorted({int(x) for x in args.offsets.split(",") if x.strip()})
    seen = set()
    deck = []
    n_motifs = 0

    for d in args.games_dir:
        for path in sorted(glob.glob(os.path.join(d, "game_*.jsonl"))):
            with open(path, encoding="utf-8") as f:
                records = [json.loads(ln) for ln in f if ln.strip()]
            for promo_idx in find_promotions(records):
                n_motifs += 1
                promo_side = records[promo_idx].get("current_player")
                # Walk back collecting earlier decisions by the promoting side.
                same_side = [j for j in range(promo_idx)
                             if records[j].get("current_player") == promo_side
                             and not records[j].get("half")
                             and records[j].get("fen")]
                for off in offsets:
                    if off > len(same_side):
                        continue
                    fen = records[same_side[-off]]["fen"]
                    key = " ".join(fen.split()[:2])
                    if key in seen or _side(fen) != promo_side:
                        continue
                    seen.add(key)
                    deck.append({"fen": fen, "current_player": promo_side,
                                 "source": f"promo_motif_{promo_side}_m{off}"})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in deck:
            f.write(json.dumps(entry) + "\n")

    from collections import Counter
    counts = Counter(e["source"] for e in deck)
    print(f"{n_motifs} human promotions found; wrote {len(deck)} positions "
          f"to {args.output}")
    for src, n in sorted(counts.items()):
        print(f"  {src}: {n}")


if __name__ == "__main__":
    main()
