"""Record head-to-head games for exhibition, across a curated set of matchups.

`selfplay_examples.py` records one model against itself; this records A against
B, both colour assignments, over several opponents in one pass, and keeps a
balanced sample of each outcome so the export shows what a model does rather
than only what it does when it wins.

Openings come from a pinned book at a **reserved exhibition block** so these
games are reproducible and cannot be confused with, or contaminate, the blocks
the screens and gates use. Nothing here is evidence: the artifact is
illustrative, and every score quoted alongside it comes from the binding runs.

Sims default to 400 -- the gate operating point -- so the play shown is the
play the measured numbers describe.

    py -3 tools/matchup_examples.py --out benchmarks/gen9_exhibition.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark import _apply, _build_engine  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

_ENGINES: dict = {}

GEN9 = "models/candidates/gen9_scratch/screen_nominee.pt"
GEN7 = "models/candidates/gen7_scratch/screen_nominee.pt"
GEN8 = "models/candidates/gen8_scratch/screen_nominee.pt"
GEN10 = "models/candidates/gen10_scratch/screen_nominee.pt"
V21B = "models/fresh_start_v21b/best_value_net.pt"
V21 = "models/fresh_start_v21/best_value_net.pt"
V20 = "models/fresh_start_v20/best_value_net.pt"
RAMP = "models/rejected/fresh_start_v18_ramp/best_value_net.pt"

# (key, label, white model, black model, note)
# Gen9 appears on both sides of every pairing that has an opponent, because a
# single colour assignment shows half of what a model is.
MATCHUPS = [
    ("self", "Gen9 vs Gen9", GEN9, GEN9,
     "Self-play. The colour skew with everything else held equal."),
    ("gen7_w", "Gen9 (White) vs Gen7", GEN9, GEN7,
     "Gen9 attacking the bar it had to beat."),
    ("gen7_b", "Gen7 (White) vs Gen9", GEN7, GEN9,
     "Gen9 defending against the bar. The leg the gate decides on."),
    ("v21b_w", "Gen9 (White) vs v21b", GEN9, V21B,
     "Against the historical gate bar."),
    ("v21b_b", "v21b (White) vs Gen9", V21B, GEN9,
     "Gen9 as Black against the strongest promoted engine."),
    ("v21_w", "Gen9 (White) vs v21", GEN9, V21,
     "Against the model that holds the version number."),
    ("v21_b", "v21 (White) vs Gen9", V21, GEN9,
     "Gen9 as Black against the numbered incumbent."),
    ("gen10_w", "Gen9 (White) vs Gen10", GEN9, GEN10,
     "The successor that failed. Gen10 traded White for a small Black gain."),
    ("gen10_b", "Gen10 (White) vs Gen9", GEN10, GEN9,
     "The rejected successor on the attack."),
    ("gen8_w", "Gen9 (White) vs Gen8", GEN9, GEN8,
     "Against the generation that failed on a 0.005 floor miss."),
    ("v20_b", "v20 (White) vs Gen9", V20, GEN9,
     "Two promotions back, as White."),
    ("ramp_b", "v18 ramp (White) vs Gen9", RAMP, GEN9,
     "The distinct-style floor-bearing sparring opponent."),
    ("anchor_b", "Heuristic (White) vs Gen9", None, GEN9,
     "The fixed yardstick. No network on White at all."),
]

CATEGORIES = ("black_win", "white_win", "draw")
LABEL = {"black_win": "Black win", "white_win": "White win", "draw": "Draw"}


def classify(result: float) -> str:
    """Match protocol: only a king capture wins; a cap ending is a draw."""
    if result >= 1:
        return "white_win"
    if result <= -1:
        return "black_win"
    return "draw"


def _init_worker(white_model, black_model, sims, engine):
    def build(path):
        if path is None:
            return _build_engine(None, sims, engine=engine)[0]
        full = str(ROOT / path) if not Path(path).is_absolute() else path
        return _build_engine(full, sims, engine=engine)[0]
    _ENGINES["white"] = build(white_model)
    _ENGINES["black"] = build(black_model)


def _play_recorded(task):
    """Play one game from a book position, keeping every frame."""
    seed, entry = task
    random.seed(seed)
    game = MonsterChessGame(entry["fen"])
    game.white_half_pending = bool(entry.get("half", False))
    game.turn_count = int(entry.get("turn_count", 0))

    frames = [{"fen": game.fen(), "move": None, "actor": None, "half": None}]
    plies = 0
    while not game.is_terminal() and plies < 600:
        is_white = bool(game.is_white_turn)
        pending = bool(getattr(game, "white_half_pending", False))
        engine = _ENGINES["white" if is_white else "black"]
        action, _probabilities, _value = engine.get_best_action(
            game, temperature=0.0)
        if action is None:
            break
        uci = action.uci()
        _apply(game, action)
        plies += 1
        frames.append({
            "fen": game.fen(),
            "move": uci,
            "actor": "White" if is_white else "Black",
            "half": 2 if (is_white and pending) else (1 if is_white else None),
        })

    result = float(game.get_result())
    return {
        "seed": seed,
        "start_fen": entry["fen"],
        "result": result,
        "category": classify(result),
        "plies": plies,
        "frames": frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--book",
                    default="books/gate_mixed_v21b_gen7_gen9_p16_20260815.json")
    ap.add_argument("--book-offset", type=int, default=400,
                    help="reserved exhibition block; keep clear of the gate "
                         "blocks (0-219) and the tuner reservation (680-747)")
    ap.add_argument("--games", type=int, default=48,
                    help="games per matchup")
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--seed", type=int, default=515000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    book_path = ROOT / args.book if not Path(args.book).is_absolute() \
        else Path(args.book)
    entries = json.loads(book_path.read_text(encoding="utf-8"))["entries"]
    need = args.book_offset + args.games
    if need > len(entries):
        raise SystemExit(
            f"book has {len(entries)} entries; offset {args.book_offset} plus "
            f"{args.games} games needs {need}")

    started = time.time()
    out_matchups = []
    for key, label, white, black, note in MATCHUPS:
        print(f"\n=== {label} ({args.games} games @ {args.sims}) ===",
              flush=True)
        block = entries[args.book_offset:args.book_offset + args.games]
        tasks = [(args.seed + i, entry) for i, entry in enumerate(block)]
        kept: dict[str, list] = {c: [] for c in CATEGORIES}
        tally = {c: 0 for c in CATEGORIES}
        plies_total = 0

        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(white, black, args.sims, args.engine)) as pool:
            for game in pool.imap_unordered(_play_recorded, tasks):
                tally[game["category"]] += 1
                plies_total += game["plies"]
                if len(kept[game["category"]]) < args.per_category:
                    kept[game["category"]].append(game)

        played = sum(tally.values())
        decisive = tally["white_win"] + tally["black_win"]
        selected = []
        for category in CATEGORIES:
            for ordinal, game in enumerate(kept[category], 1):
                game = dict(game)
                game["label"] = f"{LABEL[category]} {ordinal}"
                selected.append(game)

        out_matchups.append({
            "key": key, "label": label, "note": note,
            "white": white or "heuristic anchor",
            "black": black or "heuristic anchor",
            "games_played": played,
            "tally": {LABEL[c]: tally[c] for c in CATEGORIES},
            "white_share_of_decisive": (round(tally["white_win"] / decisive, 4)
                                        if decisive else None),
            "mean_plies": round(plies_total / played, 1) if played else None,
            "games": selected,
        })
        print(f"  White {tally['white_win']}  Black {tally['black_win']}  "
              f"draw {tally['draw']}  | kept {len(selected)} | "
              f"{(time.time() - started) / 60:.1f}m elapsed", flush=True)

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sims": args.sims,
        "book": args.book,
        "book_offset": args.book_offset,
        "games_per_matchup": args.games,
        "matchups": out_matchups,
    }
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {out} in {(time.time() - started) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
