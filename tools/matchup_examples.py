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
    """Play one game, keeping every frame.

    With no book entry the game starts from the true opening position and the
    record is a complete game -- which is what a showcase wants. Diversity then
    has to come from somewhere, because two deterministic engines replay one
    identical game however the RNG is seeded, so the first `temp_plies` are
    sampled from the search distribution at `temp`. Every move is still played
    and recorded; only the opening moves are sampled rather than argmax.
    """
    seed, entry, temp_plies, temp = task
    random.seed(seed)
    if entry is None:
        game = MonsterChessGame()
    else:
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
            game, temperature=(temp if plies < temp_plies else 0.0))
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
        "start_fen": frames[0]["fen"],
        "result": result,
        "category": classify(result),
        "plies": plies,
        "frames": frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--book", default=None,
                    help="optional. Omitted, every game is played FROM THE "
                         "TRUE OPENING POSITION and the record is a complete "
                         "game -- which is what a showcase wants. A book only "
                         "makes sense here if it is large enough to give every "
                         "game its own distinct opening; a handful of entries "
                         "reused across matchups makes every game a variation "
                         "on the same few starts.")
    ap.add_argument("--book-offset", type=int, default=0)
    ap.add_argument("--opening-temp-plies", type=int, default=8,
                    help="plies sampled from the search distribution before "
                         "play goes deterministic. Two temp-0 engines replay "
                         "one identical game, so some sampling is required; "
                         "every move is still played and recorded.")
    ap.add_argument("--opening-temp", type=float, default=1.0)
    ap.add_argument("--games", type=int, default=48,
                    help="games per matchup")
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--seed", type=int, default=515000)
    ap.add_argument("--only", default=None,
                    help="comma-separated matchup keys; default is all. Use "
                         "for expensive high-sim passes where recording all "
                         "thirteen would cost hours.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    selected = MATCHUPS
    if args.only:
        wanted = [k.strip() for k in args.only.split(",") if k.strip()]
        known = {m[0] for m in MATCHUPS}
        missing = [k for k in wanted if k not in known]
        if missing:
            raise SystemExit(f"unknown matchup keys {missing}; "
                             f"known: {sorted(known)}")
        selected = [m for m in MATCHUPS if m[0] in wanted]

    entries = None
    if args.book:
        book_path = ROOT / args.book if not Path(args.book).is_absolute() \
            else Path(args.book)
        entries = json.loads(book_path.read_text(encoding="utf-8"))["entries"]
        need = args.book_offset + args.games
        if need > len(entries):
            raise SystemExit(
                f"book has {len(entries)} entries; offset {args.book_offset} "
                f"plus {args.games} games needs {need}. A showcase book needs "
                f"one distinct opening per game, not a block reused across "
                f"matchups.")

    started = time.time()
    out_matchups = []
    seen_games: set = set()   # global: a duplicate is a duplicate anywhere
    for key, label, white, black, note in selected:
        print(f"\n=== {label} ({args.games} games @ {args.sims}) ===",
              flush=True)
        if entries is None:
            block = [None] * args.games
        else:
            block = entries[args.book_offset:args.book_offset + args.games]
        tasks = [(args.seed + i, entry, args.opening_temp_plies,
                  args.opening_temp) for i, entry in enumerate(block)]
        kept: dict[str, list] = {c: [] for c in CATEGORIES}
        tally = {c: 0 for c in CATEGORIES}
        plies_total = 0
        duplicates = 0

        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(white, black, args.sims, args.engine)) as pool:
            for game in pool.imap_unordered(_play_recorded, tasks):
                # A repeated opening between deterministic engines replays the
                # SAME GAME, and showing it twice is padding. The key is the
                # sequence of SETTLED positions -- those not mid-White-turn --
                # because White moves twice and playing its two half-moves in
                # either order transposes to the same position, which a
                # move-list key would miss. `seen_games` is shared across
                # matchups: different White models converge on identical lines
                # against the same opponent more often than one would guess
                # (four did, in the first full-game run), and a duplicate is a
                # duplicate wherever it shows up.
                signature = tuple(f["fen"] for f in game["frames"]
                                  if f.get("half") != 1)
                if signature in seen_games:
                    duplicates += 1
                    continue
                seen_games.add(signature)
                tally[game["category"]] += 1
                plies_total += game["plies"]
                if len(kept[game["category"]]) < args.per_category:
                    kept[game["category"]].append(game)
                # Stop as soon as every category is full. Playing the whole
                # block and selecting afterwards throws away most of the games
                # -- at 2 per category that was 26 of 32 discarded, and at high
                # sims each discarded game costs real minutes. --games is now a
                # ceiling rather than a quota, so a matchup whose categories
                # fill early simply stops.
                if all(len(kept[c]) >= args.per_category for c in CATEGORIES):
                    pool.terminate()
                    break

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
            "duplicates_rejected": duplicates,
            "tally": {LABEL[c]: tally[c] for c in CATEGORIES},
            "white_share_of_decisive": (round(tally["white_win"] / decisive, 4)
                                        if decisive else None),
            "mean_plies": round(plies_total / played, 1) if played else None,
            "games": selected,
        })
        print(f"  White {tally['white_win']}  Black {tally['black_win']}  "
              f"draw {tally['draw']}  | dup {duplicates} | kept {len(selected)} | "
              f"{(time.time() - started) / 60:.1f}m elapsed", flush=True)

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sims": args.sims,
        "book": args.book or "none (full games from the start)",
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
