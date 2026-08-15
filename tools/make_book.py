"""Build a fixed opening book for paired match play.

WHY A BOOK AT ALL. `tools/match.py` diversifies NN-vs-NN games by playing the
first 16 plies at temperature 0.5, sampled from the engine's own visit counts
(`benchmark.play_one`). That breaks determinism -- two temp-0 nets replay one
identical game no matter how the RNG is seeded -- but the opening is drawn from
the *candidate's own policy*. Two candidates measured against the same
incumbent at the same seed therefore start from different positions, their
scores are independent samples, and their variances add. That is why ranking 14
checkpoints in `tools/checkpoint_screen.py` costs as many games as generation
does.

A book fixes the position set. Every candidate faces the identical openings,
and match.py plays each entry twice with colours reversed, so the position's
own bias and the 0.31 colour gap cancel *inside a pair* instead of being
averaged down across hundreds of games.

WHAT IS DELIBERATELY NOT COPIED FROM TCEC. Their books are curated for
balance: positions where neither engine is winning, so games are decisive but
fair. Monster Chess is not a balanced game -- White scores about 0.67 from the
start -- and selecting for Black-playable openings would make the gate measure
a different game from the one the corpus is drawn from and the one the owner
plays in `play.ipynb`. Entries here are sampled from the bar model's own
self-play at the temperature the harness already used, so the distribution is
unchanged; only its *assignment to candidates* changes.

A book is a pinned artifact, like the gate bar. Changing it silently
invalidates comparison against every score measured under the old one.

    py -3 tools/make_book.py --model models/fresh_start_v21b/best_value_net.pt \
        --entries 400 --out books/v21b_p16.json
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import DEFAULT_GAME_WORKERS, POLICY_TEMPERATURE  # noqa: E402

SCHEMA_VERSION = 1
_engine = {}


def _init_worker(model, sims, engine):
    from benchmark import _build_engine
    # Match the arena exactly: policy logits enter search at the configured
    # engine default (1.0), while ``temperature`` below samples only the final
    # root visit counts. Applying 0.5 at both layers made the old builder much
    # more deterministic than tools/match.py despite claiming parity.
    _engine["e"], _ = _build_engine(model, sims, None, engine=engine)


def _walk(task):
    """Play `plies` sampled plies from the start and return the position.

    Returns None if the walk ended early -- a game that is already terminal
    inside the opening is not usable as a starting position.
    """
    from benchmark import _apply
    from monster_chess import MonsterChessGame

    seed, plies, temp = task
    random.seed(seed)
    game = MonsterChessGame()
    for _ in range(plies):
        if game.is_terminal():
            return None
        action, _probs, _val = _engine["e"].get_best_action(
            game, temperature=temp)
        if action is None:
            return None
        _apply(game, action)
    if game.is_terminal():
        return None
    # A position is FEN *plus* the half-move flag: during White's pending half
    # board.turn stays WHITE, so the FEN alone cannot say which half is next.
    # turn_count travels too -- MonsterChessGame(fen) restarts it at 0, which
    # would silently hand every book game ~8 extra turns before the 150-turn
    # cap and shift the draw rate relative to games played from the start.
    return {"fen": game.fen(), "half": bool(game.white_half_pending),
            "turn_count": int(game.turn_count)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_one(model, entries, plies, sims, temperature, seed, workers,
               engine, seen, book, oversample=1.6):
    """Walk one model until it has contributed `entries` fresh positions."""
    # Terminal walks and duplicates both cost entries, so ask for more than
    # needed rather than discovering the shortfall after the pool closes.
    attempts = max(entries, int(entries * oversample))
    tasks = [(seed + i, plies, temperature) for i in range(attempts)]
    start = len(book)
    completed = 0
    dropped_duplicate = 0

    t0 = time.time()
    pool = mp.Pool(workers, initializer=_init_worker,
                   initargs=(model, sims, engine))
    dropped_terminal = 0
    try:
        for i, res in enumerate(pool.imap_unordered(_walk, tasks), 1):
            completed = i
            if res is None:
                dropped_terminal += 1
            else:
                key = (res["fen"], res["half"], res["turn_count"])
                if key not in seen:
                    seen.add(key)
                    book.append(res)
                else:
                    dropped_duplicate += 1
            if i % max(1, attempts // 5) == 0:
                print(f"  [{i}/{attempts}] {len(book) - start} from this "
                      f"model, {(time.time() - t0) / 60:.1f}m", flush=True)
            if len(book) - start >= entries:
                break
    finally:
        pool.terminate()
        pool.join()
    return {"model": os.path.relpath(model, ROOT).replace("\\", "/"),
            "contributed": len(book) - start,
            "attempts_requested": attempts,
            "attempts_completed": completed,
            "dropped_terminal": dropped_terminal,
            "dropped_duplicate": dropped_duplicate,
            "minutes": round((time.time() - t0) / 60, 2)}


def build(models, entries, plies, sims, temperature, seed, workers, engine,
          oversample=1.6):
    """Return (book_entries, stats), drawing equal shares from each model.

    Several models are supported because a book inherits the opening TASTE of
    whatever produced it. Drawn from the bar alone, it quietly favours that
    lineage: models that like the same openings are measured on ground they
    already understand, and the tilt grows as the bar ages. Equal shares from
    several models spread that out.

    They are walked in sequence rather than together: each pass loads one model
    into the workers, so peak VRAM stays at one model's worth however many are
    mixed.
    """
    workers = workers or DEFAULT_GAME_WORKERS
    seen, book, per_model = set(), [], []
    share = -(-entries // len(models))          # ceil, so the total is covered
    for index, model in enumerate(models):
        print(f"\n[{index + 1}/{len(models)}] "
              f"{os.path.basename(os.path.dirname(model))}: "
              f"up to {share} entries", flush=True)
        per_model.append(_build_one(
            model, share, plies, sims, temperature,
            # Stride the seeds so two models never walk the same RNG stream.
            seed + 1000000 * index, workers, engine, seen, book,
            oversample=oversample))

    stats = {"per_model": per_model, "unique": len(book),
             "build_minutes": round(sum(m["minutes"] for m in per_model), 2)}
    return book[:entries], stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, action="append", dest="models",
                    help="model whose self-play the openings are drawn from. "
                         "Repeat it to mix provenance in equal shares, which "
                         "keeps the book from inheriting one lineage's "
                         "opening taste.")
    ap.add_argument("--entries", type=int, default=400,
                    help="unique positions; a match plays 2 games per entry, "
                         "so 400 supports an 800-game match")
    ap.add_argument("--plies", type=int, default=16,
                    help="opening depth (matches the harness default)")
    ap.add_argument("--sims", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=90000000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--oversample", type=float, default=1.6,
                    help="attempts per requested entry for each model. Increase "
                         "this without changing depth/distribution when shallow "
                         "walks collide (default: 1.6)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.oversample < 1.0:
        ap.error("--oversample must be >= 1.0")

    models = [m if os.path.isabs(m) else os.path.join(ROOT, m)
              for m in args.models]
    for m in models:
        if not os.path.isfile(m):
            raise SystemExit(f"no such model: {m}")
    print(f"building {args.entries} entries at {args.plies} plies, "
          f"temp {args.temperature}, {args.sims} sims from "
          f"{len(models)} model(s): "
          f"{', '.join(os.path.basename(os.path.dirname(m)) for m in models)}",
          flush=True)

    book, stats = build(models, args.entries, args.plies, args.sims,
                        args.temperature, args.seed, args.workers, args.engine,
                        oversample=args.oversample)
    if len(book) < args.entries:
        raise SystemExit(
            f"only {len(book)} of {args.entries} unique positions found; "
            f"deepen --plies (shallow walks collide often) or mix in another "
            f"--model. Stats: {stats}")

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "model": ", ".join(os.path.relpath(m, ROOT).replace("\\", "/")
                           for m in models),
        "model_sha256": ", ".join(sha256(m) for m in models),
        "plies": args.plies, "sims": args.sims,
        "temperature": args.temperature, "seed": args.seed,
        "search_policy_temperature": POLICY_TEMPERATURE,
        "oversample": args.oversample,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stats": stats,
        "entries": book,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\nwrote {len(book)} entries -> {out}\n  {stats}", flush=True)


if __name__ == "__main__":
    main()
