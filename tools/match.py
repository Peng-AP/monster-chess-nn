"""Parallel head-to-head match between two models (or a model and the anchor).

NN-vs-NN games need opening temperature (see benchmark.play_one): at pure
temp 0 both engines are deterministic and every game is identical. Workers
parallelize across games — a match that takes hours sequentially finishes
in minutes.

    py -3 tools/match.py --model-a models/fresh_start_v14/best_value_net.pt \\
        --model-b models/fresh_start_v12/best_value_net.pt --games 20
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

from config import (DEFAULT_GAME_WORKERS, C_PUCT, FPU_REDUCTION,
                    POLICY_TEMPERATURE)  # noqa: E402  (needs sys.path above)

_engines = {}


def match_game_seeds(games, seed):
    """Return the per-game RNG seeds used by a match, in task order.

    Exposed so multi-leg protocols can prove that their opening samples are
    disjoint instead of re-implementing this easy-to-misread layout.
    """
    n_white = games // 2
    return ([seed + i for i in range(n_white)] +
            [seed + 1000 + i for i in range(games - n_white)])


def resolve_opening_temp_plies(model_b, requested):
    '''Use sampled openings only when both opponents are neural models.'''
    if requested is not None:
        return int(requested)
    return 16 if model_b else 0


def _init_worker(model_a, model_b, sims, sims_b=None,
                 batch_a=None, batch_b=None, engine=None,
                 c_puct_a=C_PUCT, c_puct_b=C_PUCT,
                 fpu_reduction_a=FPU_REDUCTION,
                 fpu_reduction_b=FPU_REDUCTION,
                 policy_temperature_a=POLICY_TEMPERATURE,
                 policy_temperature_b=POLICY_TEMPERATURE):
    # Workers are separate processes: the choice must be passed in, not read
    # from a parent-side global.
    from benchmark import _build_engine
    _engines["a"], _ = _build_engine(
        model_a, sims, batch_a, engine=engine, c_puct=c_puct_a,
        fpu_reduction=fpu_reduction_a,
        policy_temperature=policy_temperature_a)
    _engines["b"], _ = _build_engine(
        model_b, sims_b or sims, batch_b, engine=engine, c_puct=c_puct_b,
        fpu_reduction=fpu_reduction_b,
        policy_temperature=policy_temperature_b)


def load_book(path):
    """Return (entries, metadata) for a book written by tools/make_book.py.

    A book is a pinned artifact: scores measured under one are not comparable
    to scores measured under another, or to the temperature-sampled openings
    that preceded books entirely. The metadata rides into the match artifact so
    a reader can tell which regime produced a number.
    """
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    entries = doc.get("entries") or []
    if not entries:
        raise SystemExit(f"book {path} has no entries")
    meta = {k: doc[k] for k in
            ("schema_version", "model", "model_sha256", "plies", "sims",
             "temperature", "search_policy_temperature", "seed",
             "oversample", "created", "stats")
            if k in doc}
    digest = hashlib.sha256()
    with open(full, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    meta["book_sha256"] = digest.hexdigest()
    meta["path"] = path.replace("\\", "/")
    meta["entries"] = len(entries)
    return entries, meta


def build_tasks(games, seed, opening_temp_plies, entries=None, book_name="",
                offset=0):
    """Return the per-game task list.

    Extracted from run_match so the pairing layout is testable without a GPU
    or a process pool -- the layout is the whole mechanism, and getting it
    silently wrong would produce plausible numbers with no pairing in them.
    """
    if entries is None:
        # SEED SEPARATION: per-game seeds are seed+i and seed+1000+i, so two
        # runs whose seeds differ by less than ~1000+games/2 replay overlapping
        # games. Seeds one apart share 19 of 20 -- a "fresh seed" re-run then
        # reproduces the first result exactly and looks like reassuring
        # agreement. Space independent samples by 100000 or more.
        # tests/test_match_seed_separation.py
        n_white = games // 2
        seeds = match_game_seeds(games, seed)
        return ([(True, s, opening_temp_plies, None, None)
                 for s in seeds[:n_white]] +
                [(False, s, opening_temp_plies, None, None)
                 for s in seeds[n_white:]])
    if games % 2:
        raise SystemExit(
            f"paired book play needs an even --games (got {games}); each "
            f"opening is played twice with colours reversed")
    if offset < 0 or offset + games // 2 > len(entries):
        raise SystemExit(
            f"book {book_name} has {len(entries)} entries but {games} games "
            f"at offset {offset} need entries "
            f"[{offset}, {offset + games // 2}). Play from a book is "
            f"deterministic, so reusing an entry would replay an identical "
            f"game rather than add a sample -- build a larger book instead.")
    # Both halves of a pair share a seed: play is deterministic at temp 0, and
    # a shared RNG stream keeps any residual tie-break randomness common to the
    # pair so it cancels along with the colour term.
    #
    # `offset` is how a multi-leg protocol draws a DISJOINT sample. Under a
    # book, independence lives in the entry index, not the seed: the gate's
    # confirmation replay is meant to be a fresh sample, and two legs at
    # different seeds but the same offset would replay identical openings and
    # agree by construction.
    return [(i % 2 == 0, seed + i // 2, 0, entries[offset + i // 2], i // 2)
            for i in range(games)]


def _play(task):
    """task = (a_is_white, seed, temp_plies, start, pair) -> per-game record."""
    from benchmark import play_one
    a_is_white, seed, temp_plies, start, pair = task
    random.seed(seed)
    white = _engines["a"] if a_is_white else _engines["b"]
    black = _engines["b"] if a_is_white else _engines["a"]
    if start is None:
        result, plies, _dec, opening = play_one(
            white, black, opening_temp_plies=temp_plies,
            return_opening=True)
    else:
        result, plies, _dec, opening = play_one(
            white, black, start_fen=start["fen"],
            opening_temp_plies=temp_plies,
            start_half=start.get("half", False),
            start_turn_count=start.get("turn_count", 0),
            return_opening=True)
    return ((result if a_is_white else -result), plies, a_is_white, pair,
            opening)


def _parts(record):
    """Read current five-field and legacy/test four-field result records."""
    result, plies, a_is_white, pair = record[:4]
    opening = record[4] if len(record) > 4 else None
    return result, plies, a_is_white, pair, opening


def game_score(result):
    """A's score in [0, 1] for one game, under the gate's win rule.

    Must agree with benchmark.summarize_side: only a king capture is a win, so
    the +-0.5 move-limit relabel scores as a draw. tests/test_opening_book.py
    pins the two together -- a silent divergence would make the paired standard
    error describe a different scoring rule from the headline score.
    """
    if result >= 1:
        return 1.0
    if result <= -1:
        return 0.0
    return 0.5


def paired_stats(results):
    """Standard error from colour-reversed pairs rather than from single games.

    Each book entry is played twice, once with A as White and once as Black.
    Summing the pair cancels both the position's own bias and the colour gap
    -- which at 0.31 is the single largest per-game variance term in this game
    -- so the SE of the mean pair score is well below the SE computed as if
    the games were independent. This is the whole reason the book exists.

    Returns None unless every counted pair is complete: a half-finished pair
    carries the colour term it was supposed to cancel.
    """
    if not results or any(_parts(row)[3] is None for row in results):
        return None
    by_pair = {}
    for row in results:
        result, _plies, a_is_white, pair, _opening = _parts(row)
        by_pair.setdefault(pair, {})[bool(a_is_white)] = game_score(result)
    scores = [0.5 * (sides[True] + sides[False])
              for sides in by_pair.values() if len(sides) == 2]
    n = len(scores)
    if n < 2:
        return None
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / (n - 1)
    return {"pairs": n, "pair_mean": round(mean, 4),
            "se_paired": round((var / n) ** 0.5, 4),
            "pairs_incomplete": len(by_pair) - n}


def opening_stats(results):
    """Describe the opening states actually reached by a match.

    For sampled openings, distinct seeds are not evidence of distinct games.
    Once temperature turns off, equal state + equal model colours replay the
    same deterministic continuation.  ``effective_unique_games`` therefore
    counts unique ``(A colour, complete Monster state)`` keys.  The raw state
    count is also reported so book and sampler diagnostics can be compared.
    """
    observed = []
    for row in results:
        _result, _plies, a_is_white, _pair, opening = _parts(row)
        if not opening:
            continue
        state = (opening.get("fen"), bool(opening.get("half", False)),
                 int(opening.get("turn_count", 0)))
        observed.append((bool(a_is_white), state, opening))
    if not observed:
        return None

    state_counts = {}
    colour_counts = {True: {}, False: {}}
    incomplete = 0
    for a_is_white, state, opening in observed:
        state_counts[state] = state_counts.get(state, 0) + 1
        bucket = colour_counts[a_is_white]
        bucket[state] = bucket.get(state, 0) + 1
        incomplete += not bool(opening.get("complete", False))

    multiplicities = {}
    for count in state_counts.values():
        multiplicities[str(count)] = multiplicities.get(str(count), 0) + 1
    canonical = "\n".join(
        f"{count}\t{int(half)}\t{turn}\t{fen}"
        for (fen, half, turn), count in sorted(state_counts.items()))
    top = sorted(state_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "games_observed": len(observed),
        "unique_states": len(state_counts),
        "unique_fraction": round(len(state_counts) / len(observed), 4),
        "unique_as_white": len(colour_counts[True]),
        "unique_as_black": len(colour_counts[False]),
        "effective_unique_games": (len(colour_counts[True]) +
                                   len(colour_counts[False])),
        "effective_unique_fraction": round(
            (len(colour_counts[True]) + len(colour_counts[False])) /
            len(observed), 4),
        "duplicate_games": len(observed) - len(state_counts),
        "max_multiplicity": max(state_counts.values()),
        "multiplicity_histogram": multiplicities,
        "incomplete_openings": int(incomplete),
        "states_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "top_duplicates": [
            {"count": count, "fen": state[0], "half": state[1],
             "turn_count": state[2]}
            for state, count in top if count > 1
        ],
    }


def _write_checkpoint(path, results, done, games, t0):
    """Best-effort partial artifact. Never let a checkpoint failure kill a run."""
    try:
        w, b, score = _aggregate(results)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"partial": True, "games_requested": games,
                       "games_played": done, "a_score": round(score, 4),
                       "a_as_white": w, "a_as_black": b,
                       "opening_diversity": opening_stats(results),
                       "elapsed_sec": round(time.time() - t0, 1),
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                      fh, indent=2)
        os.replace(tmp, path)
    except Exception as exc:      # a partial write must never abort the match
        print(f"  (checkpoint failed: {exc})", flush=True)


def _aggregate(results):
    """Fold finished games into (white, black, score). Used for the final
    artifact and for every partial checkpoint, so a resumed read of a killed
    run is the same shape as a completed one."""
    from benchmark import summarize_side
    white_games = [(r, p) for r, p, aw, _pair, _opening
                   in map(_parts, results) if aw]
    black_games = [(r, p) for r, p, aw, _pair, _opening
                   in map(_parts, results) if not aw]
    w = summarize_side(white_games)
    b = summarize_side(black_games)
    played = len(results)
    score = ((w["wins"] + b["wins"] + 0.5 * (w["draws"] + b["draws"])) / played
             if played else 0.0)
    return w, b, score


def run_match(model_a, model_b, games, sims, seed, opening_temp_plies=None,
              workers=None, sims_b=None, batch_a=None, batch_b=None,
              engine=None, c_puct_a=C_PUCT, c_puct_b=C_PUCT,
              fpu_reduction_a=FPU_REDUCTION,
              fpu_reduction_b=FPU_REDUCTION,
              policy_temperature_a=POLICY_TEMPERATURE,
              policy_temperature_b=POLICY_TEMPERATURE,
              stall_timeout=600.0, checkpoint_path=None, book=None,
              book_offset=0):
    """Play a match and return the result dict. The only producer of this schema.

    Callers that need several legs (tools/gate.py) go through here rather than
    re-implementing the pool, so there is exactly one JSON shape to read --
    a_score / a_as_white / a_as_black. benchmark.py's run_benchmark emits a
    *different* shape (candidate_score / white_strength / black_strength);
    confusing the two has cost a whole gate run before (HANDOFF SS10.1).
    """
    opening_temp_plies = resolve_opening_temp_plies(model_b, opening_temp_plies)
    workers = workers or DEFAULT_GAME_WORKERS
    book_meta = None

    if book:
        entries, book_meta = load_book(book)
        # The book IS the diversity, so opening sampling goes off. Leaving it
        # on would draw the opening from the candidate's own policy again and
        # put back exactly the candidate-dependence the book removes.
        opening_temp_plies = 0
        tasks = build_tasks(games, seed, 0, entries, book_name=book,
                            offset=book_offset)
        book_meta = dict(book_meta, offset=book_offset,
                         entries_used=[book_offset, book_offset + games // 2])
    else:
        tasks = build_tasks(games, seed, opening_temp_plies)

    t0 = time.time()
    pool = mp.Pool(
        workers, initializer=_init_worker,
        initargs=(model_a, model_b, sims, sims_b, batch_a, batch_b,
                  engine, c_puct_a, c_puct_b, fpu_reduction_a,
                  fpu_reduction_b, policy_temperature_a,
                  policy_temperature_b))
    try:
        iterator = pool.imap_unordered(_play, tasks)
        results = []
        # A 200-game match at 1600 sims runs for the better part of an hour and
        # used to print nothing until it was over, so "working" and "hung" were
        # indistinguishable from the log. Report periodically instead.
        step = max(1, games // 20)
        for _ in tasks:
            try:
                results.append(iterator.next(timeout=stall_timeout))
                done = len(results)
                if done % step == 0 or done == games:
                    rate = (time.time() - t0) / done
                    print(f"  [{done}/{games}] {(time.time() - t0) / 60:.1f}m "
                          f"elapsed, ~{rate * (games - done) / 60:.1f}m left",
                          flush=True)
                    # Results lived only in this process's memory until the
                    # final write, so killing a 400-game match at game 399
                    # discarded every one of them. Checkpoint the aggregate so
                    # a long run is always salvageable.
                    if checkpoint_path:
                        _write_checkpoint(checkpoint_path, results, done, games,
                                          t0)
            except mp.TimeoutError as exc:
                raise TimeoutError(
                    f"match made no progress for {stall_timeout:.0f}s "
                    f"({games - len(results)} games remain)") from exc
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    w, b, score = _aggregate(results)

    name_a = os.path.basename(os.path.dirname(model_a)) or "model-a"
    name_b = (os.path.basename(os.path.dirname(model_b))
              if model_b else "heuristic")
    return {
        "match": f"{name_a} vs {name_b}",
        "name_a": name_a, "name_b": name_b,
        "games": games, "sims": sims, "sims_b": sims_b or sims,
        "batch_a": batch_a, "batch_b": batch_b, "seed": seed,
        "search_a": {
            "c_puct": c_puct_a,
            "fpu_reduction": fpu_reduction_a,
            "policy_temperature": policy_temperature_a,
        },
        "search_b": {
            "c_puct": c_puct_b,
            "fpu_reduction": fpu_reduction_b,
            "policy_temperature": policy_temperature_b,
        },
        "opening_temp_plies": opening_temp_plies,
        # Present only for book matches. Its absence marks a score measured
        # under temperature-sampled openings, which is a different regime and
        # not comparable -- see tools/make_book.py.
        "book": book_meta,
        "paired": paired_stats(results),
        "opening_diversity": opening_stats(results),
        "workers": workers,
        "a_score": round(score, 4),
        "a_as_white": w, "a_as_black": b,
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_artifact_path": checkpoint_path,
    }


def _artifact_path(args):
    """Where this match will be written -- decided BEFORE the first game so
    partial checkpoints and the final artifact share one path."""
    if args.report_path:
        path = os.path.abspath(args.report_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    os.makedirs(args.out_dir, exist_ok=True)
    name_a = os.path.basename(os.path.dirname(args.model_a)) or "model-a"
    name_b = (os.path.basename(os.path.dirname(args.model_b))
              if args.model_b else "heuristic")
    return os.path.join(
        args.out_dir,
        f"match_{name_a}_vs_{name_b}_{time.strftime('%Y%m%d_%H%M%S')}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="candidate model (.pt)")
    ap.add_argument("--model-b", default=None,
                    help="opponent model (.pt); omit for the heuristic anchor")
    ap.add_argument("--games", type=int, default=20, help="total games (half per color)")
    ap.add_argument("--engine", choices=("python", "native"), default=None,
                    help="search engine; defaults to MONSTER_ENGINE or python")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--sims-b", type=int, default=None,
                    help="model-b simulations (default: --sims). Use for "
                         "equal-TIME comparisons when the two configs differ "
                         "in simulations per second.")
    ap.add_argument("--batch-a", type=int, default=None)
    ap.add_argument("--batch-b", type=int, default=None)
    ap.add_argument("--c-puct-a", type=float, default=C_PUCT)
    ap.add_argument("--c-puct-b", type=float, default=C_PUCT)
    ap.add_argument("--fpu-reduction-a", type=float, default=FPU_REDUCTION)
    ap.add_argument("--fpu-reduction-b", type=float, default=FPU_REDUCTION)
    ap.add_argument("--policy-temperature-a", type=float,
                    default=POLICY_TEMPERATURE)
    ap.add_argument("--policy-temperature-b", type=float,
                    default=POLICY_TEMPERATURE)
    ap.add_argument("--seed", type=int, default=20260704)
    # Left as None so resolve_opening_temp_plies() can pick the default from the
    # opponent: heuristic tie-breaks already diversify anchor games, so only
    # NN-vs-NN matches need sampled model openings.
    ap.add_argument("--opening-temp-plies", type=int, default=None,
                    help="default: 16 for NN-vs-NN, 0 vs the heuristic anchor")
    ap.add_argument("--book", default=None,
                    help="opening book from tools/make_book.py. Plays each "
                         "entry twice with colours reversed and turns opening "
                         "sampling off. Scores are NOT comparable to non-book "
                         "scores.")
    ap.add_argument("--book-offset", type=int, default=0,
                    help="first book entry to use. Reserve a disjoint block "
                         "for any run that must not replay another's openings "
                         "-- under a book a fresh seed changes nothing.")
    ap.add_argument("--workers", type=int, default=DEFAULT_GAME_WORKERS)
    ap.add_argument("--stall-timeout", type=float, default=600.0,
                    help="fail if no game completes for this many seconds")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    ap.add_argument("--report-path", default=None,
                    help="write the report to this exact path")
    args = ap.parse_args()

    if args.stall_timeout <= 0:
        ap.error("--stall-timeout must be positive")
    out = run_match(args.model_a, args.model_b, args.games, args.sims,
                    args.seed, args.opening_temp_plies, args.workers,
                    sims_b=args.sims_b, batch_a=args.batch_a,
                    batch_b=args.batch_b, engine=args.engine,
                    c_puct_a=args.c_puct_a, c_puct_b=args.c_puct_b,
                    fpu_reduction_a=args.fpu_reduction_a,
                    fpu_reduction_b=args.fpu_reduction_b,
                    policy_temperature_a=args.policy_temperature_a,
                    policy_temperature_b=args.policy_temperature_b,
                    stall_timeout=args.stall_timeout,
                    checkpoint_path=_artifact_path(args), book=args.book,
                    book_offset=args.book_offset)
    path = out["_artifact_path"]
    out.pop("_artifact_path", None)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
