"""Parallel head-to-head match between two models (or a model and the anchor).

NN-vs-NN games need opening temperature (see benchmark.play_one): at pure
temp 0 both engines are deterministic and every game is identical. Workers
parallelize across games — a match that takes hours sequentially finishes
in minutes.

    py -3 tools/match.py --model-a models/fresh_start_v14/best_value_net.pt \\
        --model-b models/fresh_start_v12/best_value_net.pt --games 20
"""
import argparse
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


def _play(task):
    """task = (a_is_white, seed, temp_plies) -> (a_result, plies)."""
    from benchmark import play_one
    a_is_white, seed, temp_plies = task
    random.seed(seed)
    white = _engines["a"] if a_is_white else _engines["b"]
    black = _engines["b"] if a_is_white else _engines["a"]
    result, plies, _dec = play_one(white, black, opening_temp_plies=temp_plies)
    return (result if a_is_white else -result), plies, a_is_white


def run_match(model_a, model_b, games, sims, seed, opening_temp_plies=None,
              workers=None, sims_b=None, batch_a=None, batch_b=None,
              engine=None, c_puct_a=C_PUCT, c_puct_b=C_PUCT,
              fpu_reduction_a=FPU_REDUCTION,
              fpu_reduction_b=FPU_REDUCTION,
              policy_temperature_a=POLICY_TEMPERATURE,
              policy_temperature_b=POLICY_TEMPERATURE,
              stall_timeout=600.0):
    """Play a match and return the result dict. The only producer of this schema.

    Callers that need several legs (tools/gate.py) go through here rather than
    re-implementing the pool, so there is exactly one JSON shape to read --
    a_score / a_as_white / a_as_black. benchmark.py's run_benchmark emits a
    *different* shape (candidate_score / white_strength / black_strength);
    confusing the two has cost a whole gate run before (HANDOFF SS10.1).
    """
    opening_temp_plies = resolve_opening_temp_plies(model_b, opening_temp_plies)
    workers = workers or DEFAULT_GAME_WORKERS

    # SEED SEPARATION: per-game seeds are seed+i and seed+1000+i, so two runs
    # whose seeds differ by less than ~1000+games/2 replay overlapping games.
    # Seeds one apart share 19 of 20 -- a "fresh seed" re-run then reproduces
    # the first result exactly and looks like reassuring agreement. Space
    # independent samples by 100000 or more. tests/test_match_seed_separation.py
    n_white = games // 2
    seeds = match_game_seeds(games, seed)
    tasks = [(True, s, opening_temp_plies) for s in seeds[:n_white]]
    tasks += [(False, s, opening_temp_plies) for s in seeds[n_white:]]

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
        for _ in tasks:
            try:
                results.append(iterator.next(timeout=stall_timeout))
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

    from benchmark import summarize_side
    white_games = [(r, p) for r, p, aw in results if aw]
    black_games = [(r, p) for r, p, aw in results if not aw]
    w = summarize_side(white_games)
    b = summarize_side(black_games)
    score = (w["wins"] + b["wins"] + 0.5 * (w["draws"] + b["draws"])) / games

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
        "workers": workers,
        "a_score": round(score, 4),
        "a_as_white": w, "a_as_black": b,
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


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
                    stall_timeout=args.stall_timeout)
    name_a, name_b = out["name_a"], out["name_b"]

    if args.report_path:
        path = os.path.abspath(args.report_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir,
                            f"match_{name_a}_vs_{name_b}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
