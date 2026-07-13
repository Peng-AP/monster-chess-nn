"""Paired promotion-defense probe for a candidate and incumbent model.

Each model defends the same starts against the same heuristic opponent. Reports
promotion prevention, defender-king survival, and game score independently.
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

from promotion_probe import (  # noqa: E402
    compare_probe_reports,
    play_probe_game,
    summarize_probe_results,
)


_white_engine = None
_black_engine = None
_runner_color = None
_defender_color = None


def _load_fens(path, source):
    fens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if source is not None and obj.get("source") != source:
                continue
            if obj.get("fen"):
                fens.append(obj["fen"])
    return fens


def _init_worker(model_path, defender_color, runner_color, sims):
    global _white_engine, _black_engine, _runner_color, _defender_color
    from benchmark import _build_engine

    model_engine, _ = _build_engine(model_path, sims)
    heuristic_engine, _ = _build_engine(None, sims)
    if defender_color == "white":
        _white_engine, _black_engine = model_engine, heuristic_engine
    else:
        _white_engine, _black_engine = heuristic_engine, model_engine
    _runner_color = runner_color
    _defender_color = defender_color


def _play_task(task):
    fen, seed, max_plies = task
    random.seed(seed)
    return play_probe_game(
        _white_engine, _black_engine, fen,
        runner_color=_runner_color,
        defender_color=_defender_color,
        max_plies=max_plies,
    )


def run_model_suite(model_path, fens, defender_color, runner_color, sims, seed,
                    workers, max_plies):
    tasks = [(fen, seed + i, max_plies) for i, fen in enumerate(fens)]
    worker_count = max(1, min(int(workers), len(tasks)))
    with mp.Pool(
        worker_count,
        initializer=_init_worker,
        initargs=(model_path, defender_color, runner_color, sims),
    ) as pool:
        results = pool.map(_play_task, tasks)
    return summarize_probe_results(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compare candidate/incumbent promotion defense on paired starts")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--start-fen-file", required=True)
    parser.add_argument("--source", required=True,
                        help="Exact source tag to select from the JSONL")
    parser.add_argument("--defender", choices=("white", "black"), required=True)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--max-plies", type=int, default=600)
    parser.add_argument("--max-prevention-drop", type=float, default=0.0)
    parser.add_argument("--max-king-survival-drop", type=float, default=0.0)
    parser.add_argument("--max-score-drop", type=float, default=0.05)
    parser.add_argument("--enforce", action="store_true",
                        help="Exit nonzero when any no-regression threshold fails")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = parser.parse_args()

    for model in (args.candidate, args.incumbent):
        if not os.path.isfile(model):
            raise FileNotFoundError(model)
    if args.sims <= 0 or args.workers <= 0 or args.max_plies <= 0:
        raise ValueError("--sims, --workers, and --max-plies must be > 0")

    fens = _load_fens(args.start_fen_file, args.source)
    if not fens:
        raise ValueError(f"no starts matched source {args.source!r}")
    runner_color = "black" if args.defender == "white" else "white"
    t0 = time.time()
    candidate = run_model_suite(
        args.candidate, fens, args.defender, runner_color,
        args.sims, args.seed, args.workers, args.max_plies)
    incumbent = run_model_suite(
        args.incumbent, fens, args.defender, runner_color,
        args.sims, args.seed, args.workers, args.max_plies)
    gate = compare_probe_reports(
        candidate, incumbent,
        max_prevention_drop=args.max_prevention_drop,
        max_king_survival_drop=args.max_king_survival_drop,
        max_score_drop=args.max_score_drop,
    )
    report = {
        "source": args.source,
        "runner_color": runner_color,
        "defender_color": args.defender,
        "starts": len(fens),
        "sims": args.sims,
        "seed": args.seed,
        "candidate": candidate,
        "incumbent": incumbent,
        "comparison": gate,
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(report, indent=2))

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(
        args.out_dir,
        f"promotion_probe_{args.defender}_defends_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved to {path}")
    if args.enforce and not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    mp.freeze_support()
    main()
