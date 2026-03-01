"""Named presets for running iterate.py with reproducible settings.

Usage examples:
  py -3 src/iterate_presets.py --show-presets
  py -3 src/iterate_presets.py --preset smoke --dry-run
  py -3 src/iterate_presets.py --preset daily --seed 20260221
  py -3 src/iterate_presets.py --preset overnight -- --human-eval-value-diagnostics
"""

import argparse
import os
import subprocess
import sys


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
ITERATE_PATH = os.path.join(SRC_DIR, "iterate.py")


PRESETS = {
    "smoke": {
        "description": "Fast gate/processing sanity run with bounded data caps",
        "expected_runtime": "~10-20 minutes on a midrange CUDA GPU",
        "artifacts": [
            "data/raw/nn_gen*",
            "data/processed/processing_summary.json",
            "models/candidates/gen_*/",
            "models/iterate_run_*.json",
        ],
        "args": [
            "--iterations", "1",
            "--games", "24",
            "--curriculum-games", "24",
            "--black-focus-games", "24",
            "--human-seed-games", "12",
            "--simulations", "40",
            "--curriculum-simulations", "20",
            "--black-focus-simulations", "30",
            "--human-seed-simulations", "40",
            "--epochs", "2",
            "--warmup-epochs", "0",
            "--position-budget", "12000",
            "--position-budget-max", "20000",
            "--max-processed-positions", "75000",
            "--alternating",
            "--opponent-sims", "40",
            "--pool-size", "4",
            "--arena-games", "12",
            "--arena-sims", "30",
            "--arena-workers", "2",
            "--black-focus-arena-games", "12",
            "--black-focus-arena-sims", "30",
            "--black-focus-gate-threshold", "0.40",
            "--gate-threshold", "0.54",
            "--gate-min-side-score", "0.45",
            "--gate-min-other-side", "0.15",
            "--human-seed-dir", "data/raw/human_games",
            "--human-seed-side", "auto",
            "--seed", "20260221",
        ],
    },
    "daily": {
        "description": "Balanced daily training batch with full gating",
        "expected_runtime": "~2-4 hours on a midrange CUDA GPU",
        "artifacts": [
            "data/raw/nn_gen* (+ _curriculum/_blackfocus/_humanseed)",
            "data/processed/processing_summary.json",
            "models/candidates/gen_*/",
            "models/arena_runs/gen_*/",
            "models/iterate_run_*.json",
        ],
        "args": [
            "--iterations", "2",
            "--games", "120",
            "--curriculum-games", "140",
            "--black-focus-games", "160",
            "--human-seed-games", "80",
            "--simulations", "100",
            "--curriculum-simulations", "40",
            "--black-focus-simulations", "80",
            "--human-seed-simulations", "140",
            "--epochs", "8",
            "--warmup-epochs", "2",
            "--warmup-start-factor", "0.1",
            "--position-budget", "180000",
            "--position-budget-max", "260000",
            "--max-processed-positions", "500000",
            "--alternating",
            "--opponent-sims", "120",
            "--pool-size", "6",
            "--arena-games", "60",
            "--arena-sims", "60",
            "--arena-workers", "4",
            "--black-focus-arena-games", "80",
            "--black-focus-arena-sims", "60",
            "--black-focus-gate-threshold", "0.40",
            "--gate-threshold", "0.54",
            "--gate-min-side-score", "0.45",
            "--gate-min-other-side", "0.15",
            "--human-seed-dir", "data/raw/human_games",
            "--human-seed-side", "auto",
            "--seed", "20260221",
            "--human-eval",
        ],
    },
    "overnight": {
        "description": "Long-run overnight training with strict gated promotion",
        "expected_runtime": "~8-12 hours on a midrange CUDA GPU",
        "artifacts": [
            "data/raw/nn_gen* (+ _curriculum/_blackfocus/_humanseed)",
            "data/processed/processing_summary.json",
            "models/candidates/gen_*/",
            "models/arena_runs/gen_*/",
            "models/iterate_run_*.json",
            "models/human_eval_incumbent_latest.json",
        ],
        "args": [
            "--iterations", "6",
            "--games", "180",
            "--curriculum-games", "220",
            "--black-focus-games", "260",
            "--human-seed-games", "140",
            "--simulations", "300",
            "--curriculum-simulations", "80",
            "--black-focus-simulations", "200",
            "--human-seed-simulations", "150",
            "--epochs", "12",
            "--warmup-epochs", "3",
            "--warmup-start-factor", "0.1",
            "--position-budget", "220000",
            "--position-budget-max", "280000",
            "--max-processed-positions", "500000",
            "--alternating",
            "--opponent-sims", "200",
            "--pool-size", "6",
            "--arena-games", "80",
            "--arena-sims", "120",
            "--arena-workers", "4",
            "--black-focus-arena-games", "150",
            "--black-focus-arena-sims", "120",
            "--black-focus-gate-threshold", "0.40",
            "--gate-threshold", "0.45",
            "--gate-min-side-score", "0.00",
            "--gate-min-other-side", "0.00",
            "--min-accept-black-score", "0.00",
            "--human-seed-dir", "data/raw/human_games",
            "--human-seed-side", "auto",
            "--seed", "20260221",
            "--human-eval",
        ],
    },
}


def _upsert_arg(args, flag, value):
    if flag in args:
        idx = args.index(flag)
        if idx + 1 >= len(args):
            args.append(str(value))
        else:
            args[idx + 1] = str(value)
        return args
    return args + [flag, str(value)]


def _cmdline(parts):
    return subprocess.list2cmdline(parts)


def _print_presets():
    print("Available presets:\n")
    for name in ("smoke", "daily", "overnight"):
        p = PRESETS[name]
        print(f"- {name}: {p['description']}")
        print(f"  expected runtime: {p['expected_runtime']}")
        print("  artifacts:")
        for a in p["artifacts"]:
            print(f"    - {a}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run iterate.py using reproducible named presets")
    parser.add_argument("--preset", type=str, choices=sorted(PRESETS.keys()),
                        help="Preset name to execute")
    parser.add_argument("--show-presets", action="store_true",
                        help="List available presets and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override preset --iterations")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override preset --seed")
    args, extra = parser.parse_known_args()

    if args.show_presets or not args.preset:
        _print_presets()
        if not args.preset:
            return

    preset = PRESETS[args.preset]
    iter_args = list(preset["args"])
    if args.iterations is not None:
        iter_args = _upsert_arg(iter_args, "--iterations", args.iterations)
    if args.seed is not None:
        iter_args = _upsert_arg(iter_args, "--seed", args.seed)

    cmd = [sys.executable, ITERATE_PATH] + iter_args + extra
    print(f"Preset: {args.preset}")
    print(f"Description: {preset['description']}")
    print(f"Expected runtime: {preset['expected_runtime']}")
    print("\nCommand:")
    print(_cmdline(cmd))

    if args.dry_run:
        return

    result = subprocess.run(cmd, cwd=SRC_DIR)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
