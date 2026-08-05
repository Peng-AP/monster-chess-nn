"""Strict two-color confirmation for preserved candidate checkpoints.

The same openings and seed are first played as bar vs bar, which measures
the color/opening baseline.  A candidate passes only if its calibrated score
delta is positive as both White and Black.  Nothing is promoted or overwritten.

Example:
    py -3 -u tools/confirm_candidates.py \
      --candidate trial13_e5=models/tuning/.../best_value_net.pt \
      --games 80 --sims 800 --seed 20560820
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from config import DEFAULT_GAME_WORKERS  # noqa: E402
from match import run_match  # noqa: E402


BAR = ROOT / "models/candidates/v19_B/best_value_net.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_candidate(spec: str) -> tuple[str, Path]:
    name, separator, raw_path = spec.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise ValueError("candidate must be NAME=CHECKPOINT")
    path = Path(raw_path.strip())
    if not path.is_absolute():
        path = ROOT / path
    return name.strip(), path


def calibrated_result(candidate: dict, calibration: dict) -> dict:
    deltas = {
        "black": (candidate["a_as_black"]["score"]
                  - calibration["a_as_black"]["score"]),
        "white": (candidate["a_as_white"]["score"]
                  - calibration["a_as_white"]["score"]),
        "aggregate": candidate["a_score"] - calibration["a_score"],
    }
    # Strict by design: equality is not a demonstrated improvement.
    return {
        "deltas": {key: float(value) for key, value in deltas.items()},
        "passes_both_colors": deltas["black"] > 0 and deltas["white"] > 0,
        "minimum_color_delta": float(min(deltas["black"], deltas["white"])),
    }


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar", default=str(BAR.relative_to(ROOT)),
                        help="Checkpoint used for self-calibration and comparison")
    parser.add_argument("--candidate", action="append", required=True,
                        help="NAME=CHECKPOINT (repeatable)")
    parser.add_argument("--games", type=int, default=80)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20560820)
    parser.add_argument("--workers", type=int, default=DEFAULT_GAME_WORKERS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.sims <= 0 or args.workers <= 0:
        parser.error("games must be positive and even; sims/workers must be > 0")
    bar = Path(args.bar)
    if not bar.is_absolute():
        bar = ROOT / bar
    if not bar.is_file():
        raise FileNotFoundError(bar)
    bar_name = bar.parent.name

    candidates = [parse_candidate(spec) for spec in args.candidate]
    if len({name for name, _path in candidates}) != len(candidates):
        parser.error("candidate names must be unique")
    for _name, path in candidates:
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"[confirm] calibrating {bar_name}: {args.games} games @ {args.sims}",
          flush=True)
    calibration = run_match(
        str(bar), str(bar), args.games, args.sims, args.seed,
        workers=args.workers, engine="native")
    results = []
    for name, path in candidates:
        print(f"[confirm] {name} vs {bar_name}", flush=True)
        match = run_match(
            str(path), str(bar), args.games, args.sims, args.seed,
            workers=args.workers, engine="native")
        result = calibrated_result(match, calibration)
        result.update({
            "name": name,
            "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": file_sha256(path),
            "match": match,
        })
        results.append(result)
        delta = result["deltas"]
        print(f"[confirm] {name}: dB={delta['black']:+.3f} "
              f"dW={delta['white']:+.3f} dAll={delta['aggregate']:+.3f} "
              f"pass={result['passes_both_colors']}", flush=True)

    payload = {
        "experiment": "strict_two_color_candidate_confirmation",
        "binding": False,
        "bar": str(bar.relative_to(ROOT)).replace("\\", "/"),
        "bar_sha256": file_sha256(bar),
        "games": args.games,
        "sims": args.sims,
        "seed": args.seed,
        "calibration": calibration,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = (ROOT / args.out if args.out else ROOT / "benchmarks" /
           f"candidate_confirmation_{time.strftime('%Y%m%d_%H%M%S')}.json")
    save_json(out, payload)
    print(f"[confirm] saved {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
