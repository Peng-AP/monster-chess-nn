"""Play-test every preserved training checkpoint and nominate one for gates.

Validation metrics decide which epochs are worth preserving; arena play decides
which preserved epoch advances.  Every unique ``selected_epoch_*.pt`` model is
tested on the same openings against the incumbent, after one incumbent
self-calibration match.  The screen only nominates a checkpoint: the normal
binding and high-fidelity gates remain decisive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from config import DEFAULT_GAME_WORKERS  # noqa: E402
from match import run_match  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover_checkpoints(model_dir: Path) -> list[dict]:
    paths = sorted(model_dir.glob("selected_epoch_*.pt"))
    best = model_dir / "best_value_net.pt"
    if best.is_file():
        paths.append(best)
    unique = []
    seen = set()
    for path in paths:
        digest = sha256(path)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append({"name": path.stem, "path": path, "sha256": digest})
    return unique


def calibrated_result(candidate: dict, calibration: dict) -> dict:
    deltas = {
        "white": (candidate["a_as_white"]["score"]
                  - calibration["a_as_white"]["score"]),
        "black": (candidate["a_as_black"]["score"]
                  - calibration["a_as_black"]["score"]),
        "aggregate": candidate["a_score"] - calibration["a_score"],
    }
    return {
        "deltas": {name: float(value) for name, value in deltas.items()},
        "minimum_color_delta": float(min(deltas["white"], deltas["black"])),
        "passes_both_colors": deltas["white"] > 0 and deltas["black"] > 0,
    }


def rank_key(result: dict) -> tuple[float, float, float]:
    delta = result["deltas"]
    return (result["minimum_color_delta"], delta["aggregate"], delta["black"])


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=DEFAULT_GAME_WORKERS)
    parser.add_argument("--engine", choices=("python", "native"), default="native")
    parser.add_argument("--stall-timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.sims <= 0 or args.workers <= 0:
        parser.error("games must be positive and even; sims/workers must be > 0")

    model_dir = Path(args.model_dir).resolve()
    incumbent = Path(args.incumbent).resolve()
    output_model = Path(args.output_model).resolve()
    report_path = Path(args.report_path).resolve()
    checkpoints = discover_checkpoints(model_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no preserved checkpoints in {model_dir}")
    if not incumbent.is_file():
        raise FileNotFoundError(incumbent)

    print(f"[checkpoint-screen] calibrating incumbent: {args.games} games "
          f"@ {args.sims}", flush=True)
    calibration = run_match(
        str(incumbent), str(incumbent), args.games, args.sims, args.seed,
        workers=args.workers, engine=args.engine,
        stall_timeout=args.stall_timeout)
    results = []
    for checkpoint in checkpoints:
        print(f"[checkpoint-screen] {checkpoint['name']} vs incumbent", flush=True)
        match = run_match(
            str(checkpoint["path"]), str(incumbent), args.games, args.sims,
            args.seed, workers=args.workers, engine=args.engine,
            stall_timeout=args.stall_timeout)
        result = calibrated_result(match, calibration)
        result.update({
            "name": checkpoint["name"],
            "checkpoint": checkpoint["path"].relative_to(ROOT).as_posix(),
            "checkpoint_sha256": checkpoint["sha256"],
            "match": match,
        })
        results.append(result)
        delta = result["deltas"]
        print(f"[checkpoint-screen] {checkpoint['name']}: "
              f"dB={delta['black']:+.3f} dW={delta['white']:+.3f} "
              f"dAll={delta['aggregate']:+.3f}", flush=True)

    selected = max(results, key=rank_key)
    selected_path = ROOT / selected["checkpoint"]
    atomic_copy(selected_path, output_model)
    payload = {
        "experiment": "bootstrap_checkpoint_arena_screen",
        "binding": False,
        "incumbent": incumbent.relative_to(ROOT).as_posix(),
        "incumbent_sha256": sha256(incumbent),
        "games": args.games,
        "sims": args.sims,
        "seed": args.seed,
        "calibration": calibration,
        "results": results,
        "selected": {
            "name": selected["name"],
            "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "arena_model": output_model.relative_to(ROOT).as_posix(),
            "arena_model_sha256": sha256(output_model),
            "rank_key": list(rank_key(selected)),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_json(report_path, payload)
    print(f"[checkpoint-screen] selected {selected['name']}", flush=True)
    print(f"[checkpoint-screen] saved {report_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
