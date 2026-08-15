"""Play-test preserved epochs with successive halving and nominate one for gates.

Sample sizes raised 2026-08-07. The old defaults (8-game probes, 20-game
finals) cannot rank checkpoints: at 20 games the standard error is 0.112, and
on that day five finalists spanning 0.187 were ranked confidently on samples
that could not resolve them -- while an 800-game match showed the same weights
scoring 1.6 SE away from their screen result. Probes now play 40 games at the
gate's own sim count and finals play 200 (SE 0.035), which is the resolution
needed to separate checkpoints that differ by the ~40 Elo measured between
epoch 4 and epoch 10 of a from-scratch run.

Every unique ``selected_epoch_*.pt`` model receives a cheap paired-color probe.
The strongest probe results, the offline-selected epoch, and the Black-best
epoch advance to the normal calibrated screen.  The screen only nominates a
checkpoint: the binding and high-fidelity gates remain decisive.
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


def weights_sha256(path: Path) -> str:
    """Hash tensor contents, independent of torch's zip serialization."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def discover_checkpoints(model_dir: Path) -> list[dict]:
    paths = sorted(model_dir.glob("selected_epoch_*.pt"))
    best = model_dir / "best_value_net.pt"
    best_weights_digest = weights_sha256(best) if best.is_file() else None
    if best.is_file():
        paths.append(best)
    unique = []
    seen = set()
    for path in paths:
        weights_digest = weights_sha256(path)
        if weights_digest in seen:
            continue
        seen.add(weights_digest)
        unique.append({
            "name": path.stem,
            "path": path,
            "sha256": sha256(path),
            "weights_sha256": weights_digest,
            "offline_selected": weights_digest == best_weights_digest,
        })
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


def choose_finalists(results: list[dict], count: int) -> list[dict]:
    """Keep arena leaders plus safeguards against proxy and White bias."""
    ranked = sorted(results, key=rank_key, reverse=True)
    selected_names = {result["name"] for result in ranked[:count]}
    selected_names.add(max(results, key=lambda result: result["deltas"]["black"])["name"])
    selected_names.update(
        result["name"] for result in results if result["offline_selected"])
    return [result for result in ranked if result["name"] in selected_names]


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
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--probe-games", type=int, default=40)
    parser.add_argument("--probe-sims", type=int, default=400)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=DEFAULT_GAME_WORKERS)
    parser.add_argument("--engine", choices=("python", "native"), default="native")
    parser.add_argument("--stall-timeout", type=float, default=600.0)
    parser.add_argument("--book", default=None,
                        help="paired opening book (tools/make_book.py). Every "
                             "checkpoint then faces the identical openings, "
                             "which is what makes their scores directly "
                             "comparable instead of independent samples.")
    args = parser.parse_args()
    if (args.games < 2 or args.games % 2 or args.sims <= 0
            or args.probe_games < 2 or args.probe_games % 2
            or args.probe_sims <= 0 or args.finalists <= 0
            or args.workers <= 0):
        parser.error("game counts must be positive and even; sims, finalists, "
                     "and workers must be > 0")

    model_dir = Path(args.model_dir).resolve()
    incumbent = Path(args.incumbent).resolve()
    output_model = Path(args.output_model).resolve()
    report_path = Path(args.report_path).resolve()
    checkpoints = discover_checkpoints(model_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no preserved checkpoints in {model_dir}")
    if not incumbent.is_file():
        raise FileNotFoundError(incumbent)

    # Probes and finals draw DISJOINT blocks. The two stages exist so that the
    # high-power confirmation is independent of the selection that produced the
    # finalists; replaying the probe openings would correlate the confirmation
    # with its own selection bias and reinstate exactly what the screen is for.
    # Both calibrations use their stage's block, since the deltas are measured
    # against them and must come from the same positions.
    probe_offset, final_offset = 0, args.probe_games // 2
    if args.book:
        from match import load_book
        needed = final_offset + args.games // 2
        available = len(load_book(args.book)[0])
        if needed > available:
            raise SystemExit(
                f"screen needs {needed} book entries "
                f"({args.probe_games // 2} probe + {args.games // 2} final) "
                f"but {args.book} has {available}")

    print(f"[checkpoint-screen] probe calibration: {args.probe_games} games "
          f"@ {args.probe_sims}", flush=True)
    probe_calibration = run_match(
        str(incumbent), str(incumbent), args.probe_games, args.probe_sims,
        args.seed,
        workers=args.workers, engine=args.engine,
        stall_timeout=args.stall_timeout,
        book=args.book, book_offset=probe_offset)
    probe_results = []
    for checkpoint in checkpoints:
        print(f"[checkpoint-screen] probe {checkpoint['name']}", flush=True)
        match = run_match(
            str(checkpoint["path"]), str(incumbent), args.probe_games,
            args.probe_sims,
            args.seed, workers=args.workers, engine=args.engine,
            stall_timeout=args.stall_timeout,
            book=args.book, book_offset=probe_offset)
        result = calibrated_result(match, probe_calibration)
        result.update({
            "name": checkpoint["name"],
            "checkpoint": checkpoint["path"].relative_to(ROOT).as_posix(),
            "checkpoint_sha256": checkpoint["sha256"],
            "weights_sha256": checkpoint["weights_sha256"],
            "offline_selected": checkpoint["offline_selected"],
            "match": match,
        })
        probe_results.append(result)
        delta = result["deltas"]
        print(f"[checkpoint-screen] probe {checkpoint['name']}: "
              f"dB={delta['black']:+.3f} dW={delta['white']:+.3f} "
              f"dAll={delta['aggregate']:+.3f}", flush=True)

    finalists = choose_finalists(probe_results, args.finalists)
    finalist_names = {result["name"] for result in finalists}
    print(f"[checkpoint-screen] finalists: "
          f"{', '.join(result['name'] for result in finalists)}", flush=True)
    screen_seed = args.seed + 1
    print(f"[checkpoint-screen] full calibration: {args.games} games "
          f"@ {args.sims}", flush=True)
    calibration = run_match(
        str(incumbent), str(incumbent), args.games, args.sims, screen_seed,
        workers=args.workers, engine=args.engine,
        stall_timeout=args.stall_timeout,
        book=args.book, book_offset=final_offset)
    results = []
    checkpoint_by_name = {checkpoint["name"]: checkpoint
                          for checkpoint in checkpoints}
    for probe in finalists:
        checkpoint = checkpoint_by_name[probe["name"]]
        print(f"[checkpoint-screen] full {checkpoint['name']}", flush=True)
        match = run_match(
            str(checkpoint["path"]), str(incumbent), args.games, args.sims,
            screen_seed, workers=args.workers, engine=args.engine,
            stall_timeout=args.stall_timeout,
            book=args.book, book_offset=final_offset)
        result = calibrated_result(match, calibration)
        result.update({
            "name": checkpoint["name"],
            "checkpoint": checkpoint["path"].relative_to(ROOT).as_posix(),
            "checkpoint_sha256": checkpoint["sha256"],
            "weights_sha256": checkpoint["weights_sha256"],
            "offline_selected": checkpoint["offline_selected"],
            "match": match,
        })
        results.append(result)
        delta = result["deltas"]
        print(f"[checkpoint-screen] full {checkpoint['name']}: "
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
        "seed": screen_seed,
        "probe": {
            "games": args.probe_games,
            "sims": args.probe_sims,
            "seed": args.seed,
            "calibration": probe_calibration,
            "results": probe_results,
            "finalist_names": sorted(finalist_names),
        },
        "calibration": calibration,
        "results": results,
        "selected": {
            "name": selected["name"],
            "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "weights_sha256": selected["weights_sha256"],
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
