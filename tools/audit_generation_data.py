"""Fail-fast audit for one processed bootstrap generation.

Deep-search reanalysis records are deliberately one-row policy teachers.  A
generic short-game retention filter once removed every such row from multiple
production generations without making processing fail.  This tool binds the
reanalysis summary and teacher files to the processed arrays, split manifest,
and value-weight mask so that omission becomes a hard pipeline error.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return payload


def _teacher_files(reanalysis_dir: Path) -> list[Path]:
    return sorted(reanalysis_dir.glob("teacher_*.jsonl"))


def _teacher_record(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            f"policy teacher must contain exactly one record: {path} has "
            f"{len(lines)}")
    try:
        record = json.loads(lines[0])
    except ValueError as exc:
        raise RuntimeError(f"invalid teacher JSON in {path}: {exc}") from exc
    if record.get("source") != "deep_search_reanalysis":
        raise RuntimeError(f"unmarked deep-search teacher: {path}")
    source = record.get("source_record")
    if not isinstance(source, dict) or not source.get("path"):
        raise RuntimeError(f"teacher lacks source_record.path: {path}")
    return record


def _split_membership(split_manifest: dict) -> dict[str, str]:
    membership = {}
    for split in ("train", "val", "test"):
        for game_id in split_manifest.get(split, []):
            normalized = str(game_id).replace("\\", "/")
            if normalized in membership:
                raise RuntimeError(
                    f"game appears in multiple processed splits: {normalized}")
            membership[normalized] = split
    return membership


def audit(raw_dir: Path, reanalysis_dir: Path, processed_dir: Path,
          expected_teachers: int, expected_black_fraction: float,
          value_floor: float, value_horizon: int) -> dict:
    raw_dir = raw_dir.resolve()
    reanalysis_dir = reanalysis_dir.resolve()
    processed_dir = processed_dir.resolve()
    summary = _read_json(reanalysis_dir / "reanalysis_summary.json")
    teachers = _teacher_files(reanalysis_dir)
    if int(summary.get("kept", -1)) != expected_teachers:
        raise RuntimeError(
            f"reanalysis summary kept={summary.get('kept')}, expected "
            f"{expected_teachers}")
    if len(teachers) != expected_teachers:
        raise RuntimeError(
            f"found {len(teachers)} teacher files, expected {expected_teachers}")

    split_manifest = _read_json(processed_dir / "split_game_ids.json")
    retention = split_manifest.get("retention") or {}
    if int(retention.get("min_nonhuman_plies", -1)) != 0:
        raise RuntimeError("processed generation did not disable short-game filtering")
    if abs(float(split_manifest.get("value_floor", -1)) - value_floor) > 1e-9:
        raise RuntimeError(
            f"processed value_floor={split_manifest.get('value_floor')}, "
            f"expected {value_floor}")
    if int(split_manifest.get("value_horizon", -1)) != value_horizon:
        raise RuntimeError(
            f"processed value_horizon={split_manifest.get('value_horizon')}, "
            f"expected {value_horizon}")

    membership = _split_membership(split_manifest)
    black = 0
    teacher_ids = []
    for path in teachers:
        record = _teacher_record(path)
        black += record.get("current_player") == "black"
        teacher_id = path.relative_to(raw_dir).as_posix()
        source_id = str(record["source_record"]["path"]).replace("\\", "/")
        if teacher_id not in membership:
            raise RuntimeError(f"teacher absent from processed splits: {teacher_id}")
        if source_id not in membership:
            raise RuntimeError(
                f"teacher source absent from processed splits: {source_id}")
        if membership[teacher_id] != membership[source_id]:
            raise RuntimeError(
                f"teacher/source split leakage: {teacher_id} is "
                f"{membership[teacher_id]}, {source_id} is {membership[source_id]}")
        teacher_ids.append(teacher_id)

    actual_black_fraction = black / expected_teachers if expected_teachers else 0.0
    # Exact stratification can differ by one row when a requested count rounds.
    fraction_tolerance = 1 / max(1, expected_teachers)
    if abs(actual_black_fraction - expected_black_fraction) > fraction_tolerance:
        raise RuntimeError(
            f"teacher Black fraction {actual_black_fraction:.6f}, expected "
            f"{expected_black_fraction:.6f}")

    positions = np.load(processed_dir / "positions.npy", mmap_mode="r")
    policy_weights = np.load(
        processed_dir / "policy_weights.npy", mmap_mode="r")
    value_weights = np.load(processed_dir / "value_weights.npy", mmap_mode="r")
    rows = len(positions)
    if len(policy_weights) != rows or len(value_weights) != rows:
        raise RuntimeError(
            "processed position/policy-weight/value-weight row counts differ")
    augment = bool(split_manifest.get("augment", True))
    expected_teacher_rows = expected_teachers * (2 if augment else 1)
    policy_only = (np.asarray(value_weights) == 0) & (
        np.asarray(policy_weights) > 0)
    actual_teacher_rows = int(policy_only.sum())
    if actual_teacher_rows != expected_teacher_rows:
        raise RuntimeError(
            f"processed policy-only rows={actual_teacher_rows}, expected "
            f"{expected_teacher_rows}; deep teachers were lost or unrelated "
            "policy-only rows entered the generation")

    return {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_dir": os.path.relpath(raw_dir, ROOT).replace("\\", "/"),
        "reanalysis_dir": os.path.relpath(
            reanalysis_dir, ROOT).replace("\\", "/"),
        "processed_dir": os.path.relpath(
            processed_dir, ROOT).replace("\\", "/"),
        "rows": rows,
        "teacher_files": len(teachers),
        "teacher_rows_after_augmentation": actual_teacher_rows,
        "teacher_black": black,
        "teacher_white": expected_teachers - black,
        "teacher_black_fraction": actual_black_fraction,
        "teacher_split_linkage_checked": len(teacher_ids),
        "value_floor": float(split_manifest["value_floor"]),
        "value_horizon": int(split_manifest["value_horizon"]),
        "min_nonhuman_plies": int(retention["min_nonhuman_plies"]),
        "verdict": "PASS",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--reanalysis-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--expected-teachers", required=True, type=int)
    parser.add_argument("--expected-black-fraction", required=True, type=float)
    parser.add_argument("--value-floor", required=True, type=float)
    parser.add_argument("--value-horizon", required=True, type=int)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.expected_teachers <= 0:
        parser.error("--expected-teachers must be positive")
    if not 0 <= args.expected_black_fraction <= 1:
        parser.error("--expected-black-fraction must be in [0, 1]")
    payload = audit(
        Path(args.raw_dir), Path(args.reanalysis_dir),
        Path(args.processed_dir), args.expected_teachers,
        args.expected_black_fraction, args.value_floor, args.value_horizon)
    out = (Path(args.out) if args.out else
           Path(args.processed_dir) / "generation_audit.json")
    _write_json(out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"generation data audit FAILED: {exc}", file=sys.stderr)
        raise
