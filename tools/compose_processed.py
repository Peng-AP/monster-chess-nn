"""Compose immutable processed corpora into one replay dataset.

Every source keeps its original train/validation/test membership; indices are
only shifted by the concatenation offset.  This preserves game-level split
integrity while allowing a stable anchor corpus to be combined with recent
self-play and reanalysis without rebuilding or mutating the anchor.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORE_ARRAYS = (
    "positions.npy",
    "mcts_values.npy",
    "game_results.npy",
    "policies.npy",
    "policy_weights.npy",
    "value_weights.npy",
)
OPTIONAL_ARRAYS = (
    "moves_left.npy",
    "moves_left_weights.npy",
    "legal_masks_packed.npy",
    "capture_results.npy",
)


def parse_source(spec):
    name, separator, path = spec.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"source must be NAME=PATH, got {spec!r}")
    return name, os.path.abspath(path)


def inspect_sources(specs):
    sources = []
    names = set()
    for spec in specs:
        name, path = parse_source(spec)
        if name in names:
            raise ValueError(f"duplicate source name: {name}")
        names.add(name)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"processed source not found: {path}")
        for filename in CORE_ARRAYS + ("splits.npz",):
            if not os.path.exists(os.path.join(path, filename)):
                raise FileNotFoundError(f"{name} lacks {filename}: {path}")
        position = np.load(os.path.join(path, "positions.npy"), mmap_mode="r")
        policy = np.load(os.path.join(path, "policies.npy"), mmap_mode="r")
        sources.append({
            "name": name,
            "path": path,
            "rows": int(len(position)),
            "position_shape": tuple(position.shape[1:]),
            "policy_shape": tuple(policy.shape[1:]),
            "arrays": [filename for filename in CORE_ARRAYS + OPTIONAL_ARRAYS
                       if os.path.exists(os.path.join(path, filename))],
        })
    if not sources:
        raise ValueError("at least one --source is required")
    expected_position = sources[0]["position_shape"]
    expected_policy = sources[0]["policy_shape"]
    for source in sources[1:]:
        if source["position_shape"] != expected_position:
            raise ValueError(
                f"position ABI mismatch: {source['name']} "
                f"{source['position_shape']} != {expected_position}")
        if source["policy_shape"] != expected_policy:
            raise ValueError(
                f"policy ABI mismatch: {source['name']} "
                f"{source['policy_shape']} != {expected_policy}")
    # Optional arrays are included only when every source supplies them. This
    # prevents silently inventing auxiliary labels for an older anchor.
    common = [filename for filename in CORE_ARRAYS + OPTIONAL_ARRAYS
              if all(filename in source["arrays"] for source in sources)]
    return sources, common


def _copy_array(filename, sources, output_dir, chunk_rows):
    arrays = [np.load(os.path.join(source["path"], filename), mmap_mode="r")
              for source in sources]
    first = arrays[0]
    total = sum(len(array) for array in arrays)
    shape = (total,) + tuple(first.shape[1:])
    for source, array in zip(sources, arrays):
        if tuple(array.shape[1:]) != tuple(first.shape[1:]):
            raise ValueError(
                f"{filename} shape mismatch in {source['name']}: "
                f"{array.shape[1:]} != {first.shape[1:]}")
        if array.dtype != first.dtype:
            raise ValueError(
                f"{filename} dtype mismatch in {source['name']}: "
                f"{array.dtype} != {first.dtype}")
    target = np.lib.format.open_memmap(
        os.path.join(output_dir, filename), mode="w+", dtype=first.dtype,
        shape=shape)
    offset = 0
    for source, array in zip(sources, arrays):
        for start in range(0, len(array), chunk_rows):
            end = min(len(array), start + chunk_rows)
            target[offset + start:offset + end] = array[start:end]
        offset += len(array)
        print(f"  {filename}: copied {source['name']} ({len(array):,} rows)",
              flush=True)
    target.flush()
    del target


def compose(sources, arrays, output_dir, chunk_rows=2048):
    os.makedirs(output_dir)
    for filename in arrays:
        _copy_array(filename, sources, output_dir, chunk_rows)

    combined_splits = {"train": [], "val": [], "test": []}
    combined_game_ids = {"train": [], "val": [], "test": []}
    offset = 0
    for source in sources:
        with np.load(os.path.join(source["path"], "splits.npz")) as split_file:
            for split in combined_splits:
                if split not in split_file.files:
                    raise ValueError(f"{source['name']} lacks {split} split")
                combined_splits[split].append(
                    np.asarray(split_file[split], dtype=np.int64) + offset)
        game_ids_path = os.path.join(source["path"], "split_game_ids.json")
        if os.path.exists(game_ids_path):
            with open(game_ids_path, encoding="utf-8") as handle:
                ids = json.load(handle)
            for split in combined_game_ids:
                combined_game_ids[split].extend(
                    f"{source['name']}::{value}" for value in ids.get(split, []))
        offset += source["rows"]

    np.savez(
        os.path.join(output_dir, "splits.npz"),
        **{split: np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
           for split, parts in combined_splits.items()},
    )
    with open(os.path.join(output_dir, "split_game_ids.json"), "w",
              encoding="utf-8") as handle:
        json.dump(combined_game_ids, handle, indent=2)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool": "tools/compose_processed.py",
        "rows": sum(source["rows"] for source in sources),
        "arrays": arrays,
        "sources": [{
            **source,
            "path": os.path.relpath(source["path"], ROOT).replace("\\", "/"),
        } for source in sources],
        "split_rows": {
            split: int(sum(len(part) for part in parts))
            for split, parts in combined_splits.items()
        },
    }
    with open(os.path.join(output_dir, "replay_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", action="append", required=True,
                    help="processed source as NAME=PATH; repeatable")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--chunk-rows", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.chunk_rows <= 0:
        ap.error("--chunk-rows must be positive")
    if os.path.exists(args.output_dir):
        ap.error(f"output directory already exists: {args.output_dir}")
    try:
        sources, arrays = inspect_sources(args.source)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    plan = {
        "output_dir": args.output_dir,
        "rows": sum(source["rows"] for source in sources),
        "arrays": arrays,
        "sources": sources,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    manifest = compose(sources, arrays, args.output_dir, args.chunk_rows)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
