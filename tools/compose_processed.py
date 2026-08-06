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
TURN_LAYER = 12


def _allocate_counts(counts, total, alpha):
    """Allocate a fixed sample budget with inverse-frequency smoothing."""
    counts = np.asarray(counts, dtype=np.int64)
    active = counts > 0
    weights = np.zeros(len(counts), dtype=np.float64)
    weights[active] = counts[active].astype(np.float64) ** (1.0 - alpha)
    exact = weights / weights.sum() * int(total)
    allocated = np.floor(exact).astype(np.int64)
    remainder = int(total - allocated.sum())
    if remainder:
        order = np.argsort(-(exact - allocated), kind="stable")
        allocated[order[:remainder]] += 1
    return allocated


def balance_training_indices(output_dir, train_indices, alpha=0.5, seed=42,
                             chunk_rows=2048):
    """Smooth side/outcome/phase strata without changing split membership.

    Phase boundaries are corpus quantiles of remaining piece count, rather
    than hand-authored tactical rules. True capture outcome is used when
    available; otherwise +/-0.5 move-limit labels are treated as draws.
    """
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if alpha <= 0 or len(train_indices) == 0:
        return train_indices, {"enabled": False, "alpha": float(alpha)}
    if not 0 <= alpha <= 1:
        raise ValueError("balance alpha must be in [0, 1]")

    positions = np.load(os.path.join(output_dir, "positions.npy"), mmap_mode="r")
    if positions.shape[-1] <= TURN_LAYER:
        raise ValueError(
            f"balanced replay requires turn layer {TURN_LAYER}, got "
            f"{positions.shape[-1]} channels")
    result_name = ("capture_results.npy"
                   if os.path.exists(os.path.join(output_dir,
                                                  "capture_results.npy"))
                   else "game_results.npy")
    results = np.load(os.path.join(output_dir, result_name), mmap_mode="r")
    side = np.empty(len(train_indices), dtype=np.int8)
    material = np.empty(len(train_indices), dtype=np.float32)
    outcome = np.empty(len(train_indices), dtype=np.int8)
    for start in range(0, len(train_indices), chunk_rows):
        end = min(len(train_indices), start + chunk_rows)
        idx = train_indices[start:end]
        block = np.asarray(positions[idx])
        white = block[:, 0, 0, TURN_LAYER] > 0
        side[start:end] = white.astype(np.int8)
        material[start:end] = block[:, :, :, :12].sum(axis=(1, 2, 3))
        relative = np.asarray(results[idx]) * np.where(white, 1.0, -1.0)
        if result_name == "capture_results.npy":
            outcome[start:end] = np.where(relative > 0, 2,
                                          np.where(relative < 0, 0, 1))
        else:
            outcome[start:end] = np.where(relative > 0.75, 2,
                                          np.where(relative < -0.75, 0, 1))

    lower, upper = np.quantile(material, (1 / 3, 2 / 3))
    phase = np.digitize(material, (lower, upper), right=True).astype(np.int8)
    stratum = side * 9 + outcome * 3 + phase
    counts = np.bincount(stratum, minlength=18)
    target_counts = _allocate_counts(counts, len(train_indices), alpha)
    rng = np.random.default_rng(seed)
    selected = []
    for key, target in enumerate(target_counts):
        if target <= 0:
            continue
        members = train_indices[stratum == key]
        selected.append(rng.choice(
            members, size=int(target), replace=target > len(members)))
    balanced = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(balanced)
    labels = []
    for key in range(18):
        labels.append({
            "side": "white" if key // 9 else "black",
            "outcome": ("loss", "draw", "win")[(key % 9) // 3],
            "phase": ("late", "middle", "early")[key % 3],
            "source_rows": int(counts[key]),
            "sampled_rows": int(target_counts[key]),
        })
    return balanced, {
        "enabled": True,
        "alpha": float(alpha),
        "seed": int(seed),
        "result_source": result_name,
        "material_quantiles": [float(lower), float(upper)],
        "source_rows": int(len(train_indices)),
        "sampled_rows": int(len(balanced)),
        "strata": labels,
    }


def parse_source(spec):
    name, separator, path = spec.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"source must be NAME=PATH, got {spec!r}")
    return name, os.path.abspath(path)


def parse_policy_only_multiplier(spec):
    name, separator, value = spec.partition("=")
    if not separator or not name or not value:
        raise ValueError(
            f"policy-only multiplier must be NAME=FLOAT, got {spec!r}")
    multiplier = float(value)
    if multiplier <= 0:
        raise ValueError("policy-only multiplier must be positive")
    return name, multiplier


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
        value_weights = None
        multiplier = float(source.get("policy_only_multiplier", 1.0))
        if filename == "policy_weights.npy" and multiplier != 1.0:
            value_weights = np.load(
                os.path.join(source["path"], "value_weights.npy"),
                mmap_mode="r")
        for start in range(0, len(array), chunk_rows):
            end = min(len(array), start + chunk_rows)
            block = np.asarray(array[start:end])
            if value_weights is not None:
                policy_only = ((np.asarray(value_weights[start:end]) == 0)
                               & (block > 0))
                if policy_only.any():
                    block = block.copy()
                    block[policy_only] *= multiplier
            target[offset + start:offset + end] = block
        offset += len(array)
        print(f"  {filename}: copied {source['name']} ({len(array):,} rows)",
              flush=True)
    target.flush()
    del target


def compose(sources, arrays, output_dir, chunk_rows=2048, balance_alpha=0.0,
            balance_seed=42):
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError(f"output directory already exists: {output_dir}")
    staging = os.path.join(
        os.path.dirname(output_dir),
        f".{os.path.basename(output_dir)}.tmp-{os.getpid()}")
    if os.path.exists(staging):
        raise FileExistsError(f"composition staging directory exists: {staging}")
    os.makedirs(staging)
    for filename in arrays:
        _copy_array(filename, sources, staging, chunk_rows)

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

    raw_train = (np.concatenate(combined_splits["train"])
                 if combined_splits["train"]
                 else np.empty(0, dtype=np.int64))
    train, balance = balance_training_indices(
        staging, raw_train, balance_alpha, balance_seed, chunk_rows)
    final_splits = {
        "train": train,
        "val": (np.concatenate(combined_splits["val"])
                if combined_splits["val"] else np.empty(0, dtype=np.int64)),
        "test": (np.concatenate(combined_splits["test"])
                 if combined_splits["test"] else np.empty(0, dtype=np.int64)),
    }
    np.savez(
        os.path.join(staging, "splits.npz"),
        **final_splits,
    )
    with open(os.path.join(staging, "split_game_ids.json"), "w",
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
            split: int(len(indices)) for split, indices in final_splits.items()
        },
        "training_balance": balance,
    }
    with open(os.path.join(staging, "replay_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(staging, output_dir)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", action="append", required=True,
                    help="processed source as NAME=PATH; repeatable")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--chunk-rows", type=int, default=2048)
    ap.add_argument("--balance-alpha", type=float, default=0.0,
                    help="0 disables; 1 fully equalizes side/outcome/phase strata")
    ap.add_argument("--balance-seed", type=int, default=42)
    ap.add_argument(
        "--policy-only-multiplier", action="append", default=[],
        help="multiply enabled policy weights whose value weight is zero in "
             "one source, as NAME=FLOAT; repeatable")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.chunk_rows <= 0:
        ap.error("--chunk-rows must be positive")
    if not 0 <= args.balance_alpha <= 1:
        ap.error("--balance-alpha must be in [0, 1]")
    if os.path.exists(args.output_dir):
        ap.error(f"output directory already exists: {args.output_dir}")
    try:
        sources, arrays = inspect_sources(args.source)
        multipliers = dict(parse_policy_only_multiplier(spec)
                           for spec in args.policy_only_multiplier)
        unknown = set(multipliers) - {source["name"] for source in sources}
        if unknown:
            raise ValueError("policy-only multiplier names unknown source(s): "
                             + ", ".join(sorted(unknown)))
        for source in sources:
            source["policy_only_multiplier"] = float(
                multipliers.get(source["name"], 1.0))
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
    manifest = compose(sources, arrays, args.output_dir, args.chunk_rows,
                       args.balance_alpha, args.balance_seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
