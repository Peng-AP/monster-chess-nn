"""Summarize the actual replay schedule and its generating checkpoint lineage.

CPU-only, small arrays only: does not load the multi-GB position/policy tensors.
Policy-only weight mass is descriptive, not a literal gradient-contribution
percentage (training normalizes weights inside each batch).
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from match_evidence import atomic_json, file_hash


def read_json(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def absolute(path):
    return Path(path).resolve() if Path(path).is_absolute() else (ROOT / path).resolve()


def weight_summary(frequency, policy, value):
    only = (value == 0) & (policy > 0)
    total = float((frequency * policy).sum(dtype=np.float64))
    only_mass = float((frequency[only] * policy[only]).sum(dtype=np.float64))
    return {"scheduled_rows": int(frequency.sum()),
            "distinct_array_rows": int(np.count_nonzero(frequency)),
            "policy_only_scheduled_rows": int(frequency[only].sum()),
            "policy_weight_mass": total, "policy_only_weight_mass": only_mass,
            "policy_only_weight_fraction": only_mass / total if total else None,
            "value_weight_mass": float((frequency * value).sum(dtype=np.float64))}


def census(data_dir, registry_path):
    directory = absolute(data_dir)
    manifest_path = directory / "replay_manifest.json"
    manifest = read_json(manifest_path)
    registry = read_json(registry_path)
    accepted = {absolute(r["path"]): r for r in registry["entries"]}
    policy = np.load(directory / "policy_weights.npy", mmap_mode="r")
    value = np.load(directory / "value_weights.npy", mmap_mode="r")
    rows = int(manifest["rows"])
    if len(policy) != rows or len(value) != rows:
        raise ValueError("replay weight-array length differs from manifest")
    if not (np.all(np.isfinite(policy)) and np.all(np.isfinite(value))
            and np.all(policy >= 0) and np.all(value >= 0)):
        raise ValueError("replay weights must be finite and nonnegative")
    with np.load(directory / "splits.npz") as file:
        splits = {name: np.asarray(file[name]) for name in ("train", "val", "test")}
    for name, indices in splits.items():
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"invalid {name} split index array")
        if len(indices) and (indices.min() < 0 or indices.max() >= rows):
            raise ValueError(f"out-of-range {name} split index")
        if len(indices) != manifest["split_rows"][name]:
            raise ValueError(f"{name} split count differs from manifest")
    frequencies = {name: np.bincount(indices, minlength=rows) for name, indices in splits.items()}
    out = {"data_dir": str(directory), "rows": rows,
           "interpretation": "actual sampled row schedule and nominal loss weights; no position tensor scan",
           "input_hashes": {str(p): file_hash(p) for p in
                            (manifest_path, Path(registry_path).resolve(), directory / "splits.npz",
                             directory / "policy_weights.npy", directory / "value_weights.npy")},
           "splits": {name: weight_summary(f, policy, value) for name, f in frequencies.items()},
           "training_balance": manifest.get("training_balance"), "sources": []}
    offset = 0
    for source in manifest["sources"]:
        end = offset + int(source["rows"])
        source_dir = absolute(source["path"])
        record = accepted.get(source_dir)
        if record is None:
            raise ValueError(f"source is not in accepted-data registry: {source_dir}")
        if int(record["rows"]) != end - offset:
            raise ValueError("source row count differs from accepted registry")
        out["sources"].append({"name": source["name"], "path": str(source_dir),
            "generation": record["generation"], "rows": end - offset,
            "generator": record.get("incumbent"), "generator_sha256": record.get("incumbent_sha256"),
            "policy_only_multiplier": source.get("policy_only_multiplier", 1.0),
            "split_stats": {name: weight_summary(f[offset:end], policy[offset:end], value[offset:end])
                            for name, f in frequencies.items()}})
        offset = end
    if offset != rows:
        raise ValueError("source row sum differs from replay manifest")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--registry", default=str(ROOT / "iterations" / "accepted_data.json"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    destination = absolute(args.output)
    if destination.exists():
        raise FileExistsError("use a fresh census output path")
    out = census(args.data_dir, args.registry)
    atomic_json(destination, out)
    for source in out["sources"]:
        stats = source["split_stats"]["train"]
        fraction = stats["policy_only_weight_fraction"]
        print(f"{source['name']}: generator={source['generator']}; "
              f"train={stats['scheduled_rows']} distinct_rows={stats['distinct_array_rows']} "
              f"policy_only_weight={format(fraction, '.1%') if fraction is not None else 'disabled'}")


if __name__ == "__main__":
    main()
