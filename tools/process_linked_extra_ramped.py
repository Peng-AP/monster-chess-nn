"""Rebuild a deep-value increment with the main corpus's game-result labels.

docs/plans/GEN52_RAMP_PLAN.md (owner, 2026-10-01: "retrain with game result
labels and retest"). `tools/process_linked_extra.py` published the deep-value
increments with strict, **undiscounted** capture-only results (value floor
1.0). Every other source trains on the processor's ramped game results
(floor 0.5, horizon 60, near_mate). The value audit found the resulting White
shift tracks the deep-value share.

This tool reprocesses the SAME raw continuation games with the SAME parent
splits and seed, and differs from the published increment in exactly one
thing: `game_results.npy` keeps the processor's ramped game results (floor
0.5, horizon 60) instead of being replaced by undiscounted captures. Value
weight x4 and zero policy weight are unchanged, so the source is still
value-only at the same weight.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import data_processor
from match_evidence import atomic_json, file_hash
from process_linked_extra import parent_splitter

VALUE_FLOOR, VALUE_HORIZON = 0.5, 60   # the main corpus's ramp (process_families.py)


def build(original_dir, out_dir):
    original_dir, out_dir = Path(original_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; increments are immutable")
    source = json.loads((original_dir / "derivation.json").read_text())
    raw_dir, parent_dir = Path(source["raw_dir"]), Path(source["parent_increment"])
    if file_hash(parent_dir / "split_game_ids.json") != source["parent_split_game_ids_sha256"]:
        raise ValueError("parent splits changed since the original increment was built")
    staging = out_dir.with_name(out_dir.name + ".processing")
    if staging.exists():
        shutil.rmtree(staging)
    data_processor._split_games_by_result = parent_splitter(json.loads((parent_dir / "split_game_ids.json").read_text()))
    data_processor.process_raw_data(raw_dir=str(raw_dir), output_dir=str(staging), seed=source["seed"],
                                    input_channels=15, value_floor=VALUE_FLOOR, value_horizon=VALUE_HORIZON,
                                    value_discount_mode="near_mate", min_nonhuman_plies=0, max_generation_age=0)
    values = np.load(staging / "value_weights.npy")
    np.save(staging / "value_weights.npy", (values * source["value_weight"]).astype(np.float32))
    np.save(staging / "policy_weights.npy", np.zeros_like(np.load(staging / "policy_weights.npy")))
    splits = json.loads((staging / "split_game_ids.json").read_text())
    games = {k: len(splits[k]) for k in ("train", "val", "test")}
    rows = int(len(np.load(staging / "game_results.npy", mmap_mode="r")))
    if rows != source["rows"] or games != source["games"]:
        raise ValueError(f"reprocessing changed the corpus: rows {rows} vs {source['rows']}, games {games} vs {source['games']}")
    # Same positions, same order: only the labels may differ.
    if file_hash(staging / "positions.npy") != source["published_sha256"].get("positions.npy"):
        raise ValueError("positions differ from the original increment")
    derivation = dict(
        tool="tools/process_linked_extra_ramped.py", original_increment=str(original_dir),
        original_derivation_sha256=file_hash(original_dir / "derivation.json"), raw_dir=str(raw_dir),
        parent_increment=str(parent_dir), seed=source["seed"], value_floor=VALUE_FLOOR, value_horizon=VALUE_HORIZON,
        value_weight=source["value_weight"],
        replaced={"value_weights.npy": f"processor value weights x {source['value_weight']}",
                  "policy_weights.npy": "zeros (value-only source)"},
        kept={"game_results.npy": "processor ramped game results (floor 0.5, horizon 60), as every other source"},
        published_sha256={n: file_hash(staging / n) for n in sorted(p.name for p in staging.glob("*.np*"))},
        rows=rows, games=games)
    atomic_json(staging / "derivation.json", derivation)
    staging.rename(out_dir)
    print(f"RAMPED EXTRA INCREMENT: {rows} rows, games {games} -> {out_dir}", flush=True)
    return derivation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original-increment", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    build(args.original_increment, args.output_dir)


if __name__ == "__main__":
    main()
