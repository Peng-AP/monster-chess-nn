"""Process parent-linked extra games into a value-only increment (gen51 deep-value arm).

Stage 3 of docs/plans/GEN51_STRENGTH_PLAN.md. The games (e.g. disagreement
continuations) are processed by the unchanged tensor processor, but each game
takes **its parent's recorded split** from the parent increment's
`split_game_ids.json`, so no position can sit in training here and in
validation/test there. A missing parent is an error.

The published increment then differs from the processor output in exactly
three arrays, recorded in `derivation.json`:

* `game_results.npy` <- `capture_results.npy` (strict captures-only outcome,
  White perspective, repetition/turn-cap draws 0; no distance discount);
* `value_weights.npy` <- value weight x `--value-weight` (plan: 4);
* `policy_weights.npy` <- 0 (value-only source: the arm tests value targets).
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_processor
from match_evidence import atomic_json, file_hash


def parent_splitter(parent_splits):
    lookup = {gid: name for name in ("train", "val", "test") for gid in parent_splits[name]}

    def split(games, seed):
        out = {"train": [], "val": [], "test": []}
        for game in games:
            parent = game.get("split_parent")
            if parent not in lookup:
                raise ValueError(f"{game['game_id']}: parent {parent!r} not in the parent increment")
            out[lookup[parent]].append(game)
        return out
    return split


def build(raw_dir, parent_dir, out_dir, seed, value_weight):
    raw_dir, parent_dir, out_dir = Path(raw_dir), Path(parent_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; increments are immutable")
    staging = out_dir.with_name(out_dir.name + ".processing")
    if staging.exists():
        shutil.rmtree(staging)
    parent_splits = json.loads((parent_dir / "split_game_ids.json").read_text())
    data_processor._split_games_by_result = parent_splitter(parent_splits)
    data_processor.process_raw_data(raw_dir=str(raw_dir), output_dir=str(staging), seed=seed,
                                    input_channels=15, value_floor=1.0, value_horizon=60,
                                    value_discount_mode="near_mate", min_nonhuman_plies=0,
                                    max_generation_age=0)
    original = {n: file_hash(staging / n) for n in ("game_results.npy", "value_weights.npy", "policy_weights.npy")}
    capture = np.load(staging / "capture_results.npy")
    values = np.load(staging / "value_weights.npy")
    np.save(staging / "game_results.npy", capture.astype(np.float32))
    np.save(staging / "value_weights.npy", (values * value_weight).astype(np.float32))
    np.save(staging / "policy_weights.npy", np.zeros_like(np.load(staging / "policy_weights.npy")))
    child_splits = json.loads((staging / "split_game_ids.json").read_text())
    derivation = dict(
        tool="tools/process_linked_extra.py", raw_dir=str(raw_dir), parent_increment=str(parent_dir),
        parent_split_game_ids_sha256=file_hash(parent_dir / "split_game_ids.json"),
        seed=seed, value_floor=1.0, value_weight=value_weight,
        replaced={"game_results.npy": "capture_results.npy (strict captures-only)",
                  "value_weights.npy": f"processor value weights x {value_weight}",
                  "policy_weights.npy": "zeros (value-only source)"},
        processor_original_sha256=original,
        published_sha256={n: file_hash(staging / n) for n in sorted(p.name for p in staging.glob("*.np*"))},
        rows=int(len(capture)), value_rows=int((values > 0).sum()),
        games={k: len(child_splits[k]) for k in ("train", "val", "test")})
    atomic_json(staging / "derivation.json", derivation)
    staging.rename(out_dir)
    print(f"LINKED EXTRA INCREMENT: {derivation['rows']} rows, games {derivation['games']}", flush=True)
    return derivation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--parent-increment", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--value-weight", type=float, required=True)
    args = ap.parse_args()
    if args.value_weight <= 0:
        ap.error("--value-weight must be positive")
    build(args.raw_dir, args.parent_increment, args.output_dir, args.seed, args.value_weight)


if __name__ == "__main__":
    main()
