"""Read-only census of how many distinct positions a processed corpus holds.

Stage 0 of docs/plans/GEN51_STRENGTH_PLAN.md. Mirror augmentation stores every
position twice (file-flipped), so positions are keyed by the smaller hash of a
row and its mirror: one key per underlying position. Counts are of encoded
network inputs (15-plane layout), which omit repetition history and turn
count, so two game states can share a key. Nothing here changes data.

Reported per corpus, and per game phase and material bucket:
  * value rows and the distinct positions behind them;
  * concentration: share of value rows on the 10/100/1000 most repeated positions;
  * positions whose value rows carry conflicting capture outcomes;
  * policy-only teacher rows that sit on positions already in the value rows.
For a replay composite, also per source generation, and how much of the
newest increment was already present in the older sources.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from match_evidence import atomic_json

TURN, HALF, PIECES = 12, 13, slice(0, 12)
CHUNK = 20000
PHASES = ("black", "white_first", "white_second")
# Monster chess starts with 21 pieces (Black 16, White king + 4 pawns).
MATERIAL = (("21-18", 18), ("17-12", 12), ("11-7", 7), ("6-2", 0))


def keys_and_features(positions):
    n = len(positions)
    keys = np.empty(n, dtype="S12")
    phase = np.empty(n, dtype=np.int8)
    material = np.empty(n, dtype=np.int8)
    for start in range(0, n, CHUNK):
        block = np.ascontiguousarray(positions[start:start + CHUNK])
        mirror = np.ascontiguousarray(block[:, :, ::-1, :])
        for i in range(len(block)):
            a = hashlib.blake2b(block[i].tobytes(), digest_size=12).digest()
            b = hashlib.blake2b(mirror[i].tobytes(), digest_size=12).digest()
            keys[start + i] = min(a, b)
        white = block[:, 0, 0, TURN] > 0
        second = block[:, 0, 0, HALF] > 0.5
        phase[start:start + len(block)] = np.where(~white, 0, np.where(second, 2, 1))
        count = block[..., PIECES].sum(axis=(1, 2, 3))
        material[start:start + len(block)] = np.select(
            [count >= floor for _, floor in MATERIAL], range(len(MATERIAL)), default=len(MATERIAL) - 1)
    return keys, phase, material


def census(keys, value_mask, teacher_mask, results):
    valued = keys[value_mask]
    counts = Counter(valued.tolist())
    n = int(value_mask.sum())
    ordered = [c for _, c in counts.most_common(1000)]
    outcomes = {}
    for k, r in zip(valued.tolist(), results[value_mask].tolist()):
        outcomes.setdefault(k, set()).add(round(float(r), 6))
    conflicted = {k for k, s in outcomes.items() if len(s) > 1}
    teachers = keys[teacher_mask].tolist()
    on_valued = [t for t in teachers if t in counts]
    return {
        "value_rows": n,
        "distinct_positions": len(counts),
        "value_rows_per_distinct_position": n / len(counts) if counts else None,
        "top_share": {str(k): sum(ordered[:k]) / n if n else None for k in (10, 100, 1000)},
        "positions_seen_once": sum(1 for c in counts.values() if c <= 2),
        "conflicting_outcome_positions": len(conflicted),
        "value_rows_on_conflicting_positions": sum(counts[k] for k in conflicted),
        "teacher_rows": len(teachers),
        "teacher_rows_on_value_positions": len(on_valued),
        "mean_value_rows_behind_a_teacher_position": (
            sum(counts[t] for t in on_valued) / len(on_valued) if on_valued else None),
    }


def audit(directory, sources=None):
    directory = Path(directory)
    started = time.time()
    positions = np.load(directory / "positions.npy", mmap_mode="r")
    value_w = np.load(directory / "value_weights.npy")
    policy_w = np.load(directory / "policy_weights.npy")
    results = np.load(directory / "capture_results.npy")
    keys, phase, material = keys_and_features(positions)
    valued, teacher = value_w > 0, (value_w <= 0) & (policy_w > 0)
    out = {"path": str(directory.relative_to(ROOT)).replace("\\", "/"), "rows": len(keys),
           "all": census(keys, valued, teacher, results),
           "by_phase": {name: census(keys, valued & (phase == i), teacher & (phase == i), results)
                        for i, name in enumerate(PHASES)},
           "by_material": {name: census(keys, valued & (material == i), teacher & (material == i), results)
                           for i, (name, _) in enumerate(MATERIAL)}}
    if sources:
        offset, seen, per_source = 0, set(), {}
        for source in sources:
            end = offset + source["rows"]
            mask = np.zeros(len(keys), dtype=bool)
            mask[offset:end] = True
            stats = census(keys, valued & mask, teacher & mask, results)
            own = set(keys[valued & mask].tolist())
            stats["distinct_positions_already_in_older_sources"] = len(own & seen)
            per_source[source["name"]] = stats
            seen |= own
            offset = end
        if offset != len(keys):
            raise ValueError("replay manifest rows do not add up to the composite")
        out["by_source"] = per_source
    out["seconds"] = round(time.time() - started, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--increment", action="append", default=[], help="processed increment directory")
    ap.add_argument("--replay", help="composed replay directory (uses its replay_manifest.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {"tool": "tools/diversity_audit.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "key": "min(blake2b-96(row), blake2b-96(file-mirrored row)) over the 15-plane input",
              "increments": {}, "notes": [
                  "Encoded inputs omit repetition history and turn count; distinct states can share a key.",
                  "positions_seen_once counts keys with at most two value rows (a row and its mirror)."]}
    for path in args.increment:
        print(f"auditing {path}", flush=True)
        report["increments"][Path(path).name] = audit(ROOT / path)
    if args.replay:
        print(f"auditing {args.replay}", flush=True)
        manifest = json.loads((ROOT / args.replay / "replay_manifest.json").read_text())
        report["replay"] = audit(ROOT / args.replay, manifest["sources"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, report)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
