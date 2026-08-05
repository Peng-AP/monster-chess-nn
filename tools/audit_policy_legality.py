"""Audit enabled policy targets against complete Monster Chess legality."""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import RAW_DATA_DIR  # noqa: E402
from data_processor import (  # noqa: E402
    _legal_policy_mask_packed,
    load_all_games,
    policy_dict_to_target,
    policy_weight_for_record,
)


def audit(raw_dir, include_human=True):
    games = load_all_games(raw_dir, include_human=include_human)
    counts = collections.Counter()
    by_source = collections.Counter()
    samples = []
    started = time.time()
    for game in games:
        for rec in game["records"]:
            counts["records"] += 1
            weight = policy_weight_for_record(rec)
            if weight <= 0:
                counts["policy_masked"] += 1
                continue
            counts["policy_enabled"] += 1
            is_white = rec["current_player"] == "white"
            target = policy_dict_to_target(rec["policy"], is_white)
            target_indices = frozenset(np.flatnonzero(target > 0))
            _packed, _mirrored, legal_indices, _mirrored_indices = (
                _legal_policy_mask_packed(rec))
            illegal = [int(index) for index in sorted(
                target_indices - legal_indices)]
            if not illegal:
                counts["enabled_legal"] += 1
                continue
            counts["enabled_illegal"] += 1
            source = rec.get("source") or game.get("source_kind") or "unknown"
            by_source[str(source)] += 1
            if len(samples) < 20:
                samples.append({
                    "game_id": game["game_id"],
                    "source": source,
                    "current_player": rec.get("current_player"),
                    "half": rec.get("half", 0),
                    "fen": rec.get("fen"),
                    "policy": rec.get("policy"),
                    "illegal_indices": illegal[:8],
                })
    return {
        "raw_dir": os.path.relpath(os.path.abspath(raw_dir), ROOT),
        "counts": dict(counts),
        "enabled_illegal_by_source": dict(by_source.most_common()),
        "samples": samples,
        "elapsed_sec": round(time.time() - started, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", nargs="?", default=RAW_DATA_DIR)
    parser.add_argument("--exclude-human", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = audit(args.raw_dir, include_human=not args.exclude_human)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
