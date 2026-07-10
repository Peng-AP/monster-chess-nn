"""Convert raw game JSONL records into training tensors.

Flat conversion + stratified game-level split (REWORK_PLAN Phase 5):
no source quotas, no repetition weights, no per-source value lambdas.
Every kept game contributes all of its positions once (plus the mirror
augmentation). Retention is bounded to two filters: generation age and
a minimum ply count for non-human games.

Source kinds are still *detected* (human_games/ and *_blackfocus/ subdir
names) — human games can be excluded and are exempt from the short-game
filter — but they no longer change weighting or targets.
"""
import json
import os
import re

import numpy as np
from tqdm import tqdm

from config import (
    TENSOR_SHAPE, POLICY_SIZE,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    DATA_RETENTION_MAX_GENERATION_AGE, DATA_RETENTION_MIN_NONHUMAN_PLIES,
    RANDOM_SEED, VALUE_TARGET_GAMMA,
)
# Re-exported for existing importers (evaluation.py, mcts.py, tests).
from encoding import (  # noqa: F401
    PIECE_TO_LAYER,
    fen_to_tensor, mirror_tensor,
    move_to_index, mirror_move_index,
    policy_dict_to_target, mirror_policy,
)


def _generation_from_game_id(game_id):
    """Extract nn_gen number from a relative game path, else None."""
    top = str(game_id).replace("\\", "/").split("/", 1)[0]
    m = re.match(r'^nn_gen(\d+)(?:_.*)?$', top)
    if not m:
        return None
    return int(m.group(1))


def _source_kind_from_rel(rel, is_human):
    if is_human:
        return "human"
    if "_blackfocus/" in rel or rel.endswith("_blackfocus"):
        return "blackfocus"
    return "selfplay"


def load_all_games(raw_dir, include_human=True,
                   max_generation_age=0, min_nonhuman_plies=0,
                   return_summary=False):
    """Load raw games from raw_dir (recursive), applying bounded retention.

    Returns a list of game dicts {game_id, records, source_kind,
    result_bucket, generation}; with return_summary=True also returns a
    retention summary dict.
    """
    paths = []
    for dirpath, _dirnames, filenames in os.walk(raw_dir):
        rel_dir = os.path.relpath(dirpath, raw_dir).replace("\\", "/")
        if (not include_human) and (rel_dir == "human_games" or rel_dir.startswith("human_games/")):
            continue
        for fname in sorted(filenames):
            if fname.endswith(".jsonl"):
                paths.append(os.path.join(dirpath, fname))

    games = []
    input_positions = 0
    for path in tqdm(paths, desc="Loading games"):
        rel = os.path.relpath(path, raw_dir).replace("\\", "/")
        is_human = "human_games/" in rel or rel.startswith("human_games")
        with open(path, "r") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]
        if not records:
            continue
        input_positions += len(records)
        result = records[-1].get("game_result", 0)
        games.append({
            "game_id": rel,
            "records": records,
            "source_kind": _source_kind_from_rel(rel, is_human),
            "result_bucket": _result_bucket(result),
            "generation": _generation_from_game_id(rel),
        })

    input_games = len(games)

    # Retention: bounded to generation age + minimum non-human plies.
    max_generation_age = int(max_generation_age or 0)
    min_nonhuman_plies = int(min_nonhuman_plies or 0)
    generations = [g["generation"] for g in games if g["generation"] is not None]
    latest_generation = max(generations) if generations else None

    kept = []
    dropped = {"age": {"games": 0, "positions": 0},
               "short_nonhuman": {"games": 0, "positions": 0}}
    for game in games:
        positions = len(game["records"])
        generation = game["generation"]
        if (
            latest_generation is not None
            and generation is not None
            and max_generation_age > 0
            and (latest_generation - int(generation)) > max_generation_age
        ):
            dropped["age"]["games"] += 1
            dropped["age"]["positions"] += positions
            continue
        if (
            game["source_kind"] != "human"
            and min_nonhuman_plies > 0
            and positions < min_nonhuman_plies
        ):
            dropped["short_nonhuman"]["games"] += 1
            dropped["short_nonhuman"]["positions"] += positions
            continue
        kept.append(game)

    summary = {
        "max_generation_age": max_generation_age,
        "min_nonhuman_plies": min_nonhuman_plies,
        "latest_generation": latest_generation,
        "input_games": input_games,
        "input_positions": input_positions,
        "kept_games": len(kept),
        "kept_positions": int(sum(len(g["records"]) for g in kept)),
        "dropped": dropped,
    }
    if not paths:
        print(f"No .jsonl files found in {raw_dir}")
    return (kept, summary) if return_summary else kept


def _result_bucket(result):
    """Bucket scalar game results to {-1, 0, +1} by sign."""
    if result > 0:
        return 1
    if result < 0:
        return -1
    return 0


def _split_games_by_result(games, seed):
    """Stratified game-level split (80/10/10) by result bucket."""
    rng = np.random.default_rng(seed)
    by_bucket = {-1: [], 0: [], 1: []}
    for game in games:
        by_bucket[game["result_bucket"]].append(game)

    split = {"train": [], "val": [], "test": []}
    for bucket in (-1, 0, 1):
        group = by_bucket[bucket]
        if not group:
            continue
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        split["train"].extend(group[:n_train])
        split["val"].extend(group[n_train:n_train + n_val])
        split["test"].extend(group[n_train + n_val:])

    for key in ("train", "val", "test"):
        rng.shuffle(split[key])

    # Small-dataset fallback: keep splits non-empty when possible.
    if len(games) >= 3 and len(split["val"]) == 0:
        donor = "test" if len(split["test"]) > 1 else "train"
        if split[donor]:
            split["val"].append(split[donor].pop())
    if len(games) >= 2 and len(split["test"]) == 0:
        donor = "val" if len(split["val"]) > 1 else "train"
        if split[donor]:
            split["test"].append(split[donor].pop())
    if len(split["train"]) == 0:
        donor = "test" if split["test"] else "val"
        if split[donor]:
            split["train"].append(split[donor].pop())

    return split


def _discounted_results(records, gamma):
    """Per-record game_result with a per-ply discount toward the game end.

    target[i] = game_result * gamma**(plies_to_end_of_segment). Gives the
    scalar value head an urgency gradient: a win 5 plies away trains higher
    than a win 50 plies away, so search in saturated won positions prefers
    the faster mate instead of shuffling. gamma >= 1.0 disables (raw results).

    Segments: merge drivers duplicate scarce human games by repeating the
    records INSIDE one file (keeps game-level splits leak-free). A record
    whose FEN equals the file's first FEN starts a new copy, so the discount
    is computed within each copy — otherwise the first copy would be scored
    as hundreds of plies from the end.
    """
    if gamma >= 1.0:
        return [rec.get("game_result", 0) for rec in records]
    start_fen = records[0].get("fen")
    bounds = [i for i, rec in enumerate(records)
              if i == 0 or rec.get("fen") == start_fen]
    bounds.append(len(records))
    out = [0.0] * len(records)
    for a, b in zip(bounds, bounds[1:]):
        for i in range(a, b):
            out[i] = records[i].get("game_result", 0) * (gamma ** (b - 1 - i))
    return out


def _convert_games_to_arrays(games, augment, value_gamma=VALUE_TARGET_GAMMA):
    """Flat conversion of game records to tensors for one split."""
    tensors = []
    values = []
    game_results = []
    policy_targets = []

    for game in tqdm(games, desc="Converting", leave=False):
        discounted = _discounted_results(game["records"], value_gamma)
        for rec, gr in zip(game["records"], discounted):
            is_white = rec["current_player"] == "white"
            half_pending = bool(rec.get("half"))
            tensor = fen_to_tensor(rec["fen"], is_white_turn=is_white,
                                   half_pending=half_pending)
            # mcts_value from data_generation is already from the
            # side-to-move perspective for both White and Black.
            # gr comes pre-discounted from _discounted_results.
            val = rec["mcts_value"]
            pol = policy_dict_to_target(rec["policy"], is_white)

            tensors.append(tensor)
            values.append(val)
            game_results.append(gr)
            policy_targets.append(pol)
            if augment:
                tensors.append(mirror_tensor(tensor))
                values.append(val)
                game_results.append(gr)
                policy_targets.append(mirror_policy(pol))

    if tensors:
        X = np.array(tensors, dtype=np.float32)
        y_value = np.array(values, dtype=np.float32)
        y_result = np.array(game_results, dtype=np.float32)
        y_policy = np.array(policy_targets, dtype=np.float32)
    else:
        X = np.zeros((0,) + TENSOR_SHAPE, dtype=np.float32)
        y_value = np.zeros((0,), dtype=np.float32)
        y_result = np.zeros((0,), dtype=np.float32)
        y_policy = np.zeros((0, POLICY_SIZE), dtype=np.float32)
    return X, y_value, y_result, y_policy


def process_raw_data(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR,
                     augment=True, seed=RANDOM_SEED, include_human=True,
                     max_generation_age=DATA_RETENTION_MAX_GENERATION_AGE,
                     min_nonhuman_plies=DATA_RETENTION_MIN_NONHUMAN_PLIES,
                     value_gamma=VALUE_TARGET_GAMMA):
    """Convert raw game records to training tensors and save.

    When augment=True (default), each position is also horizontally
    mirrored, doubling the dataset size.  The mirror preserves the
    evaluation (same value / game_result) since Monster Chess is
    file-symmetric.
    """
    games, retention_summary = load_all_games(
        raw_dir,
        include_human=include_human,
        max_generation_age=max_generation_age,
        min_nonhuman_plies=min_nonhuman_plies,
        return_summary=True,
    )
    if not games:
        print("No data to process.")
        return
    print(
        "Retention summary: "
        f"kept_games={retention_summary['kept_games']}/{retention_summary['input_games']}, "
        f"kept_positions={retention_summary['kept_positions']}/{retention_summary['input_positions']}"
    )

    split_games = _split_games_by_result(games, seed=seed)
    train_games = split_games["train"]
    val_games = split_games["val"]
    test_games = split_games["test"]

    train_ids = {g["game_id"] for g in train_games}
    val_ids = {g["game_id"] for g in val_games}
    test_ids = {g["game_id"] for g in test_games}
    if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
        raise RuntimeError("Game-level split overlap detected")
    print("Game split integrity: PASS (no overlap across train/val/test game IDs)")
    print(f"  Games: train={len(train_games)}, val={len(val_games)}, test={len(test_games)}")
    print(f"  Processing positions (augment={augment})...")

    if value_gamma < 1.0:
        print(f"  Value targets: game_result discounted per ply (gamma={value_gamma})")
    X_train, yv_train, yr_train, yp_train = _convert_games_to_arrays(train_games, augment, value_gamma)
    X_val, yv_val, yr_val, yp_val = _convert_games_to_arrays(val_games, augment, value_gamma)
    X_test, yv_test, yr_test, yp_test = _convert_games_to_arrays(test_games, augment, value_gamma)

    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y_value = np.concatenate([yv_train, yv_val, yv_test], axis=0)
    y_result = np.concatenate([yr_train, yr_val, yr_test], axis=0)
    y_policy = np.concatenate([yp_train, yp_val, yp_test], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "positions.npy"), X)
    np.save(os.path.join(output_dir, "mcts_values.npy"), y_value)
    np.save(os.path.join(output_dir, "game_results.npy"), y_result)
    np.save(os.path.join(output_dir, "policies.npy"), y_policy)

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    splits = {
        "train": np.arange(0, n_train, dtype=np.int64),
        "val": np.arange(n_train, n_train + n_val, dtype=np.int64),
        "test": np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64),
    }
    np.savez(os.path.join(output_dir, "splits.npz"), **splits)

    with open(os.path.join(output_dir, "split_game_ids.json"), "w") as f:
        json.dump({
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
            "retention": retention_summary,
            "augment": bool(augment),
            "total_positions": int(len(X)),
        }, f, indent=2)

    print(f"\nSaved to {output_dir}:")
    print(f"  positions.npy:    {X.shape}")
    print(f"  mcts_values.npy:  {y_value.shape}")
    print(f"  game_results.npy: {y_result.shape}")
    print(f"  policies.npy:     {y_policy.shape}")
    print(f"  splits.npz:       train={n_train}, val={n_val}, test={n_test}")
    print("  split_game_ids.json: game-level split membership saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process raw game data into training tensors")
    parser.add_argument("--raw-dir", type=str, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DATA_DIR)
    parser.add_argument("--no-augment", action="store_true", help="Disable mirror augmentation")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for deterministic game-level splitting (default: {RANDOM_SEED})")
    parser.add_argument("--exclude-human-games", action="store_true",
                        help="Exclude data/raw/human_games from processing")
    parser.add_argument("--max-generation-age", type=int, default=DATA_RETENTION_MAX_GENERATION_AGE,
                        help="Drop nn_gen* games older than this many generations behind latest (<=0 disables)")
    parser.add_argument("--min-nonhuman-plies", type=int, default=DATA_RETENTION_MIN_NONHUMAN_PLIES,
                        help="Drop non-human games shorter than this many plies (<=0 disables)")
    parser.add_argument("--value-gamma", type=float, default=VALUE_TARGET_GAMMA,
                        help="Per-ply discount on game_result targets (1.0 = off)")
    args = parser.parse_args()
    if args.max_generation_age is not None and args.max_generation_age < 0:
        raise ValueError("--max-generation-age must be >= 0")
    if args.min_nonhuman_plies < 0:
        raise ValueError("--min-nonhuman-plies must be >= 0")

    process_raw_data(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        augment=not args.no_augment,
        seed=args.seed,
        include_human=not args.exclude_human_games,
        max_generation_age=args.max_generation_age,
        min_nonhuman_plies=args.min_nonhuman_plies,
        value_gamma=args.value_gamma,
    )
