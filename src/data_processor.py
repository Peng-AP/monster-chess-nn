import chess
import json
import os
import re

import numpy as np
from tqdm import tqdm

from config import (
    TENSOR_SHAPE, TURN_LAYER,
    MOVE_COUNT_LAYER, PAWN_ADVANCEMENT_LAYER,
    POLICY_SIZE,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    HUMAN_DATA_WEIGHT, SLIDING_WINDOW, POSITION_BUDGET, RANDOM_SEED,
)

# Piece -> layer index
PIECE_TO_LAYER = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def fen_to_tensor(fen, is_white_turn=True):
    """Convert a FEN string to an (8, 8, 15) tensor.

    Layers:
      0-11: piece positions (binary)
      12:   turn indicator (+1 White, -1 Black)
      13:   move count within turn (0 or 1 for White's double-move)
      14:   White pawn advancement gradient (0.0 at rank 2, 1.0 at rank 8)
    """
    board = chess.Board(fen)
    tensor = np.zeros(TENSOR_SHAPE, dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            layer = PIECE_TO_LAYER[(piece.piece_type, piece.color)]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[rank, file, layer] = 1.0

    # Turn indicator
    tensor[:, :, TURN_LAYER] = 1.0 if is_white_turn else -1.0

    # Move count layer (always 0 at position level — the MCTS treats
    # double-moves atomically, so we record 0 here; this layer is
    # reserved for future per-half-move encoding)
    tensor[:, :, MOVE_COUNT_LAYER] = 0.0

    # White pawn advancement gradient
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        tensor[rank, file, PAWN_ADVANCEMENT_LAYER] = (rank - 1) / 6.0

    return tensor


def mirror_tensor(tensor):
    """Horizontally mirror a position tensor (flip files a<->h).

    Monster Chess is symmetric across the file axis, so mirroring
    produces an equally valid position with the same evaluation.
    This doubles training data for free.
    """
    # Flip along the file axis (axis 1): file 0<->7, 1<->6, etc.
    return tensor[:, ::-1, :].copy()


# ------------------------------------------------------------------
# Policy encoding: flat from_sq * 64 + to_sq  (4096 indices)
# ------------------------------------------------------------------

def move_to_index(move):
    """Convert a chess.Move to a flat policy index."""
    return move.from_square * 64 + move.to_square


def mirror_move_index(idx):
    """Mirror a flat policy index across the file axis (a<->h)."""
    from_sq = idx // 64
    to_sq = idx % 64
    from_file, from_rank = from_sq % 8, from_sq // 8
    to_file, to_rank = to_sq % 8, to_sq // 8
    new_from = from_rank * 8 + (7 - from_file)
    new_to = to_rank * 8 + (7 - to_file)
    return new_from * 64 + new_to


def policy_dict_to_target(policy_dict, is_white):
    """Convert an MCTS action_probs dict to a dense policy target vector.

    For Black: each key is a UCI move string -> index directly.
    For White: each key is "m1_uci,m2_uci". We marginalize over m2 to
    get P(m1), since the policy head predicts single moves and m2 is
    evaluated from the post-m1 board state during MCTS.
    """
    target = np.zeros(POLICY_SIZE, dtype=np.float32)
    if policy_dict is None:
        return target  # uniform-ish fallback (all zeros, masked later)

    for action_str, prob in policy_dict.items():
        if is_white:
            m1_uci = action_str.split(",")[0]
            move = chess.Move.from_uci(m1_uci)
        else:
            move = chess.Move.from_uci(action_str)
        target[move_to_index(move)] += prob

    # Renormalize (White's m1 marginal should already sum to ~1)
    total = target.sum()
    if total > 0:
        target /= total
    return target


def mirror_policy(policy_vec):
    """Mirror a dense policy vector across the file axis."""
    mirrored = np.zeros_like(policy_vec)
    for idx in range(POLICY_SIZE):
        if policy_vec[idx] > 0:
            mirrored[mirror_move_index(idx)] = policy_vec[idx]
    return mirrored


def _collect_generation_dirs(subdirs):
    """Map generation number -> list of matching nn_gen* directory names."""
    gen_pattern = re.compile(r'^nn_gen(\d+)(?:_.*)?$')
    gen_to_dirs = {}
    for d in subdirs:
        m = gen_pattern.match(d)
        if not m:
            continue
        gen = int(m.group(1))
        gen_to_dirs.setdefault(gen, []).append(d)
    return gen_to_dirs


def _count_positions_in_dir(raw_dir, dirname):
    """Count recorded positions (JSONL lines) in one raw data subdirectory."""
    total = 0
    dpath = os.path.join(raw_dir, dirname)
    for fname in os.listdir(dpath):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(dpath, fname)
        with open(path, "r", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def _filter_dirs(raw_dir, keep_generations=None, position_budget=None, include_human=True):
    """Decide which subdirectories to include based on sliding window.

    Always includes: curriculum_bootstrap/ and optionally human_games/
    Includes either:
      - last `keep_generations` nn_gen* pairs (normal + curriculum), or
      - enough recent generations to hit `position_budget` raw positions.
    Excludes everything else (normal/ heuristic bootstrap, older generations).
    """
    if keep_generations is not None and position_budget is not None:
        raise ValueError("Use either keep_generations or position_budget, not both.")
    subdirs = [d for d in os.listdir(raw_dir)
               if os.path.isdir(os.path.join(raw_dir, d))]

    gen_to_dirs = _collect_generation_dirs(subdirs)
    gen_numbers = sorted(gen_to_dirs.keys())
    kept_gens = []
    position_total = 0
    generation_position_counts = {}
    if keep_generations is not None:
        kept_gens = gen_numbers[-keep_generations:] if gen_numbers else []
    elif position_budget is not None:
        for gen in reversed(gen_numbers):
            gen_dirs = gen_to_dirs.get(gen, [])
            gen_pos = sum(_count_positions_in_dir(raw_dir, d) for d in gen_dirs)
            generation_position_counts[gen] = gen_pos
            kept_gens.append(gen)
            position_total += gen_pos
            if position_total >= position_budget:
                break
        kept_gens.sort()

    kept_gen_set = set(kept_gens)
    kept_gen_dirs = set()
    for d in subdirs:
        m = re.match(r'^nn_gen(\d+)(?:_.*)?$', d)
        if m and int(m.group(1)) in kept_gen_set:
            kept_gen_dirs.add(d)

    # Always-include dirs
    always_include = {"curriculum_bootstrap"}
    if include_human:
        always_include.add("human_games")

    include = []
    exclude = []
    for d in sorted(subdirs):
        if d in always_include or d in kept_gen_dirs:
            include.append(d)
        else:
            exclude.append(d)

    summary = {
        "kept_generations": kept_gens,
        "position_budget": position_budget,
        "estimated_positions": position_total,
        "generation_position_counts": generation_position_counts,
    }
    return include, exclude, summary


def load_all_games(raw_dir, keep_generations=None, position_budget=None, include_human=True):
    """Load game files as game-level units.

    Returns a list of dicts:
      {
        "game_id": relative path,
        "records": [position records...],
        "is_human": bool,
        "result_bucket": -1/0/1 (sign of game_result)
      }

    This keeps game boundaries intact so train/val/test can be split
    at the game level (no leakage across splits).
    """
    paths = []

    if keep_generations is not None or position_budget is not None:
        include, exclude, summary = _filter_dirs(
            raw_dir,
            keep_generations=keep_generations,
            position_budget=position_budget,
            include_human=include_human,
        )
        if position_budget is not None:
            print(
                f"Position budget window: target={position_budget}, "
                f"estimated={summary['estimated_positions']} raw positions"
            )
            print(f"  Generations kept: {summary['kept_generations']}")
        else:
            print(f"Sliding window: keeping last {keep_generations} generations")
        print(f"  Human games: {'included' if include_human else 'excluded'}")
        print(f"  Include: {', '.join(include)}")
        if exclude:
            print(f"  Exclude: {', '.join(exclude)}")
        for d in include:
            dpath = os.path.join(raw_dir, d)
            for fname in sorted(os.listdir(dpath)):
                if fname.endswith(".jsonl"):
                    paths.append(os.path.join(dpath, fname))
    else:
        for dirpath, _dirnames, filenames in os.walk(raw_dir):
            rel_dir = os.path.relpath(dirpath, raw_dir).replace("\\", "/")
            if (not include_human) and (rel_dir == "human_games" or rel_dir.startswith("human_games/")):
                continue
            for fname in sorted(filenames):
                if fname.endswith(".jsonl"):
                    paths.append(os.path.join(dirpath, fname))

    if not paths:
        print(f"No .jsonl files found in {raw_dir}")
        return []

    games = []
    for path in tqdm(paths, desc="Loading games"):
        rel = os.path.relpath(path, raw_dir).replace("\\", "/")
        is_human = "human_games/" in rel or rel.startswith("human_games")
        with open(path, "r") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]
        if not records:
            continue
        result = records[-1].get("game_result", 0)
        games.append({
            "game_id": rel,
            "records": records,
            "is_human": is_human,
            "result_bucket": _result_bucket(result),
        })

    return games


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


def _convert_games_to_arrays(games, augment, human_repeat):
    """Convert game records to tensors for one split."""
    tensors = []
    values = []
    game_results = []
    policy_targets = []
    split_game_ids = []

    for game in tqdm(games, desc="Converting", leave=False):
        repeat = human_repeat if game["is_human"] else 1
        for _ in range(repeat):
            split_game_ids.append(game["game_id"])
            for rec in game["records"]:
                is_white = rec["current_player"] == "white"
                tensor = fen_to_tensor(rec["fen"], is_white_turn=is_white)
                mv = rec["mcts_value"]
                val = mv if is_white else -mv
                gr = rec["game_result"]
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

    return X, y_value, y_result, y_policy, split_game_ids


def process_raw_data(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR,
                     augment=True, keep_generations=None, position_budget=None,
                     seed=RANDOM_SEED,
                     include_human=True):
    """Convert raw game records to training tensors and save.

    When augment=True (default), each position is also horizontally
    mirrored, doubling the dataset size.  The mirror preserves the
    evaluation (same value / game_result) since Monster Chess is
    file-symmetric.

    When keep_generations is set, only the last N NN generations are loaded.
    When position_budget is set, enough recent generations are loaded to hit
    at least that many raw positions.
    """
    games = load_all_games(
        raw_dir,
        keep_generations=keep_generations,
        position_budget=position_budget,
        include_human=include_human,
    )
    if not games:
        print("No data to process.")
        return

    split_games = _split_games_by_result(games, seed=seed)
    train_games = split_games["train"]
    val_games = split_games["val"]
    test_games = split_games["test"]

    train_ids = {g["game_id"] for g in train_games}
    val_ids = {g["game_id"] for g in val_games}
    test_ids = {g["game_id"] for g in test_games}
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError("Game-level split overlap detected")
    print("Game split integrity: PASS (no overlap across train/val/test game IDs)")
    print(f"  Games: train={len(train_games)}, val={len(val_games)}, test={len(test_games)}")
    print(f"  Processing positions (augment={augment})...")

    # Upweight human games in TRAIN split only to avoid validation/test skew.
    X_train, yv_train, yr_train, yp_train, train_game_ids = _convert_games_to_arrays(
        train_games, augment=augment, human_repeat=HUMAN_DATA_WEIGHT,
    )
    X_val, yv_val, yr_val, yp_val, val_game_ids = _convert_games_to_arrays(
        val_games, augment=augment, human_repeat=1,
    )
    X_test, yv_test, yr_test, yp_test, test_game_ids = _convert_games_to_arrays(
        test_games, augment=augment, human_repeat=1,
    )

    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y_value = np.concatenate([yv_train, yv_val, yv_test], axis=0)
    y_result = np.concatenate([yr_train, yr_val, yr_test], axis=0)
    y_policy = np.concatenate([yp_train, yp_val, yp_test], axis=0)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "positions.npy"), X)
    np.save(os.path.join(output_dir, "mcts_values.npy"), y_value)
    np.save(os.path.join(output_dir, "game_results.npy"), y_result)
    np.save(os.path.join(output_dir, "policies.npy"), y_policy)

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    splits = {
        "train": np.arange(0, n_train, dtype=np.int64),
        "val": np.arange(n_train, n_train + n_val, dtype=np.int64),
        "test": np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64),
    }
    np.savez(os.path.join(output_dir, "splits.npz"), **splits)

    split_game_ids = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
        "train_weighted_instances": len(train_game_ids),
        "val_weighted_instances": len(val_game_ids),
        "test_weighted_instances": len(test_game_ids),
    }
    with open(os.path.join(output_dir, "split_game_ids.json"), "w") as f:
        json.dump(split_game_ids, f, indent=2)

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
    parser.add_argument("--keep-generations", type=int, default=None,
                        help=f"Sliding window: keep last N generations (default: all)")
    parser.add_argument("--position-budget", type=int, default=None,
                        help=f"Position budget window: include enough recent generations to hit N raw positions (default: {POSITION_BUDGET})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for deterministic game-level splitting (default: {RANDOM_SEED})")
    parser.add_argument("--exclude-human-games", action="store_true",
                        help="Exclude data/raw/human_games from processing")
    args = parser.parse_args()
    if args.keep_generations is not None and args.position_budget is not None:
        raise ValueError("Specify only one of --keep-generations or --position-budget")

    process_raw_data(raw_dir=args.raw_dir, output_dir=args.output_dir,
                     augment=not args.no_augment,
                     keep_generations=args.keep_generations,
                     position_budget=args.position_budget,
                     seed=args.seed,
                     include_human=not args.exclude_human_games)
