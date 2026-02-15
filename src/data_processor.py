import chess
import json
import os

import numpy as np
from tqdm import tqdm

from config import (
    TENSOR_SHAPE, TURN_LAYER,
    MOVE_COUNT_LAYER, PAWN_ADVANCEMENT_LAYER,
    POLICY_SIZE,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    HUMAN_DATA_WEIGHT,
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


def load_all_games(raw_dir):
    """Load all .jsonl game files from the raw data directory (recursive).

    Human game data (from human_games/ subdirectory) is repeated
    HUMAN_DATA_WEIGHT times to upweight its influence during training.
    """
    records = []
    human_records = []
    paths = []
    for dirpath, _dirnames, filenames in os.walk(raw_dir):
        for fname in sorted(filenames):
            if fname.endswith(".jsonl"):
                paths.append(os.path.join(dirpath, fname))

    if not paths:
        print(f"No .jsonl files found in {raw_dir} (searched recursively)")
        return records

    for path in tqdm(paths, desc="Loading games"):
        is_human = "human_games" in path
        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line.strip())
                if is_human:
                    human_records.append(rec)
                else:
                    records.append(rec)

    if human_records:
        n_human = len(human_records)
        repeated = human_records * HUMAN_DATA_WEIGHT
        records.extend(repeated)
        print(f"  Human data: {n_human} positions x{HUMAN_DATA_WEIGHT} = {len(repeated)} "
              f"(of {len(records)} total)")

    return records


def process_raw_data(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR,
                     augment=True):
    """Convert raw game records to training tensors and save.

    When augment=True (default), each position is also horizontally
    mirrored, doubling the dataset size.  The mirror preserves the
    evaluation (same value / game_result) since Monster Chess is
    file-symmetric.
    """
    records = load_all_games(raw_dir)
    if not records:
        print("No data to process.")
        return

    print(f"Processing {len(records)} positions (augment={augment})...")

    tensors = []
    values = []
    game_results = []
    policy_targets = []

    for rec in tqdm(records, desc="Converting"):
        is_white = rec["current_player"] == "white"
        tensor = fen_to_tensor(rec["fen"], is_white_turn=is_white)
        # mcts_value is from the current player's perspective;
        # convert to White's perspective for consistent training targets
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

    X = np.array(tensors, dtype=np.float32)
    y_value = np.array(values, dtype=np.float32)
    y_result = np.array(game_results, dtype=np.float32)
    y_policy = np.array(policy_targets, dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "positions.npy"), X)
    np.save(os.path.join(output_dir, "mcts_values.npy"), y_value)
    np.save(os.path.join(output_dir, "game_results.npy"), y_result)
    np.save(os.path.join(output_dir, "policies.npy"), y_policy)

    # Train / val / test split (80/10/10, stratified by result)
    n = len(X)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }
    np.savez(os.path.join(output_dir, "splits.npz"), **splits)

    print(f"\nSaved to {output_dir}:")
    print(f"  positions.npy:    {X.shape}")
    print(f"  mcts_values.npy:  {y_value.shape}")
    print(f"  game_results.npy: {y_result.shape}")
    print(f"  policies.npy:     {y_policy.shape}")
    print(f"  splits.npz:       train={n_train}, val={n_val}, test={n - n_train - n_val}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process raw game data into training tensors")
    parser.add_argument("--raw-dir", type=str, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DATA_DIR)
    parser.add_argument("--no-augment", action="store_true", help="Disable mirror augmentation")
    args = parser.parse_args()
    process_raw_data(raw_dir=args.raw_dir, output_dir=args.output_dir,
                     augment=not args.no_augment)
