import chess
import json
import os

import numpy as np
from tqdm import tqdm

from config import (
    TENSOR_SHAPE, TURN_LAYER,
    MOVE_COUNT_LAYER, PAWN_ADVANCEMENT_LAYER,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
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


def load_all_games(raw_dir):
    """Load all .jsonl game files from the raw data directory."""
    records = []
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".jsonl"))
    if not files:
        print(f"No .jsonl files found in {raw_dir}")
        return records

    for fname in tqdm(files, desc="Loading games"):
        path = os.path.join(raw_dir, fname)
        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line.strip())
                records.append(rec)
    return records


def process_raw_data(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR):
    """Convert raw game records to training tensors and save."""
    records = load_all_games(raw_dir)
    if not records:
        print("No data to process.")
        return

    print(f"Processing {len(records)} positions...")

    tensors = []
    values = []
    game_results = []

    for rec in tqdm(records, desc="Converting"):
        is_white = rec["current_player"] == "white"
        tensor = fen_to_tensor(rec["fen"], is_white_turn=is_white)
        tensors.append(tensor)
        values.append(rec["mcts_value"])
        game_results.append(rec["game_result"])

    X = np.array(tensors, dtype=np.float32)
    y_value = np.array(values, dtype=np.float32)
    y_result = np.array(game_results, dtype=np.float32)

    # Save policy targets separately (sparse, variable-size dicts)
    policies = [rec["policy"] for rec in records]

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "positions.npy"), X)
    np.save(os.path.join(output_dir, "mcts_values.npy"), y_value)
    np.save(os.path.join(output_dir, "game_results.npy"), y_result)

    with open(os.path.join(output_dir, "policies.jsonl"), "w") as f:
        for p in policies:
            f.write(json.dumps(p) + "\n")

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
    print(f"  policies.jsonl:   {len(policies)} entries")
    print(f"  splits.npz:       train={n_train}, val={n_val}, test={n - n_train - n_val}")


if __name__ == "__main__":
    process_raw_data()
