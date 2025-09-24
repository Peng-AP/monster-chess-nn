import chess
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Define the mapping from a piece to a layer in our tensor
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
NUM_PIECE_LAYERS = 12
TURN_LAYER_INDEX = 12
TENSOR_SHAPE = (8, 8, NUM_PIECE_LAYERS + 1)


def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Converts a FEN string into a 3D NumPy array (tensor).
    Shape: (8, 8, 13)
    - 12 layers for piece positions (6 for white, 6 for black).
    - 1 layer for the side to move (1 for White, -1 for Black).
    """
    board = chess.Board(fen)
    tensor = np.zeros(TENSOR_SHAPE, dtype=np.int8)

    # Populate piece layers
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            layer = PIECE_TO_LAYER[(piece.piece_type, piece.color)]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[rank, file, layer] = 1

    # Populate turn layer
    if board.turn == chess.WHITE:
        tensor[:, :, TURN_LAYER_INDEX] = 1
    else:
        tensor[:, :, TURN_LAYER_INDEX] = -1

    return tensor


def process_raw_data(input_path: str, output_dir: str):
    """
    Reads a CSV of raw game data, converts it to tensors, and saves it.
    """
    print(f"Reading raw data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Found {len(df)} positions. Converting to tensors...")
    all_tensors = []
    all_labels = []

    # Use tqdm for a progress bar
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        tensor = fen_to_tensor(row['fen'])
        all_tensors.append(tensor)
        all_labels.append(row['outcome'])

    X = np.array(all_tensors, dtype=np.int8)
    y = np.array(all_labels, dtype=np.int8)

    print(f"Conversion complete. Tensor shape: {X.shape}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.join(output_dir, "training_data.npy")
    labels_path = os.path.join(output_dir, "training_labels.npy")

    np.save(data_path, X)
    np.save(labels_path, y)

    print(f"Successfully saved processed data to:")
    print(f"  - {data_path}")
    print(f"  - {labels_path}")


if __name__ == "__main__":
    # This allows the script to be run directly.
    # It finds the project root and sets the correct file paths.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_csv = os.path.join(project_root, 'data', 'raw', 'monster_chess_data.csv')
    output_folder = os.path.join(project_root, 'data', 'processed')

    # Check if the raw data exists
    if not os.path.exists(input_csv):
        print(f"Error: Raw data file not found at {input_csv}")
        print("Please run the data generation script first.")
    else:
        process_raw_data(input_csv, output_folder)
