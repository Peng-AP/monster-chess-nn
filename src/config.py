import os

# Monster Chess starting position
# White: King on e1, Pawns on c2-f2
# Black: Full standard army
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"

# MCTS parameters
MCTS_SIMULATIONS = 800
EXPLORATION_CONSTANT = 1.41  # UCB1 C parameter
TEMPERATURE_MOVES = 15       # Use high temperature for first N moves
TEMPERATURE_HIGH = 1.0       # Exploration temperature (early game)
TEMPERATURE_LOW = 0.1        # Exploitation temperature (late game)

# Game parameters
MAX_GAME_TURNS = 150

# Data generation
NUM_GAMES = 100

# Curriculum endgame positions — Black has barriers already formed
# The key insight: scattered queens DON'T work because White's double-move
# king escapes. Black needs queens/rooks forming rank/file barriers that
# restrict the king BEFORE trying to deliver mate.
CURRICULUM_FENS = [
    # --- 2 rooks vs lone king (no immediate capture) ---
    # Rooks control adjacent ranks, king on rank 1
    "4k3/8/8/8/8/r7/r7/3K4 b - - 0 1",
    "4k3/8/8/8/8/7r/7r/3K4 b - - 0 1",
    "4k3/8/8/8/r7/8/r7/3K4 b - - 0 1",
    "4k3/8/r7/8/8/8/7r/4K3 b - - 0 1",
    # --- 3 rooks vs lone king (tighter net) ---
    "4k3/8/8/8/4r3/1r6/r7/3K4 b - - 0 1",
    "4k3/8/8/8/2r5/r7/1r6/3K4 b - - 0 1",
    "4k3/8/8/8/8/2r5/r4r2/3K4 b - - 0 1",
    "4k3/8/8/8/r7/r7/r7/3K4 b - - 0 1",
    # --- Queen + rook(s) vs lone king ---
    "4k3/8/8/q7/8/8/r7/7K b - - 0 1",
    # --- Mid-game Black advantage (has extra heavy pieces + pawns) ---
    "4k3/pppp4/8/8/8/8/2PP4/4K3 b - - 0 1",
]

# Tensor encoding
NUM_PIECE_LAYERS = 12
TURN_LAYER = 12
MOVE_COUNT_LAYER = 13
PAWN_ADVANCEMENT_LAYER = 14
TOTAL_LAYERS = 15
TENSOR_SHAPE = (8, 8, TOTAL_LAYERS)

# Policy head
POLICY_SIZE = 4096       # flat from_sq(64) * to_sq(64) encoding
C_PUCT = 1.5             # PUCT exploration constant (replaces UCB1 C)
POLICY_LOSS_WEIGHT = 1.0 # weight of policy CE loss relative to value MSE

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
VALUE_TARGET = "blend"  # "game_result" or "mcts_value" or "blend"
BLEND_WEIGHT = 0.7  # weight for mcts_value when VALUE_TARGET="blend"
MODEL_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "models")

# File paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
