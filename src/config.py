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
    # === TIER 1: Immediate king capture (Black takes king this move) ===
    # Rook directly attacks king on same rank/file
    "4k3/8/8/8/8/8/4r3/4K3 b - - 0 1",   # Re2 takes Ke1
    "4k3/8/8/8/8/8/8/r3K3 b - - 0 1",   # Ra1 takes Ke1 along rank
    "4k3/8/8/8/8/8/7r/7K b - - 0 1",   # Rh2 takes Kh1
    "4k3/8/8/8/8/8/q7/K7 b - - 0 1",   # Qa2 takes Ka1
    "7k/8/8/8/8/8/6q1/7K b - - 0 1",   # Qg2 takes Kh1
    "4k3/8/8/8/8/8/3r4/3K4 b - - 0 1",   # Rd2 takes Kd1
    "4k3/8/8/8/8/8/r7/K7 b - - 0 1",   # Ra2 takes Ka1
    "7k/8/8/8/8/8/7r/7K b - - 0 1",   # Rh2 takes Kh1

    # === TIER 2: One move from capture (barrier + attack setup) ===
    # Rook on same file, one rank away — needs to advance to capture
    "4k3/8/8/8/8/4r3/8/4K3 b - - 0 1",   # Re3 can play Re1 or Re2
    "4k3/8/8/8/8/r7/8/K7 b - - 0 1",   # Ra3 approaches Ka1
    "7k/8/8/8/8/7r/8/7K b - - 0 1",   # Rh3 approaches Kh1
    "4k3/8/8/8/8/3q4/8/3K4 b - - 0 1",   # Qd3 approaches Kd1
    # Two rooks — one blocks escape, other delivers capture
    "4k3/8/8/8/8/r7/r7/K7 b - - 0 1",   # Ra2 blocks rank 2, Ra3->Ra1
    "7k/8/8/8/8/7r/7r/7K b - - 0 1",   # Rh2 blocks, Rh3->Rh1
    "4k3/8/8/8/8/8/1r6/Kr6 b - - 0 1",   # Ra1 attacks king, Rb2 covers

    # === TIER 3: Overwhelming material, king near edge ===
    # Queen + 2 rooks vs lone king
    "4k3/8/8/8/q7/r7/r7/3K4 b - - 0 1",
    "4k3/8/8/8/r7/q7/r7/7K b - - 0 1",
    "4k3/8/8/q7/8/r7/r7/K7 b - - 0 1",
    # 2 queens + rook vs lone king
    "4k3/8/8/8/q7/q7/r7/3K4 b - - 0 1",
    "4k3/8/8/q7/8/q7/8/r2K4 b - - 0 1",
    # Queen + 2 rooks, king in corner
    "4k3/8/8/8/q7/r7/r7/K7 b - - 0 1",
    "4k3/8/8/8/r7/r7/q7/7K b - - 0 1",
    # 4 rooks vs lone king
    "4k3/8/8/r7/r7/r7/r7/3K4 b - - 0 1",
    "4k3/8/8/r7/r7/r7/r7/K7 b - - 0 1",
    # 3 rooks vs lone king
    "4k3/8/8/8/r7/r7/r7/3K4 b - - 0 1",
    "4k3/8/8/8/8/2r5/r4r2/3K4 b - - 0 1",

    # === TIER 4: Mid-game Black advantage ===
    "4k3/8/8/8/8/q7/q7/4K3 b - - 0 1",
    "r3k3/8/8/8/8/r7/q7/4K3 b - - 0 1",
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
