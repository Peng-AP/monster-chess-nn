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

# Curriculum endgame positions for Monster Chess.
#
# Key strategic principles for Black vs double-move king:
#   - Need 3 heavy pieces to force capture: 2 rank/file barriers + 1 to
#     cover the edge. With only 2 barriers the king jumps over in 2 moves.
#   - 2 heavy pieces can ISOLATE the king to a strip (e.g. rank 1) by
#     controlling 2 adjacent ranks/files from a distance the king can't
#     reach in 2 moves.
#   - A protected queen is untouchable: to capture a defended piece, White
#     must start adjacent (take + escape). But a queen attacks all adjacent
#     squares, so White can never end a turn next to it — meaning it can
#     never START adjacent on the next turn to capture it.
#   - All barrier pieces must stay far enough that the king (moving 2)
#     cannot reach them.
#
# Forced result per tier: Tier 1-2 = -1.0, Tier 3-5 = -0.7
# (see CURRICULUM_TIER_RESULTS below)
CURRICULUM_FENS = [
    # === TIER 1: Forced capture — unavoidable king loss ===
    # Pattern: king in corner, 3 pieces covering all 3 reachable ranks/files
    # from the far side (unreachable in 2 king moves).
    # White has NO pawns — this is a pure endgame.
    #
    # Ka1: ranks 1,2,3 all covered. Every 2-move path lands on attacked sq.
    "7k/8/8/8/8/7r/7r/K6q b - - 0 1",   # Ka1, Qh1+Rh2+Rh3
    "4k3/8/8/8/8/7r/7r/K6q b - - 0 1",   # Ka1, Qh1+Rh2+Rh3 (Bk center)
    # Kh1: mirror
    "k7/8/8/8/8/r7/r7/q6K b - - 0 1",   # Kh1, Qa1+Ra2+Ra3
    "4k3/8/8/8/8/r7/r7/q6K b - - 0 1",   # Kh1, Qa1+Ra2+Ra3
    # Ka8: ranks 8,7,6 covered
    "K6q/7r/7r/8/8/8/8/7k b - - 0 1",   # Ka8, Qh8+Rh7+Rh6
    # Kh8: mirror
    "q6K/r7/r7/8/8/8/8/k7 b - - 0 1",   # Kh8, Qa8+Ra7+Ra6
    # File-based: king on a-file, pieces controlling files a,b,c from far rank
    "K7/8/8/8/8/8/8/qrrk4 b - - 0 1",   # Ka8, Qa1+Rb1+Rc1
    "7K/8/8/8/8/8/8/4krrq b - - 0 1",   # Kh8, Qh1+Rg1+Rf1

    # === TIER 2: One move from forced capture ===
    # Black needs to move one piece into position to complete the net.
    # Still NO White pawns — pure endgame.
    #
    # Queen needs to come to rank 1 to seal the trap
    "7k/8/8/8/8/7r/7r/K7 b - - 0 1",   # Ka1, Rh2+Rh3 — Bk plays Qh1 idea
    "k7/8/8/8/8/r7/r7/7K b - - 0 1",   # Kh1, Ra2+Ra3 — needs Qa1
    # Rook needs to tighten from rank 4 to rank 3
    "7k/8/8/8/7r/8/7r/K6q b - - 0 1",   # Ka1, Qh1+Rh2+Rh4 — Rh4->Rh3
    "4k3/8/8/8/r7/8/r7/q6K b - - 0 1",   # Kh1, Qa1+Ra2+Ra4 — Ra4->Ra3
    # Two rooks isolate king, queen approaching from distance
    "4k2q/8/8/8/8/7r/7r/K7 b - - 0 1",   # Ka1, Rh2+Rh3, Qh8 coming down
    "q3k3/8/8/8/8/r7/r7/7K b - - 0 1",   # Kh1, Ra2+Ra3, Qa8 coming down
    # Rook on king's file, one rank away — delivers capture next move
    "4k3/8/8/8/8/8/r7/K6r b - - 0 1",   # Ka1, Rh1 attacks rank 1, Ra2 blocks rank 2

    # === TIER 3: Isolation — 2 pieces confine king to edge strip ===
    # King trapped on rank 1, can slide but can't escape upward.
    # White has PAWNS — distinguishes from normal opening positions.
    #
    "4k3/8/8/8/8/7r/7r/K1PP4 b - - 0 1",   # Ka1 confined, White still has pawns
    "4k3/8/8/8/8/r7/r7/2PPK3 b - - 0 1",   # Ke1 confined, White has pawns
    "4k3/8/8/8/8/7r/7r/3KP3 b - - 0 1",   # Kd1 confined, has pawn
    # King trapped on a-file by file barriers, with pawns
    "K1P5/8/8/8/8/8/1r6/1r2k3 b - - 0 1",   # Ka8 confined, has pawn
    # Protected queen as immovable barrier (queen defended by rook)
    "4k3/8/8/8/8/r3q3/8/K1P5 b - - 0 1",   # Qe3 defended by Ra3, White has pawn
    "4k3/8/8/8/8/q3r3/8/4PK2 b - - 0 1",   # Qa3 defended by Re3, White has pawn

    # === TIER 4: Overwhelming material, king near edge ===
    # White has PAWNS — model must learn that even with pawns, this is lost.
    # Q+2R
    "4k3/8/8/8/q7/r7/r7/K1PP4 b - - 0 1",
    "4k3/8/8/8/r7/r7/q7/2PPK3 b - - 0 1",
    "4k3/8/8/8/r7/q7/r7/2PKP3 b - - 0 1",
    # 2Q+R
    "4k3/8/8/8/q7/q7/r7/2PKP3 b - - 0 1",
    "4k3/8/8/8/r7/q7/q7/3KPP2 b - - 0 1",
    # 4R vs king+pawns
    "4k3/8/8/r7/r7/r7/r7/K1PP4 b - - 0 1",
    "4k3/8/8/r7/r7/r7/r7/2PPK3 b - - 0 1",
    # 3R vs king+pawns
    "4k3/8/8/8/r7/r7/r7/2PKP3 b - - 0 1",

    # === TIER 5: Mid-game Black advantage ===
    # White has PAWNS — realistic mid-game where Black has promoted.
    "4k3/8/8/8/8/q7/q7/2PPK3 b - - 0 1",   # 2 queens vs king+pawns
    "r3k3/8/8/8/8/r7/q7/3KPP2 b - - 0 1",   # Q+2R vs king+pawns
]

# Value label per tier: strong signal for forced wins, softer for advantages
CURRICULUM_TIER_BOUNDARIES = [8, 15, 21, 29]  # indices where tiers end
CURRICULUM_TIER_VALUES = [-1.0, -1.0, -0.7, -0.7, -0.7]  # per tier

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
DIRICHLET_ALPHA = 0.3    # Dirichlet noise concentration parameter
DIRICHLET_EPSILON = 0.25 # fraction of noise mixed into root priors
POLICY_LOSS_WEIGHT = 1.0 # weight of policy CE loss relative to value MSE

# Training
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
VALUE_LOSS_EXPONENT = 2.5  # power-law loss (Stockfish uses 2.5 vs MSE's 2.0)
LR_GAMMA = 0.95           # exponential LR decay per epoch
EPOCHS = 50
VALUE_TARGET = "mcts_value"  # "game_result" or "mcts_value" or "blend"
BLEND_WEIGHT = 0.7      # weight for mcts_value when VALUE_TARGET="blend"
BLEND_START = 0.8       # lambda at epoch 1 (trust MCTS more early)
BLEND_END = 0.5         # lambda at final epoch (shift toward game results)
MODEL_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "models")

# Human data
HUMAN_DATA_WEIGHT = 40    # repeat human game positions N times during processing
SLIDING_WINDOW = 2        # keep only the last N NN generations for training
OPPONENT_SIMULATIONS = 200  # MCTS sims for frozen opponent in alternating training

# Sub-goal reward shaping (Black strategic progress in heuristic eval)
WHITE_PAWN_VALUE = 0.18             # value per White pawn (was 0.10)
PAWN_ELIMINATION_BONUS = 0.10       # bonus per eliminated White pawn (4 - count)
BLOCKED_PAWN_PENALTY = 0.08         # penalty per White pawn that can't advance
KING_DISPLACEMENT_WEIGHT = 0.06     # reward per unit of king displacement from center
KING_MOBILITY_WEIGHT = 0.01         # penalty per restricted square (2-move reachability)
BARRIER_RANK_FILE_WEIGHT = 0.12     # bonus per barrier rank/file between king and edge

# File paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
