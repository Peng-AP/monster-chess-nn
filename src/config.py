import os

# ============================ RECIPES ============================
# The defaults below reproduce NEITHER live checkpoint. They are starting
# points for new work, not a record of how anything was built. To rebuild a
# live model, pass its recipe explicitly — do not trust a bare default.
#
#   v17 (the incumbent, 15ch, WDL head):
#     data_processor.py --channels 15 --value-floor 0.97 --value-horizon 10
#     train.py --target game_result --value-head wdl --epochs 30 --seed 42
#
#   ramp (models/rejected/fresh_start_v18_ramp, 15ch, scalar head):
#     data_processor.py --channels 15 --value-floor 0.5 --value-horizon 60
#     train.py --target game_result --value-head scalar --epochs 30 --seed 42
#
# Known divergences between these defaults and the live models, all deliberate:
#   * VALUE_HEAD_MODE is "scalar" (ramp-era) while the incumbent is "wdl".
#   * VALUE_TARGET_FLOOR/HORIZON are 0.97/10 (v17-era), not the ramp's 0.5/60.
#   * TENSOR_SHAPE is 17 channels, but BOTH live checkpoints are 15 — omit
#     --channels 15 and you get a corpus neither model can be compared on.
#   * MCTS_SIMULATIONS is 800; every match/benchmark actually runs 400.
# =================================================================

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
RANDOM_SEED = 42


# Tensor encoding
TURN_LAYER = 12
MOVE_COUNT_LAYER = 13
# v16/v17 checkpoints used channel 14 as a White-only pawn-advancement
# feature. New training uses a symmetric positional encoding; inference
# detects legacy checkpoints from the stem weight shape and asks encoding.py
# for the old 15-channel layout automatically.
LEGACY_TENSOR_CHANNELS = 15
LEGACY_PAWN_ADVANCEMENT_LAYER = 14
RANK_COORD_LAYER = 14
WHITE_PAWN_PROGRESS_LAYER = 15
BLACK_PAWN_PROGRESS_LAYER = 16
TENSOR_SHAPE = (8, 8, 17)

# Policy head
POLICY_SIZE = 4096       # legacy flat from_sq(64) * to_sq(64) encoding
# Optional LC0-style extension.  A source/destination action cannot distinguish
# q/r/b/n promotions.  There are two promotion ranks, eight source files,
# three destination-file offsets and four promoted pieces: 2*8*3*4 = 192.
PROMOTION_POLICY_SIZE = 192
PROMOTION_AWARE_POLICY_SIZE = POLICY_SIZE + PROMOTION_POLICY_SIZE
C_PUCT = 1.5             # PUCT exploration constant (replaces UCB1 C)
FPU_REDUCTION = 0.30     # First-Play Urgency reduction for unvisited PUCT children
POLICY_TEMPERATURE = 1.0 # scale policy logits before legal-move softmax
DIRICHLET_ALPHA = 0.3    # Dirichlet noise concentration parameter
DIRICHLET_EPSILON = 0.25 # fraction of noise mixed into root priors
POLICY_TARGET_PSEUDOCOUNT = 0.0  # policy-target smoothing as a FRACTION of total root
                                 # visits, spread uniformly (0 = raw visit-count
                                 # targets, the AlphaZero default). See mcts.get_best_action.
POLICY_LOSS_WEIGHT = 1.0 # weight of policy CE loss relative to value MSE
POLICY_HEAD_CHANNELS = 32  # policy head bottleneck channels (widened from 16 for fresh start)
POLICY_HEAD_TYPE = "dense"  # "dense" or compact source-to-destination "attention"
POLICY_ATTENTION_CHANNELS = 32
SIDE_POLICY_ADAPTERS = False  # opt-in White/Black residual attention projection
STEM_CHANNELS = 64
RESIDUAL_BLOCK_CHANNELS = (
    64, 64, 128, 128, 128, 128, 128, 128
)  # deeper tower (stage F backbone scaling)
# Value head geometry. The default head starts with global average pooling, so
# it sees only per-channel means: measured effective rank of that vector is ~6
# on both live checkpoints, against ~481 for the policy head's spatial input
# (2026-07-25 audit). MCTS searches on the value signal, and the pawn phase --
# where Black's whole deficit lives -- is exactly where board geometry matters
# most. SPATIAL_VALUE_HEAD swaps in an AlphaZero-style head that keeps the 8x8
# layout (conv 1x1 -> flatten -> FC). Costs ~529K params (~5% of the net).
# Kept OFF by default: this is an A/B candidate, not a promoted change.
SPATIAL_VALUE_HEAD = False
VALUE_HEAD_CONV_CHANNELS = 32  # 1x1 bottleneck width for the spatial value head

USE_SE_BLOCKS = False     # optional squeeze-excitation in residual blocks
SE_REDUCTION = 16         # channel reduction ratio for SE bottleneck

# Optional LC0-style auxiliary target: predict remaining recorded decisions.
# Search can consume it through a bounded utility, but both the head and its
# search use remain opt-in so old recipes and checkpoints behave identically.
USE_MOVES_LEFT_HEAD = False
MOVES_LEFT_HEAD_CHANNELS = 64
MOVES_LEFT_LOSS_WEIGHT = 0.01
MOVES_LEFT_MAX_EFFECT = 0.03
MOVES_LEFT_THRESHOLD = 0.80
MOVES_LEFT_SLOPE = 0.10

# Training
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.1
VALUE_LOSS_EXPONENT = 2.5  # power-law loss (Stockfish uses 2.5 vs MSE's 2.0)
LR_GAMMA = 0.95           # exponential LR decay per epoch
EPOCHS = 50
VALUE_TARGET = "game_result"  # "game_result" or "mcts_value". Grounded on outcomes
                              # (REWORK_PLAN.md Phase 2.1): the value target must carry
                              # information the net does not already have.
VALUE_TARGET_HORIZON = 10   # plies from game end inside which the game_result
                            # target ramps FLOOR -> 1.0 (data_processor). Purpose
                            # is narrow: a near-mate tiebreak so search prefers
                            # mate-in-1 over mate-in-3 instead of drifting/
                            # suiciding at saturation. Positions further than the
                            # horizon all get the same flat FLOOR factor — no
                            # side bias, no other behavior change (the global
                            # v13 discount taxed Black's long wins; rejected).
VALUE_TARGET_FLOOR = 0.97   # plateau factor beyond the horizon (1.0 = off)
VALUE_TARGET_DISCOUNT_MODE = "near_mate"  # "near_mate" or full-game "progress"
VALUE_HEAD_MODE = "scalar"  # "scalar" or "wdl"
WDL_LOSS_WEIGHT = 0.5       # auxiliary CE weight when VALUE_HEAD_MODE="wdl"
WDL_DRAW_EPSILON = 0.05     # |target| <= eps is treated as draw for WDL labels
MODEL_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "models")

# The incumbent checkpoint. iterate.py used to promote to (and play.py default
# to) models/best_value_net.pt, which has never existed — so play.py silently
# fell back to the heuristic and a fresh iterate.py run would believe there was
# no incumbent at all. Both now resolve the real one through this constant.
INCUMBENT_MODEL = os.path.join(MODEL_DIR, "fresh_start_v22", "best_value_net.pt")

# Data retention (data_processor.py)
DATA_RETENTION_MAX_GENERATION_AGE = 32  # drop nn_gen* older than this many generations behind latest (<=0 disables)
DATA_RETENTION_MIN_NONHUMAN_PLIES = 4   # drop non-human games shorter than this many plies (<=0 disables)
OPPONENT_SIMULATIONS = 200  # MCTS sims for frozen opponent in alternating training
SKIP_CHECK_POSITIONS = True  # drop in-check positions during data generation by default

# Self-improvement loop (iterate.py)
ITERATE_GAMES = 400              # self-play games per generation
ITERATE_BLACKFOCUS_GAMES = 150   # black-focus games per generation (0 disables)
ITERATE_SIMS = 400               # generation MCTS simulations
ITERATE_ARENA_GAMES = 100        # candidate-vs-incumbent gate games
ITERATE_ARENA_SIMS = 400         # arena + anchor simulations
ITERATE_GATE_THRESHOLD = 0.55    # min candidate score vs incumbent
ITERATE_ANCHOR_GAMES = 20        # anchor benchmark games per candidate
ITERATE_ANCHOR_EPSILON = 0.05    # allowed anchor-score regression
ITERATE_MAX_GENERATION_AGE = 4   # processing window in generations
ITERATE_EPOCHS = 12              # V20 fine-tuning ceiling; early epochs dominate

# White-king aggression (heuristic eval, adopted 2026-07-07). 1.0/1.0 = the
# original GA-tuned baseline; these values scale the White-king terms:
#   KING_GEOM_SCALE   weights the pure-geometry confinement penalties
#                     (displacement + edge proximity) — the GA baseline over-
#                     penalized a safe, advancing king (real confinement is
#                     already caught by the adjacent-attacked + mobility terms).
#   KING_ATTACK_SCALE weights the king's attacking tropism rewards.
# Validated same-side paired vs v8-Black: White 0.44 -> 0.875 (+0.44), Black
# conversion unchanged. See tools/heuristic_ab.py.
KING_GEOM_SCALE = 0.4
KING_ATTACK_SCALE = 1.8

# Sub-goal reward shaping (Black strategic progress in heuristic eval)
WHITE_PAWN_VALUE = 0.18             # value per White pawn (was 0.10)
PAWN_ELIMINATION_BONUS = 0.14       # bonus per eliminated White pawn (4 - count)
BLOCKED_PAWN_PENALTY = 0.12         # penalty per White pawn that can't advance
KING_DISPLACEMENT_WEIGHT = 0.06     # reward per unit of king displacement from center
KING_MOBILITY_WEIGHT = 0.01         # penalty per restricted square (2-move reachability)
BARRIER_RANK_FILE_WEIGHT = 0.12     # bonus per barrier rank/file between king and edge
PIECE_SAFETY_BONUS = 0.08           # bonus per Black heavy piece at safe distance (>= 3) from White king
BLACK_KING_EXPOSURE_PENALTY = 0.04  # penalty per square adjacent to Black king attacked by White

# Default worker count for anything that plays games in a process pool.
#
# NOT os.cpu_count() and not cpu_count()-2. On the 5060 Ti box every worker
# builds its own CUDA context, and 14 of them dies at init with
# "fatal : Memory allocation failure", leaving orphaned ~1.4 GB processes
# behind -- the zombie-worker pattern that has cost this project runs before.
# Measured aggregate throughput at 400 sims (2026-08-01) plateaus long before
# the crash anyway: 4 workers 5.39 decisions/s, 8 workers 7.11, 12 workers
# 7.38. Eight buys the whole win and leaves the box usable while it runs.
#
# Every --workers flag still overrides this; it is the default that matters,
# because the default is what an unattended overnight driver uses.
DEFAULT_GAME_WORKERS = 8

# File paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
