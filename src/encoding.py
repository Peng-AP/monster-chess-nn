"""Board/move encoding shared by data processing, evaluation, and MCTS.

Split out of data_processor.py (REWORK_PLAN Phase 5 §8.2).
"""
import chess
import numpy as np

from config import (
    TENSOR_SHAPE, TURN_LAYER,
    MOVE_COUNT_LAYER,
    LEGACY_TENSOR_CHANNELS, LEGACY_PAWN_ADVANCEMENT_LAYER,
    RANK_COORD_LAYER, WHITE_PAWN_PROGRESS_LAYER, BLACK_PAWN_PROGRESS_LAYER,
    POLICY_SIZE, PROMOTION_POLICY_SIZE, PROMOTION_AWARE_POLICY_SIZE,
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


def fen_to_tensor(fen, is_white_turn=True, half_pending=False,
                  input_channels=None):
    """Convert a FEN string to the requested checkpoint encoding.

    Current 17-channel layers:
      0-11: piece positions (binary)
      12:   turn indicator (+1 White, -1 Black)
      13:   half-move indicator (1.0 on White's SECOND half-move, else 0.0)
      14:   signed rank coordinate (-1.0 rank 1, +1.0 rank 8)
      15:   White-pawn progress toward rank 8
      16:   Black-pawn progress toward rank 1

    Legacy 15-channel checkpoints retain their original White-only channel 14.
    Selecting by channel count prevents an incompatible encoding from loading
    silently while keeping v16/v17 available for matches and human play.
    """
    board = chess.Board(fen)
    channels = TENSOR_SHAPE[2] if input_channels is None else int(input_channels)
    if channels not in (LEGACY_TENSOR_CHANNELS, TENSOR_SHAPE[2]):
        raise ValueError(f"Unsupported position encoding with {channels} channels")
    tensor = np.zeros((8, 8, channels), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            layer = PIECE_TO_LAYER[(piece.piece_type, piece.color)]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[rank, file, layer] = 1.0

    # Turn indicator
    tensor[:, :, TURN_LAYER] = 1.0 if is_white_turn else -1.0

    # Half-move indicator: 1.0 when this is the second of White's two moves
    # (white_half_pending), so the network can value/route the two halves
    # differently (REWORK_PLAN.md Phase 3.2).  0.0 for Black and for White's first
    # half.  Legacy records without a half flag decode as 0.0 (backward compatible).
    tensor[:, :, MOVE_COUNT_LAYER] = 1.0 if half_pending else 0.0

    if channels == LEGACY_TENSOR_CHANNELS:
        # Exact v16/v17 layout.
        for sq in board.pieces(chess.PAWN, chess.WHITE):
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            tensor[rank, file, LEGACY_PAWN_ADVANCEMENT_LAYER] = (rank - 1) / 6.0
        return tensor

    # Translation-aware rank position for every piece, then equal promotion
    # progress features for both pawn colors. These are representation facts,
    # not evaluation rules; the network learns what they mean from games.
    for rank in range(8):
        tensor[rank, :, RANK_COORD_LAYER] = (rank - 3.5) / 3.5
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        tensor[rank, file, WHITE_PAWN_PROGRESS_LAYER] = np.clip(
            (rank - 1) / 6.0, 0.0, 1.0)
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        tensor[rank, file, BLACK_PAWN_PROGRESS_LAYER] = np.clip(
            (6 - rank) / 6.0, 0.0, 1.0)

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
# Policy encoding
#   0..4095: source * 64 + destination (legacy ABI)
#   4096..4287: color/rank * source-file * direction * promotion piece
# ------------------------------------------------------------------

_PROMOTION_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
_PROMOTION_PIECE_TO_OFFSET = {
    piece: offset for offset, piece in enumerate(_PROMOTION_PIECES)
}

def move_to_index(move):
    """Convert a move to the legacy source/destination policy index."""
    return move.from_square * 64 + move.to_square


def promotion_move_to_index(move):
    """Return the distinct promotion-policy index for a promotion move.

    The two color bands are identified by the pawn's source rank.  Requiring a
    real promotion geometry makes malformed UCI suffixes fail loudly instead
    of silently entering an unrelated target cell.
    """
    if move.promotion not in _PROMOTION_PIECE_TO_OFFSET:
        raise ValueError("move is not a q/r/b/n promotion")
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    if (from_rank, to_rank) == (6, 7):
        color_band = 0
    elif (from_rank, to_rank) == (1, 0):
        color_band = 1
    else:
        raise ValueError("promotion move does not cross a promotion rank")
    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    direction = to_file - from_file
    if direction not in (-1, 0, 1):
        raise ValueError("promotion destination must be on an adjacent file")
    direction_offset = direction + 1
    piece_offset = _PROMOTION_PIECE_TO_OFFSET[move.promotion]
    return (POLICY_SIZE + color_band * 96 + from_file * 12
            + direction_offset * 4 + piece_offset)


def move_to_policy_index(move, promotion_aware=False):
    """Policy index for either the legacy or promotion-aware ABI."""
    if promotion_aware and move.promotion is not None:
        return promotion_move_to_index(move)
    return move_to_index(move)


def mirror_move_index(idx):
    """Mirror either policy ABI index across the file axis (a<->h)."""
    if idx >= POLICY_SIZE:
        relative = idx - POLICY_SIZE
        if relative >= PROMOTION_POLICY_SIZE:
            raise ValueError(f"policy index out of range: {idx}")
        color_band, relative = divmod(relative, 96)
        from_file, relative = divmod(relative, 12)
        direction_offset, piece_offset = divmod(relative, 4)
        mirrored_file = 7 - from_file
        mirrored_direction = 2 - direction_offset
        return (POLICY_SIZE + color_band * 96 + mirrored_file * 12
                + mirrored_direction * 4 + piece_offset)
    from_sq = idx // 64
    to_sq = idx % 64
    from_file, from_rank = from_sq % 8, from_sq // 8
    to_file, to_rank = to_sq % 8, to_sq // 8
    new_from = from_rank * 8 + (7 - from_file)
    new_to = to_rank * 8 + (7 - to_file)
    return new_from * 64 + new_to


def policy_dict_to_target(policy_dict, is_white, promotion_aware=False):
    """Convert an MCTS action_probs dict to a dense policy target vector.

    For Black: each key is a UCI move string -> index directly.
    For White: each key is "m1_uci,m2_uci". We marginalize over m2 to
    get P(m1), since the policy head predicts single moves and m2 is
    evaluated from the post-m1 board state during MCTS.
    """
    policy_size = PROMOTION_AWARE_POLICY_SIZE if promotion_aware else POLICY_SIZE
    target = np.zeros(policy_size, dtype=np.float32)
    if policy_dict is None:
        return target  # uniform-ish fallback (all zeros, masked later)

    for action_str, prob in policy_dict.items():
        if is_white:
            m1_uci = action_str.split(",")[0]
            move = chess.Move.from_uci(m1_uci)
        else:
            move = chess.Move.from_uci(action_str)
        target[move_to_policy_index(move, promotion_aware)] += prob

    # Renormalize (White's m1 marginal should already sum to ~1)
    total = target.sum()
    if total > 0:
        target /= total
    return target


# Precomputed once: index i of a policy vector maps to _MIRROR_PERM[i] in the
# mirrored vector. Replaces a 4096-iteration Python loop that ran per augmented
# record — a real cost on multi-GB corpora, since every record is mirrored.
# mirror_move_index stays as the definition (and for external callers).
_MIRROR_PERM = np.array([mirror_move_index(i) for i in range(POLICY_SIZE)],
                        dtype=np.intp)
_PROMOTION_MIRROR_PERM = np.array(
    [mirror_move_index(i) for i in range(PROMOTION_AWARE_POLICY_SIZE)],
    dtype=np.intp,
)


def mirror_policy(policy_vec):
    """Mirror a dense policy vector across the file axis."""
    if policy_vec.shape[-1] == POLICY_SIZE:
        permutation = _MIRROR_PERM
    elif policy_vec.shape[-1] == PROMOTION_AWARE_POLICY_SIZE:
        permutation = _PROMOTION_MIRROR_PERM
    else:
        raise ValueError(
            f"unsupported policy width {policy_vec.shape[-1]}; expected "
            f"{POLICY_SIZE} or {PROMOTION_AWARE_POLICY_SIZE}")
    mirrored = np.zeros_like(policy_vec)
    mirrored[permutation] = policy_vec
    return mirrored
