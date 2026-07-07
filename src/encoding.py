"""Board/move encoding shared by data processing, evaluation, and MCTS.

Split out of data_processor.py (REWORK_PLAN Phase 5 §8.2).
"""
import chess
import numpy as np

from config import (
    TENSOR_SHAPE, TURN_LAYER,
    MOVE_COUNT_LAYER, PAWN_ADVANCEMENT_LAYER,
    POLICY_SIZE,
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


def fen_to_tensor(fen, is_white_turn=True, half_pending=False):
    """Convert a FEN string to an (8, 8, 15) tensor.

    Layers:
      0-11: piece positions (binary)
      12:   turn indicator (+1 White, -1 Black)
      13:   half-move indicator (1.0 on White's SECOND half-move, else 0.0)
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

    # Half-move indicator: 1.0 when this is the second of White's two moves
    # (white_half_pending), so the network can value/route the two halves
    # differently (REWORK_PLAN.md Phase 3.2).  0.0 for Black and for White's first
    # half.  Legacy records without a half flag decode as 0.0 (backward compatible).
    tensor[:, :, MOVE_COUNT_LAYER] = 1.0 if half_pending else 0.0

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
