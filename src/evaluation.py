import chess
import random

# Piece values normalised so that total material ≈ 1.0
# Standard total material (one side): P*8 + N*2 + B*2 + R*2 + Q = 3900 centipawns
_RAW_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
_MAX_MATERIAL = 3900  # approximate max one-side material (no king)


def evaluate(game_state):
    """Heuristic evaluation returning a value in [-1, 1] from White's perspective.

    Designed to be swappable with a neural-network evaluation later.
    """
    board = game_state.board

    if board.king(chess.WHITE) is None:
        return -1.0
    if board.king(chess.BLACK) is None:
        return 1.0

    score = 0.0

    # --- Material ---
    for piece_type, value in _RAW_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    # --- White pawn advancement (critical in Monster Chess) ---
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        rank = chess.square_rank(sq)  # 0-7
        # Bonus grows sharply as pawns approach promotion
        score += (rank - 1) * 150  # rank 1 = starting, rank 6 = one away

    # --- Promotion bonus: White queen presence ---
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    score += white_queens * 1500

    # --- King advancement for White (aggressive king is good) ---
    wk = board.king(chess.WHITE)
    if wk is not None:
        score += chess.square_rank(wk) * 50

    # Normalise into [-1, 1] with a tanh-like squash
    # Divisor chosen so that a full-piece advantage ≈ 0.7
    import math
    return math.tanh(score / 2000.0)


def random_rollout(game_state, max_depth=30):
    """Play random moves to estimate the position value.

    Returns +1, -1, or 0.
    """
    state = game_state.clone()
    for _ in range(max_depth):
        if state.is_terminal():
            break
        actions = state.get_legal_actions()
        if not actions:
            break
        action = random.choice(actions)
        state.apply_action(action)

    return state.get_result()


def hybrid_evaluate(game_state, rollout_weight=0.3, rollout_depth=20):
    """Combine heuristic evaluation with a random rollout.

    Returns a value in [-1, 1].
    """
    heuristic = evaluate(game_state)
    rollout = random_rollout(game_state, max_depth=rollout_depth)
    return (1 - rollout_weight) * heuristic + rollout_weight * rollout
