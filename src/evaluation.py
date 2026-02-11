import chess


def evaluate(game_state, max_depth=50):
    """Evaluate a position using endgame knowledge + random rollouts.

    Key insight for Monster Chess: White can only win by promoting pawns.
    If White has no pawns (and no queen from promotion), Black wins.
    This lets us short-circuit obvious endgames instead of relying on
    random rollouts to stumble into checkmate.

    Returns a float in [-1, 1] from White's perspective.
    Designed to be swappable with a neural-network evaluation later.
    """
    board = game_state.board

    # Terminal states
    if board.king(chess.WHITE) is None:
        return -1.0
    if board.king(chess.BLACK) is None:
        return 1.0

    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))

    # White has no pawns and no queen — White cannot win
    if white_pawns == 0 and white_queens == 0:
        return -0.95

    # Fewer pawns = worse for White (each pawn lost is a big deal
    # when you only start with 4)
    pawn_factor = white_pawns / 4.0  # 1.0 = all pawns, 0.0 = none

    # Check pawn advancement — pawns close to promotion are very valuable
    advancement_bonus = 0.0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        rank = chess.square_rank(sq)
        if rank >= 6:  # 7th or 8th rank
            advancement_bonus += 0.15
        elif rank >= 4:  # 5th-6th rank
            advancement_bonus += 0.05

    # Queen on board = a pawn already promoted, very strong for White
    queen_bonus = white_queens * 0.3

    # Blend: use material assessment as a base, rollout for dynamics
    material_eval = -0.3 + pawn_factor * 0.6 + advancement_bonus + queen_bonus
    material_eval = max(-1.0, min(1.0, material_eval))

    # Random rollout for tactical assessment
    state = game_state.clone()
    for _ in range(max_depth):
        if state.is_terminal():
            break
        if not state.apply_random_action():
            break
    rollout_result = float(state.get_result())

    # Blend material knowledge with rollout (70/30)
    return 0.7 * material_eval + 0.3 * rollout_result
