import chess


def evaluate(game_state, max_depth=30):
    """Evaluate a Monster Chess position heuristically.

    Monster Chess specifics:
      - White king gets 2 moves/turn, so a lone king can still hunt
        and capture Black's king. White without pawns is NOT lost.
      - Black wins by promoting pawns into heavy pieces (Q/R) and
        building a mating barrier to push White's king to the edge.
      - White wins by advancing pawns to promote, or capturing Black's king
        directly with the double-move advantage.

    Returns a float in [-1, 1] from White's perspective.
    Designed to be swappable with a neural-network evaluation later.
    """
    board = game_state.board

    # Terminal
    if board.king(chess.WHITE) is None:
        return -1.0
    if board.king(chess.BLACK) is None:
        return 1.0

    score = 0.0

    # ---- White's chances ----

    # White pawn count (starts with 4)
    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    score += white_pawns * 0.12

    # White pawn advancement
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        rank = chess.square_rank(sq)
        score += (rank - 1) * 0.04  # rank 1=start, rank 6=one away

    # White queens (promoted pawns)
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    score += white_queens * 0.35

    # ---- Black's chances ----

    # Black heavy pieces (queens + rooks) — needed for mating net
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
    black_rooks = len(board.pieces(chess.ROOK, chess.BLACK))
    black_heavy = black_queens + black_rooks
    score -= black_heavy * 0.06

    # Black pawn advancement (pawns promote on rank 0 for Black)
    black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        rank = chess.square_rank(sq)
        score -= (6 - rank) * 0.02  # closer to rank 0 = closer to promotion

    # Black promoted queens beyond starting one
    if black_queens > 1:
        score -= (black_queens - 1) * 0.25  # extra queens = mating net material

    # ---- King confinement (helps Black's mating pattern) ----
    wk = board.king(chess.WHITE)
    if wk is not None:
        wk_rank = chess.square_rank(wk)
        wk_file = chess.square_file(wk)
        # Distance from center (0-3 scale, 3 = edge)
        rank_edge = min(wk_rank, 7 - wk_rank)
        file_edge = min(wk_file, 7 - wk_file)
        edge_dist = min(rank_edge, file_edge)
        # White king on edge is bad for White when Black has heavy pieces
        if black_heavy >= 3:
            score -= (3 - edge_dist) * 0.08

    # ---- Material balance (general) ----
    # Black minor pieces help control squares
    black_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    score -= (black_knights + black_bishops) * 0.03

    # Clamp and blend with a short rollout for tactical awareness
    heuristic = max(-0.95, min(0.95, score))

    # Short rollout (tactical check — catches immediate captures/mates)
    state = game_state.clone()
    for _ in range(max_depth):
        if state.is_terminal():
            break
        if not state.apply_random_action():
            break
    rollout_result = float(state.get_result())

    return 0.8 * heuristic + 0.2 * rollout_result
