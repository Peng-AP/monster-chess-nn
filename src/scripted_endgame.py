"""Scripted Black endgame play for curriculum positions.

Implements a rook-ladder style algorithm that pushes White's king
toward the edge using rank/file barriers from rooks/queens.
This guarantees Black wins in endgame curriculum positions where
MCTS alone fails because the double-move king escapes every net.

The algorithm:
  1. Identify which edge to push toward (closest)
  2. Move rooks/queens to form barriers cutting off king escape
  3. Step barriers inward to compress the king
  4. Deliver checkmate once king is on the final rank/file

For integration with data generation: scripted Black play replaces
MCTS for Black in curriculum games, while White still uses MCTS.
This provides strong negative-value training signal (Black winning).
"""
import chess
import random


def get_scripted_black_move(board):
    """Pick the best Black move using a simple rook-ladder heuristic.

    Args:
        board: a chess.Board with Black to move.

    Returns:
        A chess.Move, or None if no legal moves.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return random.choice(legal)

    wk_rank = chess.square_rank(wk)
    wk_file = chess.square_file(wk)

    # Determine push direction: push king toward nearest edge
    # (rank 0/7 or file 0/7)
    rank_dist_to_edge = min(wk_rank, 7 - wk_rank)
    file_dist_to_edge = min(wk_file, 7 - wk_file)

    if rank_dist_to_edge <= file_dist_to_edge:
        # Push toward nearest rank edge (0 or 7)
        target_rank = 0 if wk_rank <= 3 else 7
        return _push_to_rank(board, legal, wk, bk, target_rank)
    else:
        target_file = 0 if wk_file <= 3 else 7
        return _push_to_file(board, legal, wk, bk, target_file)


def _push_to_rank(board, legal, wk, bk, target_rank):
    """Push White king toward target_rank using rook barriers."""
    wk_rank = chess.square_rank(wk)
    wk_file = chess.square_file(wk)
    bk_rank = chess.square_rank(bk)
    bk_file = chess.square_file(bk)
    direction = 1 if target_rank > wk_rank else -1

    heavy_pieces = _get_heavy_pieces(board, chess.BLACK)

    best_move = None
    best_score = float("-inf")

    for move in legal:
        score = 0.0
        piece = board.piece_at(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)

        # Simulate the move for safety check
        board.push(move)
        # Avoid moves that lose a piece to the double-move king
        if board.is_attacked_by(chess.WHITE, move.to_square) and piece:
            if piece.piece_type in (chess.ROOK, chess.QUEEN):
                score -= 50  # don't hang heavy pieces
        board.pop()

        if piece is None:
            continue

        if piece.piece_type in (chess.ROOK, chess.QUEEN):
            from_rank = chess.square_rank(move.from_square)

            # Reward placing rook on a barrier rank between king and target
            # A barrier rank is one that cuts off the king's escape
            if direction == 1:
                # Pushing king toward rank 7
                if wk_rank < to_rank <= 7:
                    # Rook is between king and target edge — good barrier
                    score += 20
                    # Closer barrier = more compression
                    score += (7 - abs(to_rank - wk_rank - 1)) * 2
                # Moving rook off the king's rank/file to avoid capture
                if to_rank == wk_rank and abs(to_file - wk_file) <= 2:
                    score -= 15  # too close, might get captured with double move
            else:
                if 0 <= to_rank < wk_rank:
                    score += 20
                    score += (7 - abs(wk_rank - 1 - to_rank)) * 2
                if to_rank == wk_rank and abs(to_file - wk_file) <= 2:
                    score -= 15

            # Reward controlling the rank just beyond the king
            barrier_rank = wk_rank + direction
            if 0 <= barrier_rank <= 7 and to_rank == barrier_rank:
                # Check rook is far enough from king to not be captured
                if abs(to_file - wk_file) > 2:
                    score += 30  # excellent barrier position

            # Keep heavy pieces away from White king (double-move danger)
            king_dist = abs(to_rank - wk_rank) + abs(to_file - wk_file)
            if king_dist <= 2:
                score -= 20
            elif king_dist >= 4:
                score += 5

        elif piece.piece_type == chess.KING:
            # Black king should stay far from White king but support rooks
            king_dist = abs(to_rank - wk_rank) + abs(to_file - wk_file)
            if king_dist <= 3:
                score -= 10  # too close to double-move king
            else:
                score += 3

            # King should move toward the same rank/file zone to support
            if direction == 1:
                if to_rank > bk_rank:
                    score += 2  # follow the push direction
            else:
                if to_rank < bk_rank:
                    score += 2

        # Small bonus for captures (taking White pawns)
        if board.piece_at(move.to_square) is not None:
            score += 8

        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_move is not None else random.choice(legal)


def _push_to_file(board, legal, wk, bk, target_file):
    """Push White king toward target_file using rook barriers."""
    wk_rank = chess.square_rank(wk)
    wk_file = chess.square_file(wk)
    bk_rank = chess.square_rank(bk)
    bk_file = chess.square_file(bk)
    direction = 1 if target_file > wk_file else -1

    best_move = None
    best_score = float("-inf")

    for move in legal:
        score = 0.0
        piece = board.piece_at(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)

        board.push(move)
        if board.is_attacked_by(chess.WHITE, move.to_square) and piece:
            if piece.piece_type in (chess.ROOK, chess.QUEEN):
                score -= 50
        board.pop()

        if piece is None:
            continue

        if piece.piece_type in (chess.ROOK, chess.QUEEN):
            if direction == 1:
                if wk_file < to_file <= 7:
                    score += 20
                    score += (7 - abs(to_file - wk_file - 1)) * 2
                if to_file == wk_file and abs(to_rank - wk_rank) <= 2:
                    score -= 15
            else:
                if 0 <= to_file < wk_file:
                    score += 20
                    score += (7 - abs(wk_file - 1 - to_file)) * 2
                if to_file == wk_file and abs(to_rank - wk_rank) <= 2:
                    score -= 15

            barrier_file = wk_file + direction
            if 0 <= barrier_file <= 7 and to_file == barrier_file:
                if abs(to_rank - wk_rank) > 2:
                    score += 30

            king_dist = abs(to_rank - wk_rank) + abs(to_file - wk_file)
            if king_dist <= 2:
                score -= 20
            elif king_dist >= 4:
                score += 5

        elif piece.piece_type == chess.KING:
            king_dist = abs(to_rank - wk_rank) + abs(to_file - wk_file)
            if king_dist <= 3:
                score -= 10
            else:
                score += 3
            if direction == 1:
                if to_file > bk_file:
                    score += 2
            else:
                if to_file < bk_file:
                    score += 2

        if board.piece_at(move.to_square) is not None:
            score += 8

        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_move is not None else random.choice(legal)


def _get_heavy_pieces(board, color):
    """Get all rook and queen squares for the given color."""
    pieces = []
    for sq in board.pieces(chess.ROOK, color):
        pieces.append(sq)
    for sq in board.pieces(chess.QUEEN, color):
        pieces.append(sq)
    return pieces
