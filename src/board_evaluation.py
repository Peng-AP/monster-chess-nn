import chess
import random


piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def evaluate_board(board):
    eval = 0
    # Evaluate Black's material
    for piece_type in piece_values:
        eval -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    # Evaluate White's material (scaled higher because they are more valuable)
    white_pawn_value = piece_values[chess.PAWN] * 10
    eval += len(board.pieces(chess.PAWN, chess.WHITE)) * white_pawn_value + (20000 if board.king(chess.WHITE) != None else 0)
    return eval

