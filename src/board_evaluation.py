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

def experimental_eval(board):
    eval = 0
    if(board.king(chess.BLACK) == None): return 999999
    if(board.king(chess.WHITE) == None): return -999999

    for piece_type in piece_values:
        if(piece_type == chess.KING):
            black_rank = chess.square_rank(board.king(chess.BLACK))
            white_rank = chess.square_rank(board.king(chess.WHITE))
            eval -= ((20000 if board.king(chess.BLACK) != None else 0) - (7-black_rank) * 5000)
            eval += ((20000 if board.king(chess.WHITE) != None else 0) + (white_rank) * 1000)
        else:
            eval -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
            white_pawns = board.pieces(chess.PAWN, chess.WHITE)
            for pawn in white_pawns:
                rank = chess.square_rank(pawn)
                eval += (rank - 1) * 1000/6
            if(board.pieces(chess.QUEEN, chess.WHITE)):
                eval += 30000

    return eval

#print(chess.square_rank(chess.Board("rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1").king(chess.WHITE)))