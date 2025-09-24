# src/monster_chess_engine/move_selector.py

import chess
import random
from .board_evaluation import evaluate_board

def choose_white_move(board):
    first_moves = list(board.pseudo_legal_moves)
    if not first_moves: return None
    bestPair, bestEval = (None, None), -float('inf')
    for move in first_moves:
        board.push(move)
        board.turn = chess.WHITE
        second_moves = list(board.pseudo_legal_moves)
        for second in second_moves:
            board.push(second)
            eval = evaluate_board(board)

            if(eval > bestEval or (eval == bestEval and random.choice([True, False]))):
                bestPair = (move, second)
            bestEval = max(eval, bestEval)

            board.pop()

        board.pop()
        board.turn = chess.WHITE
    return bestPair


def choose_black_move(board):
    moves = list(board.legal_moves)
    if not moves: return None
    best, bestEval = None, float('inf')
    for move in moves:
        board.push(move)
        eval = evaluate_board(board)
        if(eval < bestEval or (eval == bestEval and random.choice([True, False]))):
            best = move
        bestEval = min(eval, bestEval)
        board.pop()
    return best



def choose_move(board, color):
    if color == chess.WHITE:
        return choose_white_move(board)
    else:
        return choose_black_move(board)