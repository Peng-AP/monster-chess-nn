# src/monster_chess_engine/move_selector.py

import chess
import random
from board_evaluation import evaluate_board

def choose_white(board):
    first_moves = list(board.pseudo_legal_moves)
    if not first_moves: return None
    pairs = []
    for move in first_moves:
        board.push(move)
        if(board.king(chess.BLACK) == None): 
            board.pop()
            return [(move, chess.Move.null())]
        board.turn = chess.WHITE
        second_moves = list(board.pseudo_legal_moves)
        for second in second_moves:
            board.push(second)
            if(board.king(chess.BLACK) == None): 
                board.pop()
                board.turn = chess.BLACK
                board.pop()
                return [(move, second)]
            board.pop()
            pairs.append((move, second))
        board.pop()
        board.turn = chess.WHITE
    return pairs


def choose_black(board):
    return list(board.pseudo_legal_moves)

def minimax(board, depth, color):
    #print(board.fen(), f"depth: {depth}", "white" if color == chess.WHITE else 'black')

    if(depth == 0 or board.king(chess.WHITE) == None or board.king(chess.BLACK) == None):
        return evaluate_board(board)
    
    if(color == chess.WHITE):
        max_eval = -float('inf')
        valid_moves = choose_white(board)
        if(not valid_moves): 
            print('bing')
            return 0
        else:
            for pair in valid_moves:
                board.push(pair[0])
                if(board.king(chess.BLACK) == None): 
                    board.pop()
                    return float('inf')
                board.turn = chess.WHITE
                board.push(pair[1])

                eval = minimax(board, depth - 1, chess.BLACK)
                max_eval = max(eval, max_eval)

                board.pop()
                board.turn = chess.BLACK
                board.pop()
            
            return max_eval
        
    if(color == chess.BLACK):
        min_eval = float('inf')
        valid_moves = choose_black(board)
        if(not valid_moves): return 0
        for move in valid_moves:
            board.push(move)
            eval = minimax(board, depth - 1, chess.WHITE)
            min_eval = min(eval, min_eval)

            board.pop()
        
        return min_eval
        
        


def choose_move(board, color):
    depth = 3
    if(color == chess.WHITE):
        best_pair = None
        max_eval = -float('inf')
        valid_double_moves = choose_white(board)
        for pair in valid_double_moves:
            board.push(pair[0])
            if(board.king(chess.BLACK) == None): return (pair[0], chess.Move.null())
            board.turn = chess.WHITE
            board.push(pair[1])

            eval = minimax(board, depth - 1, chess.BLACK)
            board.pop()
            board.turn = chess.BLACK
            board.pop()

            if(eval > max_eval or (eval == max_eval and random.choice([True, False]))):
                max_eval = eval
                best_pair = pair

        return best_pair
    elif(color == chess.BLACK):
        best = None
        min_eval = float('inf')
        valid_moves = choose_black(board)
        for move in valid_moves:
            board.push(move)
            eval = minimax(board, depth - 1, chess.WHITE)
            board.pop()

            if(eval < min_eval or (eval == min_eval and random.choice([True, False]))):
                min_eval = eval
                best = move
            
        return best
    else: print("SOMETHINGS WRONG")