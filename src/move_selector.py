import chess
import random
from .board_evaluation import evaluate_board

# This function chooses a move for the current player
def choose_white_move(board):
    legal_moves1 = list(board.legal_moves)
    if not legal_moves1:
        return None

    best_first_move = None
    # White wants to MINIMIZE the evaluation score
    best_final_eval = float('inf')

    # Iterate through all possible first moves
    for move1 in legal_moves1:
        board.push(move1)
        
        # Now, find the best possible second move from this new position
        legal_moves2 = list(board.legal_moves)
        if not legal_moves2:
            # If the game ends after one move, that's our final evaluation
            eval_after_2_plies = evaluate_board(board)
        else:
            best_eval_after_move2 = float('inf')
            for move2 in legal_moves2:
                board.push(move2)
                current_eval = evaluate_board(board)
                board.pop() # Undo move2
                # White seeks the best (lowest) score for itself
                if current_eval < best_eval_after_move2:
                    best_eval_after_move2 = current_eval
            eval_after_2_plies = best_eval_after_move2

        board.pop() # Undo move1

        # Check if this sequence of two moves is the best we've seen so far
        if eval_after_2_plies < best_final_eval:
            best_final_eval = eval_after_2_plies
            best_first_move = move1

    return best_first_move or random.choice(legal_moves1)

def choose_black_move(board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_move = None
    # Black wants to MAXIMIZE the evaluation score
    best_eval = -float('inf')

    for move in legal_moves:
        board.push(move)
        current_eval = evaluate_board(board)
        board.pop()

        if current_eval > best_eval:
            best_eval = current_eval
            best_move = move

    return best_move or random.choice(legal_moves)

def choose_move(board):
    if board.turn == chess.WHITE:
        return choose_white_move(board)
    else:
        return choose_black_move(board)