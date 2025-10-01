import chess
from board_evaluation import evaluate_board
from move_selector import choose_move

board = chess.Board("1Q1k2nr/p1p2pp1/4b3/3pq2p/8/6K1/8/8 w k - 0 9")
print(board.pseudo_legal_moves)
print(choose_move(board, chess.WHITE))