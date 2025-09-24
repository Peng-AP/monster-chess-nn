# src/monster_chess_engine/game_logic.py

import chess
from .move_selector import choose_move

# New helper function to check if a king is missing
def is_king_captured(board):
    return not board.king(chess.WHITE) or not board.king(chess.BLACK)

def play_game():
    # Monster Chess starting position (custom FEN)
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1")
    game_positions = []
    
    # Add a turn limit to prevent infinitely long games
    max_turns = 200 # 100 moves for each player
    turn_count = 0

    outcome = 0 # Default to draw

    while turn_count < max_turns:
        game_positions.append(board.fen())

        move = choose_move(board)
        if not move: # No legal moves, likely a stalemate situation
            break 
        
        board.push(move)

        # --- CHECK FOR GAME OVER AFTER EVERY MOVE ---
        if is_king_captured(board):
            # The player who just moved is the winner
            outcome = 1 if board.turn == chess.BLACK else -1 # White moved, turn is now Black
            break

        # --- HANDLE WHITE'S SECOND MOVE ---
        if board.turn == chess.WHITE:
            game_positions.append(board.fen()) # Save position after 1st move
            move2 = choose_move(board)
            if not move2:
                break
            
            board.push(move2)

            if is_king_captured(board):
                outcome = 1 if board.turn == chess.BLACK else -1
                break
        
        turn_count += 1
    
    # If the loop finished without a king capture, it's a draw due to turn limit
    # The 'outcome' variable will still be 0 in this case.

    # Return a list of tuples: (position, final_outcome)
    return [(fen, outcome) for fen in game_positions]


# --- Main data generation block ---
if __name__ == "__main__":
    NUM_GAMES = 1000
    all_game_data = []

    print(f"Generating data from {NUM_GAMES} games...")
    for i in range(NUM_GAMES):
        if (i + 1) % 10 == 0:
            print(f"  ... playing game {i + 1}/{NUM_GAMES}")
        game_data = play_game()
        all_game_data.extend(game_data)

    # Save the data to a file
    with open("data/monster_chess_data.csv", "w") as f:
        f.write("fen,outcome\n") # Header
        for fen, outcome in all_game_data:
            f.write(f'"{fen}",{outcome}\n')

    print(f"\nDone! Saved {len(all_game_data)} positions to data/monster_chess_data.csv")