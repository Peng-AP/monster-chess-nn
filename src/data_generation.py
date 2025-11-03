import chess
from move_selector import choose_move

def outcome(board):
    if(board.king(chess.WHITE) == None): return -1
    elif(board.king(chess.BLACK) == None): return 1
    return 0

def play_game():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1")
    game_positions = []
    
    max_turns = 100 # A turn is one double-move by White or one move by Black
    turn_count = 0
    res = 0 # Default to draw

    # We must track whose turn it is ourselves, as board.turn becomes unreliable
    is_white_turn = True

    while turn_count < max_turns:
        print(f"Turn {turn_count+1}")
        if is_white_turn:
            # --- WHITE'S DOUBLE MOVE ---
            game_positions.append(board.fen())
            
            # First move
            moves = choose_move(board, chess.WHITE)

            if moves == None: break
            board.push(moves[0])
            res = outcome(board)
            if(res != 0): break
            board.turn = chess.WHITE

            # Second move
            board.push(moves[1])
            res = outcome(board)
            if(res != 0): break
        
        else: # It is Black's turn
            game_positions.append(board.fen())
            
            move = choose_move(board, chess.BLACK)
            if not move: break
            board.push(move)
            res = outcome(board)
            if(res != 0): break
        
        # Flip the turn for our custom game loop
        is_white_turn = not is_white_turn
        turn_count += 1

    return [(fen, res) for fen in game_positions]



# --- Main data generation block ---
if __name__ == "__main__":
    NUM_GAMES = 1
    all_game_data = []

    print(f"Generating data from {NUM_GAMES} games...")
    for i in range(NUM_GAMES):
        if (i + 1) % 10 == 0:
            print(f"  ... playing game {i + 1}/{NUM_GAMES}")
        game_data = play_game()
        all_game_data.extend(game_data)

    # Save the data to a file
    with open("data/raw/monster_chess_data.csv", "w") as f:
        f.write("fen,outcome\n") # Header
        for fen, result in all_game_data:
            f.write(f'"{fen}",{result}\n')

    print(f"\nDone! Saved {len(all_game_data)} positions to data/raw/monster_chess_data.csv")