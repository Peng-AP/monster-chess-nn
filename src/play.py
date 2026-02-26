"""Play Monster Chess against the trained neural network.

Usage:
    python play.py                         # play as Black (default, recommended)
    python play.py --color white           # play as White (double moves)
    python play.py --model path/to/model   # use specific model
    python play.py --heuristic             # play against heuristic eval (no model)
    python play.py --sims 400              # MCTS simulations (higher = stronger)
"""
import argparse
import json
import os
import sys
import time

import chess

from config import MCTS_SIMULATIONS, MODEL_DIR, RAW_DATA_DIR
from monster_chess import MonsterChessGame
from mcts import MCTS

# Piece letters (uppercase = White, lowercase = Black)
PIECE_LETTERS = {
    (chess.PAWN, chess.WHITE): "P",
    (chess.KNIGHT, chess.WHITE): "N",
    (chess.BISHOP, chess.WHITE): "B",
    (chess.ROOK, chess.WHITE): "R",
    (chess.QUEEN, chess.WHITE): "Q",
    (chess.KING, chess.WHITE): "K",
    (chess.PAWN, chess.BLACK): "p",
    (chess.KNIGHT, chess.BLACK): "n",
    (chess.BISHOP, chess.BLACK): "b",
    (chess.ROOK, chess.BLACK): "r",
    (chess.QUEEN, chess.BLACK): "q",
    (chess.KING, chess.BLACK): "k",
}

# Square colors (ANSI true-color for reliable rendering)
BG_LIGHT = "\033[48;2;240;217;181m"   # warm light square
BG_DARK = "\033[48;2;181;136;99m"     # brown dark square
BG_HL_LIGHT = "\033[48;2;247;247;105m"  # highlight light
BG_HL_DARK = "\033[48;2;218;195;71m"    # highlight dark
FG_WHITE = "\033[1;38;2;255;255;255m"   # bold bright white
FG_BLACK = "\033[1;38;2;0;0;0m"         # bold black
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def render_board(board, last_move=None, flip=False):
    """Render the board with Unicode pieces and colors."""
    print()
    ranks = range(8) if flip else range(7, -1, -1)
    files = range(8)

    # Highlight squares from last move
    highlights = set()
    if last_move:
        if isinstance(last_move, tuple):
            m1, m2 = last_move
            highlights.add(m1.from_square)
            highlights.add(m1.to_square)
            if m2 != chess.Move.null():
                highlights.add(m2.from_square)
                highlights.add(m2.to_square)
        else:
            highlights.add(last_move.from_square)
            highlights.add(last_move.to_square)

    for rank in ranks:
        # Each rank prints two rows for height
        top_line = f"     "
        mid_line = f"  {rank + 1}  "
        for file in files:
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            is_light = (rank + file) % 2 == 1
            if sq in highlights:
                bg = BG_HL_LIGHT if is_light else BG_HL_DARK
            else:
                bg = BG_LIGHT if is_light else BG_DARK

            cell_w = "     "  # 5 chars wide
            if piece:
                fg = FG_WHITE if piece.color == chess.WHITE else FG_BLACK
                sym = PIECE_LETTERS[(piece.piece_type, piece.color)]
                mid_line += f"{bg}{fg}  {sym}  {RESET}"
            else:
                mid_line += f"{bg}{cell_w}{RESET}"
            top_line += f"{bg}{cell_w}{RESET}"
        print(top_line)
        print(mid_line)
    print(f"\n       a    b    c    d    e    f    g    h\n")


def format_eval(value):
    """Format evaluation as a bar + number."""
    bar_len = 20
    # Map [-1, 1] to [0, bar_len]
    pos = int((value + 1) / 2 * bar_len)
    pos = max(0, min(bar_len, pos))
    bar = "█" * pos + "░" * (bar_len - pos)
    if value > 0.3:
        label = f"{BOLD}+{value:.2f}{RESET} (White)"
    elif value < -0.3:
        label = f"{BOLD}{value:.2f}{RESET} (Black)"
    else:
        label = f"{value:.2f} (even)"
    return f"  [{bar}] {label}"


def parse_move(text, board):
    """Parse user input into a chess.Move. Accepts UCI (e2e4) or SAN (Nf3)."""
    text = text.strip()
    # Try UCI first
    try:
        move = chess.Move.from_uci(text)
        if move in board.pseudo_legal_moves or move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError):
        pass
    # Try SAN
    try:
        return board.parse_san(text)
    except (ValueError, chess.InvalidMoveError, chess.AmbiguousMoveError) as e:
        return None


def get_human_white_action(game):
    """Get a double-move from the human playing White."""
    board = game.board
    board.turn = chess.WHITE
    legal_first = list(board.pseudo_legal_moves)

    while True:
        text = input(f"  {BOLD}Your move 1/2:{RESET} ").strip()
        if text.lower() in ("quit", "resign", "q"):
            return "resign"
        m1 = parse_move(text, board)
        if m1 is None:
            print(f"  Invalid move. Legal moves: {', '.join(m.uci() for m in legal_first[:20])}")
            if len(legal_first) > 20:
                print(f"  ... and {len(legal_first) - 20} more")
            continue
        if m1 not in legal_first:
            print(f"  Not a legal move. Try again.")
            continue
        break

    # Apply first move temporarily to get second moves
    board.push(m1)
    if board.king(chess.BLACK) is None:
        board.pop()
        return (m1, chess.Move.null())

    render_board(board, last_move=m1)
    board.turn = chess.WHITE
    legal_second = list(board.pseudo_legal_moves)

    while True:
        text = input(f"  {BOLD}Your move 2/2:{RESET} ").strip()
        if text.lower() in ("quit", "resign", "q"):
            board.pop()
            return "resign"
        m2 = parse_move(text, board)
        if m2 is None:
            print(f"  Invalid move. Legal moves: {', '.join(m.uci() for m in legal_second[:20])}")
            if len(legal_second) > 20:
                print(f"  ... and {len(legal_second) - 20} more")
            continue
        if m2 not in legal_second:
            print(f"  Not a legal move. Try again.")
            continue
        # Check if the pair leaves White king safe
        board.push(m2)
        wk = board.king(chess.WHITE)
        if wk is not None and board.is_attacked_by(chess.BLACK, wk):
            # Check if ANY pair is safe — if not, allow it (forced blunder)
            board.pop()
            board.turn = chess.BLACK
            board.pop()
            safe_exists = any(True for _ in _safe_pairs_gen(game))
            board.push(m1)
            board.turn = chess.WHITE
            if safe_exists:
                print(f"  That leaves your king attacked. Pick a different move.")
                continue
            else:
                print(f"  {DIM}(forced — no safe moves exist){RESET}")
        else:
            board.pop()
        break

    board.turn = chess.BLACK
    board.pop()
    return (m1, m2)


def _safe_pairs_gen(game):
    """Yield one safe pair to check if any exist (short-circuit)."""
    clone = game.clone()
    actions = clone.get_legal_actions()
    for a in actions[:1]:
        yield a


def get_human_black_action(game):
    """Get a single move from the human playing Black."""
    legal = game.get_legal_actions()

    if not legal:
        return None

    while True:
        text = input(f"  {BOLD}Your move:{RESET} ").strip()
        if text.lower() in ("quit", "resign", "q"):
            return "resign"
        move = parse_move(text, game.board)
        if move is None or move not in legal:
            print(f"  Invalid. Legal moves: {', '.join(m.uci() for m in legal[:20])}")
            if len(legal) > 20:
                print(f"  ... and {len(legal) - 20} more")
            continue
        return move


def save_game_data(records, output_dir):
    """Save game records for training data."""
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".jsonl")])
    path = os.path.join(output_dir, f"game_{existing:05d}.jsonl")
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Play Monster Chess against the AI")
    parser.add_argument("--color", choices=["white", "black"], default="black",
                        help="Your color (default: black)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (default: models/best_value_net.pt)")
    parser.add_argument("--heuristic", action="store_true",
                        help="Use heuristic evaluation instead of NN")
    parser.add_argument("--sims", type=int, default=MCTS_SIMULATIONS,
                        help=f"MCTS simulations (default: {MCTS_SIMULATIONS})")
    parser.add_argument("--save-data", action="store_true",
                        help="Save game data for training")
    parser.add_argument("--fen", type=str, default=None,
                        help="Start from a custom FEN position")
    args = parser.parse_args()

    # Load evaluator
    eval_fn = None
    heuristic_eval = None
    if not args.heuristic:
        model_path = args.model or os.path.join(MODEL_DIR, "best_value_net.pt")
        if os.path.exists(model_path):
            from evaluation import NNEvaluator
            print(f"Loading model: {model_path}")
            eval_fn = NNEvaluator(model_path)
            print(f"  Model loaded (CUDA: {eval_fn.device.type == 'cuda'})")
        else:
            print(f"No model at {model_path} — falling back to heuristic eval")
    if args.save_data:
        from evaluation import evaluate as heuristic_evaluate
        heuristic_eval = heuristic_evaluate

    engine = MCTS(num_simulations=args.sims, eval_fn=eval_fn)
    human_is_white = args.color == "white"
    session_id = time.strftime("%Y%m%d_%H%M%S")
    ai_color = "black" if human_is_white else "white"
    model_used = args.model or os.path.join(MODEL_DIR, "best_value_net.pt")

    print(f"\n{'='*50}")
    print(f"  MONSTER CHESS")
    print(f"{'='*50}")
    print(f"  You play:    {'White (double moves)' if human_is_white else 'Black'}")
    print(f"  AI plays:    {'Black' if human_is_white else 'White (double moves)'}")
    print(f"  Evaluator:   {'Heuristic' if eval_fn is None else 'Neural Network'}")
    print(f"  Simulations: {args.sims}")
    print(f"{'='*50}")
    print(f"  Enter moves as UCI (e2e4) or SAN (Nf3)")
    print(f"  Type 'quit' or 'resign' to end the game")
    print(f"{'='*50}")

    if args.fen:
        game = MonsterChessGame(fen=args.fen)
    else:
        game = MonsterChessGame()

    records = []
    move_number = 0
    last_action = None

    render_board(game.board, flip=human_is_white)

    while not game.is_terminal():
        is_white = game.is_white_turn
        is_human = (is_white == human_is_white)

        if is_human:
            if human_is_white:
                action = get_human_white_action(game)
            else:
                action = get_human_black_action(game)

            if action == "resign":
                winner = "Black" if human_is_white else "White"
                print(f"\n  You resigned. {winner} wins!")
                result = -1 if human_is_white else 1
                for rec in records:
                    rec["game_result"] = result
                break

            if action is None:
                print(f"\n  No legal moves! Stalemate.")
                for rec in records:
                    rec["game_result"] = 0
                break

            if args.save_data:
                eval_white = heuristic_eval(game) if eval_fn is None else eval_fn(game)
                side_value = eval_white if is_white else -eval_white
                if is_white:
                    m1, m2 = action
                    policy = {f"{m1.uci()},{m2.uci()}": 1.0}
                else:
                    policy = {action.uci(): 1.0}
                records.append({
                    "fen": game.fen(),
                    "mcts_value": round(side_value, 4),
                    "policy": policy,
                    "current_player": "white" if is_white else "black",
                    "source": "human_game",
                    "actor": "human",
                    "session_id": session_id,
                    "human_color": args.color,
                    "ai_color": ai_color,
                    "ai_simulations": args.sims,
                    "ai_eval_type": "heuristic" if eval_fn is None else "nn",
                    "ai_model_path": None if eval_fn is None else model_used,
                })

            game.apply_action(action)
            last_action = action
            render_board(game.board, last_move=action, flip=human_is_white)

        else:
            side = "White" if is_white else "Black"
            print(f"  {DIM}AI ({side}) thinking...{RESET}", end="", flush=True)
            t0 = time.time()
            action, action_probs, root_value = engine.get_best_action(
                game, temperature=0.1,
            )
            elapsed = time.time() - t0

            if action is None:
                print(f"\n  AI has no legal moves! Stalemate.")
                for rec in records:
                    rec["game_result"] = 0
                break

            # Record AI's position for training data
            if args.save_data:
                records.append({
                    "fen": game.fen(),
                    "mcts_value": round(root_value, 4),
                    "policy": action_probs,
                    "current_player": "white" if is_white else "black",
                    "source": "human_game",
                    "actor": "ai",
                    "session_id": session_id,
                    "human_color": args.color,
                    "ai_color": ai_color,
                    "ai_simulations": args.sims,
                    "ai_eval_type": "heuristic" if eval_fn is None else "nn",
                    "ai_model_path": None if eval_fn is None else model_used,
                })

            game.apply_action(action)
            last_action = action

            # Display AI's move
            if is_white:
                m1, m2 = action
                move_str = f"{m1.uci()} + {m2.uci()}"
            else:
                move_str = action.uci()

            print(f"\r  AI ({side}): {BOLD}{move_str}{RESET}  "
                  f"({elapsed:.1f}s, eval {root_value:+.2f})")

            render_board(game.board, last_move=action, flip=human_is_white)
            print(format_eval(root_value))
            print()

        move_number += 1

    # Game over
    if game.is_terminal():
        result = game.get_result()
        if result > 0:
            print(f"\n  {BOLD}White wins!{RESET} (king captured)")
        elif result < 0:
            print(f"\n  {BOLD}Black wins!{RESET} (king captured)")
        else:
            print(f"\n  {BOLD}Draw{RESET} (turn limit)")

        if args.save_data:
            for rec in records:
                rec["game_result"] = result

    # Save data
    if args.save_data and records:
        out_dir = os.path.join(RAW_DATA_DIR, "human_games")
        path = save_game_data(records, out_dir)
        print(f"\n  Game saved to {path} ({len(records)} positions)")

    print()


if __name__ == "__main__":
    main()
