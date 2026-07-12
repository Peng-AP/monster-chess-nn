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


def _allows_immediate_king_loss(game, action):
    """True if, after `action`, the opponent can capture the mover's king on
    the very next turn (the observed despair-suicide)."""
    from evaluation import _white_threat_scan, _black_can_capture_king
    tmp = game.clone()
    tmp.apply_action(action)
    if tmp.is_terminal():
        return False  # game over already; nothing to hang
    if tmp.is_white_turn:
        return _white_threat_scan(tmp)          # mover was Black
    return _black_can_capture_king(tmp.board)   # mover was White


def _swindle_override(game, action, val):
    """Swindle mode (play-time only): never hand over the king when a
    surviving alternative exists.

    v15 diagnosis (2026-07-12): a well-calibrated value head reads every move
    in a lost position as ~-0.98, so search picks among hopeless moves
    indifferently and often hangs the king to the fastest mate. Training
    labels can't fix this without re-taxing the winner (v13 lesson), so the
    resistance rule lives at play time: keep the searched action unless it
    allows an immediate king capture AND some legal action does not.
    Benchmark / arena / data generation never call this.
    """
    # No eval gate: allowing your own king to be captured is never correct at
    # ANY evaluation (the game simply ends), so the net is unconditional.
    if action is None:
        return action
    if not _allows_immediate_king_loss(game, action):
        return action
    legal = game.get_legal_actions()
    for alt in legal:
        tmp = game.clone()
        tmp.apply_action(alt)
        if tmp.is_terminal():
            r = tmp.get_result()
            if r != 0 and (r > 0) == game.is_white_turn:
                return alt  # a winning action existed — take it
            continue
        if not _allows_immediate_king_loss(game, alt):
            return alt
    return action  # every move loses the king — forced


def ai_select_full_action(engine, game, temperature, swindle=False):
    """Search half-moves but return a full atomic action for play/display.

    The engine now searches White's turn as two half-moves, so get_best_action
    returns a single move.  For White we search m1, apply it on a clone, then search
    m2 — letting play.py keep applying atomic (m1, m2) pairs and rendering both moves
    together.  Black is already a single move (REWORK_PLAN.md Phase 3).

    swindle=True enables the lost-position resistance override (human play
    only — see _swindle_override).
    """
    action, probs, val = engine.get_best_action(game, temperature=temperature)
    if action is None:
        return None, probs, val
    if game.is_white_turn and not game.white_half_pending and not isinstance(action, tuple):
        tmp = game.clone()
        tmp.apply_search_action(action)
        if tmp.is_terminal():
            return (action, chess.Move.null()), probs, val
        action2, _p2, _v2 = engine.get_best_action(tmp, temperature=temperature)
        if action2 is None:
            action = (action, chess.Move.null())
        else:
            action = (action, action2)
    if swindle:
        action = _swindle_override(game, action, val)
    return action, probs, val


def parse_move(text, board, legal_set=None):
    """Parse user input into a chess.Move. Accepts UCI (e2e4) or SAN (Nf3).

    legal_set: when provided, candidate moves are validated against it and a
    manual-SAN fallback runs over it.  parse_san rejects moves that are legal
    in Monster Chess but illegal in standard chess (king moves into check on
    White's first half, king-captures-king), so SAN input for those moves
    only works through the fallback.  King moves also accept the no-'x' and
    lowercase forms ("Kd4"/"kxd4") — 'k' is not a file letter, so this is
    unambiguous.
    """
    text = text.strip()
    valid = set(legal_set) if legal_set is not None else None
    # Try UCI first
    try:
        move = chess.Move.from_uci(text)
        if valid is not None:
            if move in valid:
                return move
        elif move in board.pseudo_legal_moves or move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError):
        pass
    # Try standard SAN
    try:
        move = board.parse_san(text)
        if valid is None or move in valid:
            return move
    except (ValueError, chess.InvalidMoveError, chess.AmbiguousMoveError):
        pass
    if valid is None:
        return None
    # Manual-SAN fallback over the true legal set (Monster Chess moves that
    # python-chess's standard-chess validator refuses).
    clean = text.rstrip('+#')
    for move in valid:
        try:
            if board.san(move).rstrip('+#') == clean:
                return move
        except Exception:
            pass  # board.san can raise on king-capture moves
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
        is_cap = board.is_capture(move) or board.piece_at(move.to_square) is not None
        to_sq = chess.square_name(move.to_square)
        if piece.piece_type == chess.PAWN:
            manual = (chr(ord('a') + chess.square_file(move.from_square)) + 'x' + to_sq
                      if is_cap else to_sq)
            if manual == clean:
                return move
        else:
            manual = piece.symbol().upper() + ('x' if is_cap else '') + to_sq
            if manual == clean:
                return move
            if piece.piece_type == chess.KING and clean.lower() in (
                    "k" + to_sq, "kx" + to_sq):
                return move
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
        m1 = parse_move(text, board, legal_set=legal_first)
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
        m2 = parse_move(text, board, legal_set=legal_second)
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
        if board.king(chess.BLACK) is None:
            # m2 captured the Black king — wins UNCONDITIONALLY, own-king
            # safety is irrelevant (the game ends first).
            board.pop()
            break
        wk = board.king(chess.WHITE)
        if wk is not None and board.is_attacked_by(chess.BLACK, wk):
            # Check if ANY winning-or-safe pair exists — if not, allow it
            # (forced blunder)
            board.pop()
            board.turn = chess.BLACK
            board.pop()
            safe_exists = _winning_or_safe_pair_exists(game)
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


def _winning_or_safe_pair_exists(game):
    """True if White has a pair that wins (king capture) or leaves the king safe.

    get_legal_actions lists winning pairs first, then safe pairs, and falls
    back to all pairs only when neither exists — so testing the first pair
    is sufficient.
    """
    clone = game.clone()
    pairs = clone.get_legal_actions()
    if not pairs:
        return False
    m1, m2 = pairs[0]
    b = clone.board
    b.turn = chess.WHITE
    b.push(m1)
    if b.king(chess.BLACK) is None:
        return True
    b.turn = chess.WHITE
    if m2 != chess.Move.null():
        b.push(m2)
    if b.king(chess.BLACK) is None:
        return True
    wk = b.king(chess.WHITE)
    return wk is not None and not b.is_attacked_by(chess.BLACK, wk)


def get_human_black_action(game):
    """Get a single move from the human playing Black."""
    legal = game.get_legal_actions()

    if not legal:
        return None

    while True:
        text = input(f"  {BOLD}Your move:{RESET} ").strip()
        if text.lower() in ("quit", "resign", "q"):
            return "resign"
        move = parse_move(text, game.board, legal_set=legal)
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

    # Human play: deterministic priors (no root noise), early stop allowed.
    engine = MCTS(num_simulations=args.sims, eval_fn=eval_fn,
                  root_noise=False, allow_early_stop=True)
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
            action, action_probs, root_value = ai_select_full_action(
                engine, game, temperature=0.1, swindle=True,
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
