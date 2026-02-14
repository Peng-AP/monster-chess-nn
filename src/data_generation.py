import argparse
import chess
import json
import os
import time
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed

from tqdm import tqdm

from config import (
    MCTS_SIMULATIONS, NUM_GAMES,
    TEMPERATURE_HIGH, TEMPERATURE_LOW, TEMPERATURE_MOVES,
    RAW_DATA_DIR, CURRICULUM_FENS,
    CURRICULUM_TIER_BOUNDARIES, CURRICULUM_TIER_VALUES,
)
from monster_chess import MonsterChessGame
from mcts import MCTS

# Global variables set by each worker process
_eval_fn = None
_curriculum = False
_scripted_black = False
_force_result = None


def _init_worker(model_path, curriculum, scripted_black, force_result):
    """Initializer for worker processes — loads model once per worker."""
    global _eval_fn, _curriculum, _scripted_black, _force_result
    _curriculum = curriculum
    _scripted_black = scripted_black
    _force_result = force_result
    if model_path:
        from evaluation import NNEvaluator
        _eval_fn = NNEvaluator(model_path)


def play_game(num_simulations):
    """Play one full game of Monster Chess via MCTS self-play.

    Uses NN evaluator if loaded by worker, otherwise heuristic.
    Uses curriculum starting positions if _curriculum is set.
    Uses scripted Black play if _scripted_black is set (curriculum only).

    Returns a list of records, one per position:
        {fen, mcts_value, policy, current_player, game_result}
    """
    import random as rng

    engine = MCTS(num_simulations=num_simulations, eval_fn=_eval_fn)

    scripted_fn = None
    if _scripted_black:
        from scripted_endgame import get_scripted_black_move
        scripted_fn = get_scripted_black_move

    chosen_fen_idx = None
    if _curriculum:
        chosen_fen_idx = rng.randrange(len(CURRICULUM_FENS))
        fen = CURRICULUM_FENS[chosen_fen_idx]
        game = MonsterChessGame(fen=fen)
    else:
        game = MonsterChessGame()
    records = []
    move_number = 0

    while not game.is_terminal():
        is_white = game.is_white_turn

        # Data quality: decide whether to record this position
        skip_record = (
            move_number < 5                       # skip noisy early plies
            or game.board.is_check()              # skip tactical positions
            or rng.random() < 0.3                 # random subsample (keep ~70%)
        )

        if not is_white and scripted_fn is not None:
            # Use scripted play for Black in curriculum games
            game.board.turn = chess.BLACK
            action = scripted_fn(game.board)
            if action is None:
                break
            if not skip_record:
                from evaluation import evaluate
                root_value = evaluate(game)
                records.append({
                    "fen": game.fen(),
                    "mcts_value": round(-root_value, 4),
                    "policy": {action.uci(): 1.0},
                    "current_player": "black",
                })
            game.apply_action(action)
        else:
            temperature = TEMPERATURE_HIGH if move_number < TEMPERATURE_MOVES else TEMPERATURE_LOW
            action, action_probs, root_value = engine.get_best_action(
                game, temperature=temperature,
            )
            if action is None:
                break
            if not skip_record:
                records.append({
                    "fen": game.fen(),
                    "mcts_value": round(root_value, 4),
                    "policy": action_probs,
                    "current_player": "white" if is_white else "black",
                })
            game.apply_action(action)

        move_number += 1

    if _force_result is not None:
        result = _force_result
    elif _curriculum and chosen_fen_idx is not None:
        # Per-tier forced value: look up which tier this FEN belongs to
        tier = 0
        for boundary in CURRICULUM_TIER_BOUNDARIES:
            if chosen_fen_idx < boundary:
                break
            tier += 1
        result = CURRICULUM_TIER_VALUES[tier]
    else:
        result = game.get_result()
    for rec in records:
        rec["game_result"] = result

    return records


def save_game(records, output_dir, game_id):
    """Save a single game's data as a JSON-lines file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"game_{game_id:05d}.jsonl")
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _worker(args):
    """Worker function for multiprocessing."""
    game_id, num_simulations = args
    t0 = time.time()
    records = play_game(num_simulations)
    elapsed = time.time() - t0
    return game_id, records, elapsed


def main():
    parser = argparse.ArgumentParser(description="Generate Monster Chess training data via MCTS self-play")
    parser.add_argument("--num-games", type=int, default=NUM_GAMES)
    parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS)
    parser.add_argument("--output-dir", type=str, default=RAW_DATA_DIR)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--use-model", type=str, default=None,
                        help="Path to trained .pt model for NN evaluation")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use endgame curriculum starting positions")
    parser.add_argument("--scripted-black", action="store_true",
                        help="Use scripted endgame play for Black (curriculum only)")
    parser.add_argument("--force-result", type=int, default=None, choices=[-1, 0, 1],
                        help="Override game result for all games (1=White, -1=Black, 0=Draw)")
    args = parser.parse_args()

    workers = args.workers or os.cpu_count()
    model_path = args.use_model

    # With NN evaluation, use fewer workers (GPU memory / TF overhead)
    if model_path:
        workers = min(workers, 4)
        print(f"Using NN evaluator: {model_path}")
        print(f"(Limiting to {workers} workers for NN memory)")

    if args.curriculum:
        print("Using curriculum endgame starting positions")
    if args.scripted_black:
        print("Using scripted endgame play for Black")
    if args.force_result is not None:
        label = {1: "White win", -1: "Black win", 0: "Draw"}[args.force_result]
        print(f"Forcing game result: {args.force_result} ({label})")

    total_positions = 0
    results = {1: 0, -1: 0, 0: 0}

    print(f"Generating {args.num_games} games with {args.simulations} MCTS simulations per move...")
    print(f"Workers: {workers}")
    print(f"Output directory: {args.output_dir}\n")

    tasks = [(i, args.simulations) for i in range(args.num_games)]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(model_path, args.curriculum, args.scripted_black, args.force_result),
    ) as executor:
        futures = {executor.submit(_worker, task): task[0] for task in tasks}

        with tqdm(total=args.num_games, desc="Games") as pbar:
            try:
                for future in as_completed(futures, timeout=600):
                    try:
                        game_id, records, elapsed = future.result()
                    except (BrokenExecutor, Exception) as e:
                        gid = futures[future]
                        tqdm.write(f"  Game {gid}: FAILED ({e}), skipping")
                        pbar.update(1)
                        continue
                    save_game(records, args.output_dir, game_id=game_id)

                    n_moves = len(records)
                    total_positions += n_moves
                    game_result = records[-1]["game_result"] if records else 0
                    results[game_result] = results.get(game_result, 0) + 1

                    if game_result > 0:
                        winner = "White"
                    elif game_result < 0:
                        winner = "Black"
                    else:
                        winner = "Draw"
                    tqdm.write(f"  Game {game_id}: {n_moves} moves, {winner} ({game_result}), {elapsed:.1f}s")
                    pbar.update(1)
            except TimeoutError:
                hung = sum(1 for f in futures if not f.done())
                tqdm.write(f"\n  WARNING: {hung} game(s) timed out after 600s, skipping")
                pbar.update(hung)
                for f in futures:
                    f.cancel()

    white = sum(v for k, v in results.items() if k > 0)
    black = sum(v for k, v in results.items() if k < 0)
    draws = results.get(0, 0)
    print(f"\nDone! {total_positions} total positions across {args.num_games} games.")
    print(f"Results — White: {white}, Black: {black}, Draw: {draws}")


if __name__ == "__main__":
    main()
