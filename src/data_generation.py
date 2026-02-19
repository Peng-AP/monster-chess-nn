import argparse
import chess
import json
import os
import signal
import time
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed

from tqdm import tqdm

from config import (
    MCTS_SIMULATIONS, NUM_GAMES, OPPONENT_SIMULATIONS,
    TEMPERATURE_HIGH, TEMPERATURE_LOW, TEMPERATURE_MOVES,
    RAW_DATA_DIR, CURRICULUM_FENS,
    CURRICULUM_TIER_BOUNDARIES, CURRICULUM_TIER_VALUES,
)
from monster_chess import MonsterChessGame
from mcts import MCTS

def _kill_workers(executor):
    """Force-kill any lingering worker processes after shutdown."""
    # ProcessPoolExecutor stores workers in _processes (dict of pid -> process)
    procs = getattr(executor, '_processes', None)
    if not procs:
        return
    for pid, proc in list(procs.items()):
        if proc.is_alive():
            try:
                os.kill(pid, signal.SIGTERM)
                proc.join(timeout=3)
                if proc.is_alive():
                    os.kill(pid, 9)  # SIGKILL
            except (OSError, ProcessLookupError):
                pass


# Global variables set by each worker process
_eval_fn = None
_opponent_eval_fn = None
_opponent_eval_pool = []
_curriculum = False
_curriculum_live_results = False
_scripted_black = False
_force_result = None
_train_side = "both"
_opponent_sims = OPPONENT_SIMULATIONS


def _is_training_side_position(is_white):
    """Return True when this position belongs to the side being trained."""
    if _train_side == "both":
        return True
    if _train_side == "white":
        return is_white
    if _train_side == "black":
        return not is_white
    return True


def _should_skip_record(move_number, is_white, board_is_check, rng):
    """Data retention policy with side-aware emphasis for alternating mode.

    Rationale:
      - In alternating mode, keep more positions from the training side,
        especially early plies where opening/defensive patterns are learned.
      - Keep stronger filtering on non-training-side positions to avoid
        blowing up dataset size with lower-priority signal.
    """
    if _curriculum:
        # Curriculum positions are already targeted; keep all unless tactical.
        early_skip = False
        subsample_p = 0.10
    else:
        is_train_pos = _is_training_side_position(is_white)
        if _train_side == "both":
            early_cutoff = 5
            subsample_p = 0.30
        elif is_train_pos:
            # Preserve early training-side plies (was previously skipped).
            early_cutoff = 0
            subsample_p = 0.15
        else:
            # Deprioritize opponent-side opening plies.
            early_cutoff = 5
            subsample_p = 0.45
        early_skip = move_number < early_cutoff

    random_skip = rng.random() < subsample_p
    return early_skip or board_is_check or random_skip


def _init_worker(model_path, opponent_model_path, curriculum, curriculum_live_results, scripted_black,
                 force_result, train_side, opponent_sims, opponent_pool_paths):
    """Initializer for worker processes - loads model(s) once per worker."""
    global _eval_fn, _opponent_eval_fn, _opponent_eval_pool
    global _curriculum, _curriculum_live_results, _scripted_black
    global _force_result, _train_side, _opponent_sims
    _curriculum = curriculum
    _curriculum_live_results = curriculum_live_results
    _scripted_black = scripted_black
    _force_result = force_result
    _train_side = train_side
    _opponent_sims = opponent_sims
    if model_path:
        from evaluation import NNEvaluator
        _eval_fn = NNEvaluator(model_path)
    if opponent_model_path:
        from evaluation import NNEvaluator
        _opponent_eval_fn = NNEvaluator(opponent_model_path)
    _opponent_eval_pool = []
    if opponent_pool_paths:
        from evaluation import NNEvaluator
        for p in opponent_pool_paths:
            _opponent_eval_pool.append(NNEvaluator(p))


def play_game(num_simulations):
    """Play one full game of Monster Chess via MCTS self-play.

    Uses NN evaluator if loaded by worker, otherwise heuristic.
    Uses curriculum starting positions if _curriculum is set.
    Uses scripted Black play if _scripted_black is set (curriculum only).

    When _train_side is "white" or "black", uses separate MCTS engines
    for each side (frozen-opponent alternating training).

    Returns a list of records, one per position:
        {fen, mcts_value, policy, current_player, game_result}
    """
    import random as rng

    # Build engine(s) based on training mode
    if _train_side == "both":
        # Single engine for both sides (backward-compatible)
        engine = MCTS(num_simulations=num_simulations, eval_fn=_eval_fn)
        white_engine = engine
        black_engine = engine
    else:
        # Dual engines: training side gets full sims + main model,
        # opponent side gets fewer sims + frozen/heuristic model
        opponent_eval = _opponent_eval_fn
        if _opponent_eval_pool:
            opponent_eval = rng.choice(_opponent_eval_pool)
        train_engine = MCTS(num_simulations=num_simulations, eval_fn=_eval_fn)
        opponent_engine = MCTS(num_simulations=_opponent_sims,
                               eval_fn=opponent_eval)
        if _train_side == "white":
            white_engine = train_engine
            black_engine = opponent_engine
        else:  # "black"
            white_engine = opponent_engine
            black_engine = train_engine

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

        # Side-aware retention to protect scarce training-side signal.
        skip_record = _should_skip_record(
            move_number=move_number,
            is_white=is_white,
            board_is_check=game.board.is_check(),
            rng=rng,
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
            # Pick the engine for the current side
            engine = white_engine if is_white else black_engine
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
    elif _curriculum and (not _curriculum_live_results) and chosen_fen_idx is not None:
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
    """Save one game's data as JSON lines. Returns path or None if empty."""
    if not records:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"game_{game_id:05d}.jsonl")
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _worker(args):
    """Worker function for multiprocessing."""
    game_id, num_simulations, seed = args
    if seed is not None:
        import random as rng
        rng.seed(seed)
        np_seed = seed % (2 ** 32 - 1)
        try:
            import numpy as np
            np.random.seed(np_seed)
        except Exception:
            pass
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
    parser.add_argument("--curriculum-live-results", action="store_true",
                        help="For curriculum starts, use actual game_result instead of forced tier values")
    parser.add_argument("--scripted-black", action="store_true",
                        help="Use scripted endgame play for Black (curriculum only)")
    parser.add_argument("--force-result", type=int, default=None, choices=[-1, 0, 1],
                        help="Override game result for all games (1=White, -1=Black, 0=Draw)")
    parser.add_argument("--train-side", type=str, default="both",
                        choices=["white", "black", "both"],
                        help="Which side is being trained (frozen-opponent mode)")
    parser.add_argument("--opponent-model", type=str, default=None,
                        help="Path to frozen opponent model (omit = heuristic)")
    parser.add_argument("--opponent-pool-dir", type=str, default=None,
                        help="Directory containing archived opponent .pt models")
    parser.add_argument("--opponent-pool-size", type=int, default=5,
                        help="Use latest N models from opponent pool dir")
    parser.add_argument("--opponent-sims", type=int, default=OPPONENT_SIMULATIONS,
                        help=f"MCTS simulations for frozen opponent (default: {OPPONENT_SIMULATIONS})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (optional, enables deterministic game seeding)")
    args = parser.parse_args()

    workers = args.workers or os.cpu_count()
    model_path = args.use_model
    opponent_model_path = args.opponent_model
    opponent_pool_paths = []
    if args.opponent_pool_dir:
        if not os.path.isdir(args.opponent_pool_dir):
            raise FileNotFoundError(f"--opponent-pool-dir not found: {args.opponent_pool_dir}")
        pool_candidates = sorted(
            [os.path.join(args.opponent_pool_dir, f)
             for f in os.listdir(args.opponent_pool_dir)
             if f.endswith(".pt")],
            key=lambda p: os.path.getmtime(p),
        )
        if args.opponent_pool_size > 0:
            pool_candidates = pool_candidates[-args.opponent_pool_size:]
        opponent_pool_paths = pool_candidates

    # With NN evaluation, use fewer workers (GPU memory)
    if model_path or opponent_model_path or opponent_pool_paths:
        workers = min(workers, 4)
        if model_path:
            print(f"Using NN evaluator: {model_path}")
        if opponent_model_path:
            print(f"Using opponent NN evaluator: {opponent_model_path}")
        if opponent_pool_paths:
            print(f"Using opponent pool ({len(opponent_pool_paths)} models):")
            for p in opponent_pool_paths:
                print(f"  - {p}")
        print(f"(Limiting to {workers} workers for NN memory)")

    if args.train_side != "both":
        print(f"Alternating training: {args.train_side} side is training "
              f"({args.simulations} sims), opponent gets {args.opponent_sims} sims")

    if args.curriculum:
        print("Using curriculum endgame starting positions")
        if args.curriculum_live_results:
            print("Curriculum labels: using live game results")
        else:
            print("Curriculum labels: using forced tier values")
    if args.scripted_black:
        print("Using scripted endgame play for Black")
    if args.force_result is not None:
        label = {1: "White win", -1: "Black win", 0: "Draw"}[args.force_result]
        print(f"Forcing game result: {args.force_result} ({label})")
    if args.seed is not None:
        print(f"Using deterministic base seed: {args.seed}")

    total_positions = 0
    results = {1: 0, -1: 0, 0: 0}
    saved_games = 0
    skipped_empty = 0
    failed_games = 0
    timed_out_games = 0

    print(f"Generating {args.num_games} games with {args.simulations} MCTS simulations per move...")
    print(f"Workers: {workers}")
    print(f"Output directory: {args.output_dir}\n")

    tasks = []
    for i in range(args.num_games):
        task_seed = None if args.seed is None else args.seed + i
        tasks.append((i, args.simulations, task_seed))

    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(model_path, opponent_model_path, args.curriculum, args.curriculum_live_results,
                  args.scripted_black, args.force_result,
                  args.train_side, args.opponent_sims, opponent_pool_paths),
    )
    try:
        futures = {executor.submit(_worker, task): task[0] for task in tasks}

        with tqdm(total=args.num_games, desc="Games") as pbar:
            try:
                for future in as_completed(futures, timeout=600):
                    try:
                        game_id, records, elapsed = future.result()
                    except (BrokenExecutor, Exception) as e:
                        gid = futures[future]
                        tqdm.write(f"  Game {gid}: FAILED ({e}), skipping")
                        failed_games += 1
                        pbar.update(1)
                        continue
                    path = save_game(records, args.output_dir, game_id=game_id)
                    if path is None:
                        skipped_empty += 1
                        tqdm.write(f"  Game {game_id}: 0 recorded moves, skipped save ({elapsed:.1f}s)")
                        pbar.update(1)
                        continue

                    n_moves = len(records)
                    total_positions += n_moves
                    saved_games += 1
                    game_result = records[-1]["game_result"]
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
                tqdm.write(f"\n  WARNING: {hung} game(s) timed out after 600s, killing workers")
                timed_out_games += hung
                pbar.update(hung)
    finally:
        # shutdown(wait=False) returns immediately; cancel_futures prevents
        # queued tasks from starting. Then kill any still-running workers.
        executor.shutdown(wait=False, cancel_futures=True)
        _kill_workers(executor)

    white = sum(v for k, v in results.items() if k > 0)
    black = sum(v for k, v in results.items() if k < 0)
    draws = results.get(0, 0)
    print(f"\nDone! {total_positions} total positions across {saved_games} saved games (attempted {args.num_games}).")
    if skipped_empty > 0:
        print(f"Skipped empty games: {skipped_empty}")
    if failed_games > 0:
        print(f"Failed games: {failed_games}")
    if timed_out_games > 0:
        print(f"Timed out games: {timed_out_games}")
    print(f"Results - White: {white}, Black: {black}, Draw: {draws} (saved games only)")


if __name__ == "__main__":
    main()
