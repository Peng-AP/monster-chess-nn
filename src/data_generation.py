import argparse
import chess
import json
import os
import random
import signal
import time
from collections import Counter
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed

from tqdm import tqdm

from config import (
    MCTS_SIMULATIONS, NUM_GAMES, OPPONENT_SIMULATIONS,
    TEMPERATURE_HIGH, TEMPERATURE_LOW, TEMPERATURE_MOVES,
    RAW_DATA_DIR, CURRICULUM_FENS,
    CURRICULUM_TIER_BOUNDARIES, CURRICULUM_TIER_VALUES,
    SKIP_CHECK_POSITIONS,
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
_skip_check_positions = SKIP_CHECK_POSITIONS
_start_fens = []


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
    check_skip = _skip_check_positions and board_is_check
    return early_skip or check_skip or random_skip


def _fen_turn_side(fen):
    """Return side-to-move label from FEN, or None if invalid."""
    try:
        board = chess.Board(fen)
    except Exception:
        return None
    return "white" if board.turn == chess.WHITE else "black"


def _iter_start_fen_paths(file_path, dir_path):
    """Yield JSONL paths from explicit file and/or directory."""
    if file_path:
        yield file_path
    if dir_path:
        for name in sorted(os.listdir(dir_path)):
            if name.endswith(".jsonl"):
                yield os.path.join(dir_path, name)


def _convert_white_to_black_start(fen, rng, policy_dict=None):
    """Advance one White turn to produce a Black-to-move start FEN.

    If a white-action policy is available, prefer the highest-probability
    legal action to keep converted starts aligned with recorded play.
    """
    try:
        game = MonsterChessGame(fen=fen)
    except Exception:
        return None
    if not game.is_white_turn or game.is_terminal():
        return None
    actions = game.get_legal_actions()
    if not actions:
        return None
    action = None
    if isinstance(policy_dict, dict) and policy_dict:
        legal_set = set(actions)
        # Try highest-probability actions first.
        ordered = sorted(policy_dict.items(), key=lambda kv: float(kv[1]), reverse=True)
        for action_str, _prob in ordered:
            try:
                parsed = MonsterChessGame.str_to_action(action_str, is_white=True)
            except Exception:
                continue
            if parsed in legal_set:
                action = parsed
                break
    if action is None:
        action = rng.choice(actions)
    game.apply_action(action)
    if game.is_white_turn:
        return None
    return game.fen()


def _load_start_fens(file_path=None, dir_path=None, side_filter="any",
                     max_positions=0, seed=None,
                     convert_white_to_black=False):
    """Load start FEN candidates from JSONL records with optional side filter."""
    if not file_path and not dir_path:
        return [], {
            "files": 0,
            "records_total": 0,
            "records_valid": 0,
            "records_kept": 0,
        }

    side_filter = str(side_filter or "any").lower()
    if side_filter not in ("any", "white", "black"):
        raise ValueError("start FEN side filter must be one of: any, white, black")

    all_fens = []
    white_side_candidates = []
    records_total = 0
    records_valid = 0
    files_used = 0
    for path in _iter_start_fen_paths(file_path=file_path, dir_path=dir_path):
        if not os.path.isfile(path):
            continue
        files_used += 1
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records_total += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, str):
                    fen = obj
                    rec_side = _fen_turn_side(fen)
                else:
                    fen = obj.get("fen") if isinstance(obj, dict) else None
                    if not fen:
                        continue
                    rec_side = (
                        str(obj.get("current_player", "")).strip().lower()
                        if isinstance(obj, dict) else ""
                    )
                    if rec_side not in ("white", "black"):
                        rec_side = _fen_turn_side(fen)
                if rec_side is None:
                    continue
                records_valid += 1
                if side_filter != "any" and rec_side != side_filter:
                    if side_filter == "black" and rec_side == "white" and convert_white_to_black:
                        policy = obj.get("policy") if isinstance(obj, dict) else None
                        white_side_candidates.append((fen, policy))
                    continue
                all_fens.append(fen)

    converted_added = 0
    if side_filter == "black" and convert_white_to_black and white_side_candidates:
        convert_rng = random.Random(seed if seed is not None else 0)
        for white_fen, white_policy in white_side_candidates:
            converted = _convert_white_to_black_start(
                white_fen, convert_rng, policy_dict=white_policy
            )
            if converted:
                all_fens.append(converted)
                converted_added += 1

    if max_positions and max_positions > 0 and len(all_fens) > max_positions:
        rng = random.Random(seed)
        all_fens = rng.sample(all_fens, k=max_positions)

    return all_fens, {
        "files": files_used,
        "records_total": records_total,
        "records_valid": records_valid,
        "records_kept": len(all_fens),
        "white_candidates_for_conversion": len(white_side_candidates),
        "converted_added": converted_added,
    }


def _init_worker(model_path, opponent_model_path, curriculum, curriculum_live_results, scripted_black,
                 force_result, train_side, opponent_sims, opponent_pool_paths,
                 skip_check_positions, start_fens):
    """Initializer for worker processes - loads model(s) once per worker."""
    global _eval_fn, _opponent_eval_fn, _opponent_eval_pool
    global _curriculum, _curriculum_live_results, _scripted_black
    global _force_result, _train_side, _opponent_sims, _skip_check_positions, _start_fens
    _curriculum = curriculum
    _curriculum_live_results = curriculum_live_results
    _scripted_black = scripted_black
    _force_result = force_result
    _train_side = train_side
    _opponent_sims = opponent_sims
    _skip_check_positions = skip_check_positions
    _start_fens = list(start_fens) if start_fens else []
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
    if _start_fens:
        fen = rng.choice(_start_fens)
        game = MonsterChessGame(fen=fen)
    elif _curriculum:
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


def _simulation_stats(sim_values):
    """Build compact stats for sampled per-game simulation budgets."""
    if not sim_values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "counts": {},
        }
    counts = Counter(sim_values)
    mean_val = sum(sim_values) / len(sim_values)
    return {
        "min": int(min(sim_values)),
        "max": int(max(sim_values)),
        "mean": float(mean_val),
        "counts": {str(k): int(v) for k, v in sorted(counts.items())},
    }


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
    return game_id, num_simulations, records, elapsed


def main():
    parser = argparse.ArgumentParser(description="Generate Monster Chess training data via MCTS self-play")
    parser.add_argument("--num-games", type=int, default=NUM_GAMES)
    parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS)
    parser.add_argument("--simulations-min", type=int, default=None,
                        help="Minimum simulations per game (default: --simulations)")
    parser.add_argument("--simulations-max", type=int, default=None,
                        help="Maximum simulations per game (default: --simulations)")
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
    parser.add_argument("--keep-check-positions", action="store_true",
                        help="Keep positions where side to move is in check (default: skip them)")
    parser.add_argument("--start-fen-file", type=str, default=None,
                        help="Optional JSONL file of start positions (expects records with 'fen')")
    parser.add_argument("--start-fen-dir", type=str, default=None,
                        help="Optional directory of JSONL files for start positions")
    parser.add_argument("--start-fen-side", type=str, default="any",
                        choices=["any", "white", "black"],
                        help="Filter loaded start positions by side to move")
    parser.add_argument("--start-fen-max-positions", type=int, default=0,
                        help="Cap loaded start positions (0 = no cap)")
    parser.add_argument("--start-fen-convert-white-to-black", action="store_true",
                        help="When filtering for black starts, convert white-to-move records by sampling one White turn")
    args = parser.parse_args()
    if args.simulations <= 0:
        raise ValueError("--simulations must be > 0")
    if args.simulations_min is not None and args.simulations_min <= 0:
        raise ValueError("--simulations-min must be > 0")
    if args.simulations_max is not None and args.simulations_max <= 0:
        raise ValueError("--simulations-max must be > 0")
    if args.start_fen_max_positions < 0:
        raise ValueError("--start-fen-max-positions must be >= 0")

    sim_min = args.simulations if args.simulations_min is None else args.simulations_min
    sim_max = args.simulations if args.simulations_max is None else args.simulations_max
    if sim_min > sim_max:
        raise ValueError("--simulations-min must be <= --simulations-max")
    if args.curriculum and (args.start_fen_file or args.start_fen_dir):
        raise ValueError("--curriculum cannot be combined with start-fen sources")
    if args.start_fen_file and not os.path.isfile(args.start_fen_file):
        raise FileNotFoundError(f"--start-fen-file not found: {args.start_fen_file}")
    if args.start_fen_dir and not os.path.isdir(args.start_fen_dir):
        raise FileNotFoundError(f"--start-fen-dir not found: {args.start_fen_dir}")

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

    start_fens, start_fen_stats = _load_start_fens(
        file_path=args.start_fen_file,
        dir_path=args.start_fen_dir,
        side_filter=args.start_fen_side,
        max_positions=args.start_fen_max_positions,
        seed=args.seed,
        convert_white_to_black=args.start_fen_convert_white_to_black,
    )
    if (args.start_fen_file or args.start_fen_dir) and not start_fens:
        raise ValueError("No start positions loaded after applying filters")
    if start_fens:
        print(
            f"Using start-position source: kept {len(start_fens)} FENs "
            f"(files={start_fen_stats['files']}, valid={start_fen_stats['records_valid']}, "
            f"total={start_fen_stats['records_total']}, side={args.start_fen_side}, "
            f"converted={start_fen_stats.get('converted_added', 0)})"
        )

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
    if args.keep_check_positions:
        print("Retention: keeping in-check positions")
    else:
        print("Retention: skipping in-check positions")
    if sim_min == sim_max:
        print(f"Simulation budget: fixed {sim_min} per game")
    else:
        print(f"Simulation budget: randomized uniformly in [{sim_min}, {sim_max}] per game")

    total_positions = 0
    results = {1: 0, -1: 0, 0: 0}
    saved_games = 0
    skipped_empty = 0
    failed_games = 0
    timed_out_games = 0

    if sim_min == sim_max:
        print(f"Generating {args.num_games} games with {sim_min} MCTS simulations per move...")
    else:
        print(
            f"Generating {args.num_games} games with randomized MCTS simulations per move "
            f"in [{sim_min}, {sim_max}]..."
        )
    print(f"Workers: {workers}")
    print(f"Output directory: {args.output_dir}\n")

    tasks = []
    sampled_simulations = []
    schedule_rng = random.Random(args.seed) if args.seed is not None else random.Random()
    for i in range(args.num_games):
        task_seed = None if args.seed is None else args.seed + i
        if sim_min == sim_max:
            sim_for_game = sim_min
        else:
            sim_for_game = schedule_rng.randint(sim_min, sim_max)
        sampled_simulations.append(sim_for_game)
        tasks.append((i, sim_for_game, task_seed))

    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(model_path, opponent_model_path, args.curriculum, args.curriculum_live_results,
                  args.scripted_black, args.force_result,
                  args.train_side, args.opponent_sims, opponent_pool_paths,
                  not args.keep_check_positions, start_fens),
    )
    try:
        futures = {executor.submit(_worker, task): task[0] for task in tasks}

        with tqdm(total=args.num_games, desc="Games") as pbar:
            try:
                for future in as_completed(futures, timeout=600):
                    try:
                        game_id, _sim_used, records, elapsed = future.result()
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
    sim_stats = _simulation_stats(sampled_simulations)
    if sim_stats["min"] is not None:
        print(
            f"Simulation usage stats: min={sim_stats['min']} max={sim_stats['max']} "
            f"mean={sim_stats['mean']:.2f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_games_requested": int(args.num_games),
            "saved_games": int(saved_games),
            "skipped_empty": int(skipped_empty),
            "failed_games": int(failed_games),
            "timed_out_games": int(timed_out_games),
            "total_positions": int(total_positions),
            "results": {
                "white_wins": int(white),
                "black_wins": int(black),
                "draws": int(draws),
            },
            "simulations": {
                "configured_base": int(args.simulations),
                "configured_min": int(sim_min),
                "configured_max": int(sim_max),
                "sampled_stats": sim_stats,
            },
            "workers": int(workers),
            "train_side": args.train_side,
            "curriculum": bool(args.curriculum),
            "curriculum_live_results": bool(args.curriculum_live_results),
            "scripted_black": bool(args.scripted_black),
            "keep_check_positions": bool(args.keep_check_positions),
            "seed": int(args.seed) if args.seed is not None else None,
            "start_fen": {
                "enabled": bool(len(start_fens) > 0),
                "file": args.start_fen_file,
                "dir": args.start_fen_dir,
                "side_filter": args.start_fen_side,
                "max_positions": int(args.start_fen_max_positions),
                "convert_white_to_black": bool(args.start_fen_convert_white_to_black),
                "loaded_positions": int(len(start_fens)),
                "source_stats": start_fen_stats,
            },
        }, f, indent=2)
    print(f"Generation summary saved to {summary_path}")


if __name__ == "__main__":
    main()
