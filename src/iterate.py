"""Continuous self-play iteration loop.

Supports two modes:
  - Standard: both sides use the same model (original behavior)
  - Alternating: frozen-opponent training, alternates which side trains

Repeats: generate NN-guided games -> reprocess all data -> fine-tune model.
Each generation is saved to data/raw/nn_genN/ so nothing is overwritten.
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=SRC_DIR)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  FAILED: {desc} (exit code {result.returncode}, {elapsed:.0f}s)")
        sys.exit(1)
    print(f"\n  Done in {format_time(elapsed)}")
    return elapsed


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def find_next_gen(raw_dir):
    """Find the next generation number by scanning existing nn_gen* dirs."""
    gen = 1
    while os.path.exists(os.path.join(raw_dir, f"nn_gen{gen}")):
        gen += 1
    return gen


def count_data(raw_dir):
    """Count total games and positions across all raw data."""
    total_games = 0
    total_positions = 0
    for dirpath, _dirnames, filenames in os.walk(raw_dir):
        for fname in filenames:
            if fname.endswith(".jsonl"):
                total_games += 1
                path = os.path.join(dirpath, fname)
                with open(path) as f:
                    total_positions += sum(1 for _ in f)
    return total_games, total_positions


def summarize_generation(gen_dir):
    """Read game results from a generation directory and print a summary."""
    white_wins = 0
    black_wins = 0
    draws = 0
    game_lengths = []
    files = sorted(glob.glob(os.path.join(gen_dir, "*.jsonl")))
    empty_files = 0
    counted_games = 0
    for path in files:
        with open(path) as f:
            lines = f.readlines()
        if lines:
            last = json.loads(lines[-1])
            r = last.get("game_result", 0)
            if r > 0:
                white_wins += 1
            elif r < 0:
                black_wins += 1
            else:
                draws += 1
            game_lengths.append(len(lines))
            counted_games += 1
        else:
            empty_files += 1

    if not files:
        return None

    n = counted_games
    if n == 0:
        print(f"\n  --- Generation Summary ---")
        print(f"  Files:     {len(files)} (all empty)")
        print(f"  Games:     0")
        print(f"  Positions: 0")
        return {
            "games": 0,
            "positions": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "avg_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "empty_files": empty_files,
            "files": len(files),
        }

    avg_len = sum(game_lengths) / n
    min_len = min(game_lengths)
    max_len = max(game_lengths)
    total_pos = sum(game_lengths)

    summary = {
        "games": n,
        "positions": total_pos,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "avg_length": avg_len,
        "min_length": min_len,
        "max_length": max_len,
        "empty_files": empty_files,
        "files": len(files),
    }

    gen_summary_path = os.path.join(gen_dir, "generation_summary.json")
    if os.path.exists(gen_summary_path):
        try:
            with open(gen_summary_path, "r", encoding="utf-8") as f:
                generation_summary = json.load(f)
            summary["simulations"] = generation_summary.get("simulations")
        except Exception as e:
            summary["generation_summary_error"] = str(e)

    print(f"\n  --- Generation Summary ---")
    print(f"  Games:     {n} (files={len(files)}, empty={empty_files})")
    print(f"  Positions: {total_pos}")
    print(
        f"  Results:   White {white_wins} ({100*white_wins/n:.0f}%)  |  "
        f"Black {black_wins} ({100*black_wins/n:.0f}%)  |  "
        f"Draw {draws} ({100*draws/n:.0f}%)"
    )
    print(f"  Length:    avg {avg_len:.0f}  |  min {min_len}  |  max {max_len}")
    sim_info = summary.get("simulations")
    if isinstance(sim_info, dict):
        sampled = sim_info.get("sampled_stats", {})
        if sampled.get("min") is not None:
            print(
                f"  Sims:      min {sampled['min']}  |  max {sampled['max']}  |  "
                f"mean {sampled['mean']:.2f}"
            )

    return summary


def _scan_result_dir(gen_dir):
    """Return white/black/draw counts from a directory of game_*.jsonl files."""
    summary = {"white_wins": 0, "black_wins": 0, "draws": 0, "games": 0}
    files = sorted(glob.glob(os.path.join(gen_dir, "*.jsonl")))
    for path in files:
        with open(path) as f:
            lines = f.readlines()
        if not lines:
            continue
        last = json.loads(lines[-1])
        r = last.get("game_result", 0)
        if r > 0:
            summary["white_wins"] += 1
        elif r < 0:
            summary["black_wins"] += 1
        else:
            summary["draws"] += 1
        summary["games"] += 1
    return summary


def _candidate_score(result_summary, candidate_is_white):
    if candidate_is_white:
        wins = result_summary["white_wins"]
        losses = result_summary["black_wins"]
    else:
        wins = result_summary["black_wins"]
        losses = result_summary["white_wins"]
    draws = result_summary["draws"]
    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total if total > 0 else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": total,
        "score": score,
    }


def _list_archive_models(archive_dir):
    return sorted(glob.glob(os.path.join(archive_dir, "gen_*.pt")))


def _append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _seed_archive_if_empty(model_path, archive_dir, manifest_path):
    models = _list_archive_models(archive_dir)
    if models or not os.path.exists(model_path):
        return
    os.makedirs(archive_dir, exist_ok=True)
    seed_path = os.path.join(archive_dir, "gen_0000.pt")
    shutil.copy2(model_path, seed_path)
    _append_jsonl(manifest_path, {
        "event": "seed_archive",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generation": 0,
        "model_path": seed_path,
    })


def _run_arena(candidate_model, incumbent_model, gen, model_dir, arena_games,
               arena_sims, arena_workers, base_seed):
    """Evaluate candidate vs incumbent with color swap. Returns score dict."""
    arena_root = os.path.join(model_dir, "arena_runs", f"gen_{gen:04d}")
    white_dir = os.path.join(arena_root, "candidate_white")
    black_dir = os.path.join(arena_root, "candidate_black")
    if os.path.exists(arena_root):
        shutil.rmtree(arena_root)
    os.makedirs(white_dir, exist_ok=True)
    os.makedirs(black_dir, exist_ok=True)

    n_white = arena_games // 2
    n_black = arena_games - n_white

    if n_white > 0:
        run([
            sys.executable, "data_generation.py",
            "--num-games", str(n_white),
            "--simulations", str(arena_sims),
            "--workers", str(arena_workers),
            "--output-dir", white_dir,
            "--use-model", candidate_model,
            "--train-side", "white",
            "--opponent-model", incumbent_model,
            "--opponent-sims", str(arena_sims),
            "--seed", str(base_seed + gen * 100 + 11),
        ], f"Arena gen {gen}: candidate as White vs incumbent ({n_white} games)")

    if n_black > 0:
        run([
            sys.executable, "data_generation.py",
            "--num-games", str(n_black),
            "--simulations", str(arena_sims),
            "--workers", str(arena_workers),
            "--output-dir", black_dir,
            "--use-model", candidate_model,
            "--train-side", "black",
            "--opponent-model", incumbent_model,
            "--opponent-sims", str(arena_sims),
            "--seed", str(base_seed + gen * 100 + 22),
        ], f"Arena gen {gen}: candidate as Black vs incumbent ({n_black} games)")

    white_summary = _scan_result_dir(white_dir)
    black_summary = _scan_result_dir(black_dir)
    white_score = _candidate_score(white_summary, candidate_is_white=True)
    black_score = _candidate_score(black_summary, candidate_is_white=False)

    total_games = white_score["games"] + black_score["games"]
    total_wins = white_score["wins"] + black_score["wins"]
    total_draws = white_score["draws"] + black_score["draws"]
    overall_score = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0.0

    return {
        "arena_root": arena_root,
        "candidate_white": white_score,
        "candidate_black": black_score,
        "total_games": total_games,
        "total_wins": total_wins,
        "total_draws": total_draws,
        "score": overall_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Iterate self-play training loop")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--curriculum-games", type=int, default=200,
                        help="Curriculum endgame games per iteration (0 to disable)")
    parser.add_argument("--black-focus-games", type=int, default=0,
                        help="Extra black-focused games from black-advantage starts (black training side only)")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--curriculum-simulations", type=int, default=50,
                        help="MCTS simulations for curriculum games (lower = faster)")
    parser.add_argument("--black-focus-simulations", type=int, default=120,
                        help="MCTS simulations for black-focused start games")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Warmup epochs when fine-tuning (default: from config)")
    parser.add_argument("--warmup-start-factor", type=float, default=None,
                        help="Warmup start LR factor when fine-tuning (default: from config)")
    parser.add_argument("--keep-generations", type=int, default=None,
                        help="Sliding window: keep last N generations (default: from config)")
    parser.add_argument("--position-budget", type=int, default=None,
                        help="Position budget window: include enough recent generations to hit N raw positions")
    parser.add_argument("--alternating", action="store_true",
                        help="Use frozen-opponent alternating training")
    parser.add_argument("--opponent-sims", type=int, default=None,
                        help="MCTS sims for frozen opponent (default: from config)")
    parser.add_argument("--pool-size", type=int, default=5,
                        help="Number of archived models to sample for opponent pool")
    parser.add_argument("--arena-games", type=int, default=50,
                        help="Candidate-vs-incumbent arena games for gating")
    parser.add_argument("--arena-sims", type=int, default=100,
                        help="MCTS simulations per move in arena games")
    parser.add_argument("--arena-workers", type=int, default=2,
                        help="Workers for arena evaluation game generation")
    parser.add_argument("--gate-threshold", type=float, default=0.55,
                        help="Accept candidate if arena score >= threshold")
    parser.add_argument("--gate-min-other-side", type=float, default=0.45,
                        help="Alternating mode: minimum arena score on non-trained side")
    parser.add_argument("--no-gating", action="store_true",
                        help="Disable arena gating and always promote candidate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (default: from config)")
    parser.add_argument("--human-eval", action="store_true",
                        help="Evaluate final incumbent on recorded human games")
    parser.add_argument("--human-eval-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "raw", "human_games"),
                        help="Directory of human game JSONL files for post-run evaluation")
    parser.add_argument("--human-eval-value-diagnostics", action="store_true",
                        help="Include model value diagnostics on human positions (slower)")
    parser.add_argument("--exclude-human-games", action="store_true",
                        help="Exclude data/raw/human_games from training data processing")
    parser.add_argument("--keep-check-positions", action="store_true",
                        help="Keep in-check positions during self-play data generation")
    parser.add_argument("--selfplay-sims-jitter-pct", type=float, default=None,
                        help="Randomize self-play per-game simulations by +/- this fraction (e.g. 0.20)")
    args = parser.parse_args()

    from config import (
        SLIDING_WINDOW, POSITION_BUDGET, OPPONENT_SIMULATIONS, RANDOM_SEED,
        WARMUP_EPOCHS, WARMUP_START_FACTOR, SELFPLAY_SIMS_JITTER_PCT,
    )
    base_seed = args.seed if args.seed is not None else RANDOM_SEED
    set_seed(base_seed)
    keep_gens = args.keep_generations if args.keep_generations is not None else SLIDING_WINDOW
    position_budget = args.position_budget if args.position_budget is not None else POSITION_BUDGET
    selfplay_sims_jitter_pct = (
        args.selfplay_sims_jitter_pct
        if args.selfplay_sims_jitter_pct is not None else SELFPLAY_SIMS_JITTER_PCT
    )
    if position_budget is not None and position_budget <= 0:
        position_budget = None
    opponent_sims = args.opponent_sims if args.opponent_sims is not None else OPPONENT_SIMULATIONS
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else WARMUP_EPOCHS
    warmup_start_factor = (
        args.warmup_start_factor
        if args.warmup_start_factor is not None else WARMUP_START_FACTOR
    )
    if not (0.0 <= args.gate_threshold <= 1.0):
        raise ValueError("--gate-threshold must be in [0, 1]")
    if not (0.0 <= args.gate_min_other_side <= 1.0):
        raise ValueError("--gate-min-other-side must be in [0, 1]")
    if args.pool_size < 0:
        raise ValueError("--pool-size must be >= 0")
    if args.black_focus_games < 0:
        raise ValueError("--black-focus-games must be >= 0")
    if args.black_focus_simulations <= 0:
        raise ValueError("--black-focus-simulations must be > 0")
    if selfplay_sims_jitter_pct < 0.0 or selfplay_sims_jitter_pct >= 1.0:
        raise ValueError("--selfplay-sims-jitter-pct must be in [0.0, 1.0)")
    if args.alternating and not args.no_gating and args.arena_games < 2:
        raise ValueError("--arena-games must be >= 2 for alternating side-aware gating")
    if args.keep_generations is not None and args.position_budget is not None:
        raise ValueError("Specify only one of --keep-generations or --position-budget")

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    model_dir = os.path.join(PROJECT_ROOT, "models")
    model_path = os.path.join(model_dir, "best_value_net.pt")
    frozen_path = os.path.join(model_dir, "frozen_opponent.pt")
    archive_dir = os.path.join(model_dir, "archive")
    archive_manifest = os.path.join(archive_dir, "manifest.jsonl")

    if not os.path.exists(model_path):
        print("No model found - running initial bootstrap (heuristic games + curriculum + train)...\n")
        # Generate heuristic normal games
        run([
            sys.executable, "data_generation.py",
            "--num-games", str(args.games),
            "--simulations", "800",
            "--output-dir", os.path.join(raw_dir, "normal"),
            "--seed", str(base_seed),
        ], f"Bootstrap: generating {args.games} heuristic normal games")
        # Generate heuristic curriculum games (per-tier forced values applied automatically)
        run([
            sys.executable, "data_generation.py",
            "--num-games", str(args.curriculum_games),
            "--simulations", str(args.curriculum_simulations),
            "--output-dir", os.path.join(raw_dir, "curriculum_bootstrap"),
            "--curriculum",
            "--scripted-black",
            "--seed", str(base_seed + 1),
        ], f"Bootstrap: generating {args.curriculum_games} curriculum games (tiered forced values)")
        # Process (no sliding window for bootstrap - only 2 dirs exist)
        run([
            sys.executable, "data_processor.py",
            "--raw-dir", raw_dir,
            "--output-dir", processed_dir,
            "--seed", str(base_seed),
            *(["--exclude-human-games"] if args.exclude_human_games else []),
        ], "Bootstrap: processing all data")
        # Train initial model
        run([
            sys.executable, "train.py",
            "--target", "mcts_value",
            "--epochs", str(args.epochs),
            "--data-dir", processed_dir,
            "--model-dir", model_dir,
            "--seed", str(base_seed),
        ], "Bootstrap: training initial model")
        print("\n  Bootstrap complete. Starting iteration loop.\n")

    _seed_archive_if_empty(model_path, archive_dir, archive_manifest)

    start_gen = find_next_gen(raw_dir)
    total_games_existing, total_pos_existing = count_data(raw_dir)
    loop_start = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_metadata = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": get_git_commit(),
        "seed": base_seed,
        "args": vars(args),
        "warmup_epochs_effective": warmup_epochs,
        "warmup_start_factor_effective": warmup_start_factor,
        "gating_enabled": not args.no_gating,
        "gate_threshold": args.gate_threshold,
        "gate_min_other_side": args.gate_min_other_side,
        "arena_games": args.arena_games,
        "arena_sims": args.arena_sims,
        "black_focus_games": args.black_focus_games,
        "black_focus_simulations": args.black_focus_simulations,
        "opponent_pool_size": args.pool_size,
        "position_budget_effective": position_budget,
        "selfplay_sims_jitter_pct": selfplay_sims_jitter_pct,
        "archive_dir": archive_dir,
        "human_eval_enabled": args.human_eval,
        "human_eval_dir": args.human_eval_dir,
        "human_eval_value_diagnostics": args.human_eval_value_diagnostics,
        "exclude_human_games": args.exclude_human_games,
        "keep_check_positions": args.keep_check_positions,
        "mode": "alternating" if args.alternating else "standard",
        "start_generation": start_gen,
        "model_path": model_path,
        "iterations": [],
    }

    mode_str = "ALTERNATING" if args.alternating else "STANDARD"

    print(f"\n{'#'*60}")
    print(f"  SELF-PLAY ITERATION LOOP ({mode_str})")
    print(f"{'#'*60}")
    print(f"  Iterations:  {args.iterations}")
    print(f"  Games/iter:  {args.games} normal + {args.curriculum_games} curriculum + {args.black_focus_games} black-focus")
    print(f"  Simulations: {args.simulations} normal, {args.curriculum_simulations} curriculum, {args.black_focus_simulations} black-focus")
    if selfplay_sims_jitter_pct > 0:
        print(f"  Sim jitter:  +/- {100 * selfplay_sims_jitter_pct:.0f}% (self-play generation only)")
    if args.alternating:
        print(f"  Opponent:    {opponent_sims} sims (pool/frozen fallback)")
        print(f"  Pool size:   {args.pool_size} archived models")
    if args.no_gating:
        print("  Gating:      disabled (auto-promote candidates)")
    else:
        print(f"  Gating:      arena {args.arena_games} games @ {args.arena_sims} sims, threshold={args.gate_threshold:.2f}")
        if args.alternating:
            print(f"               alternating side floor={args.gate_min_other_side:.2f} on non-trained side")
    print(f"  Epochs:      {args.epochs}")
    if position_budget is not None:
        budget_human = "no human_games" if args.exclude_human_games else "with human_games"
        print(f"  Window:      position budget {position_budget} raw positions (+ curriculum_bootstrap, {budget_human})")
    else:
        if args.exclude_human_games:
            print(f"  Window:      last {keep_gens} generations (+ curriculum_bootstrap, no human_games)")
        else:
            print(f"  Window:      last {keep_gens} generations (+ curriculum_bootstrap + human_games)")
    print(f"  Check pos:   {'keep' if args.keep_check_positions else 'skip'}")
    print(f"  Starting at: generation {start_gen}")
    print(f"  Existing:    {total_games_existing} games, {total_pos_existing} positions")
    print(f"  Model:       {model_path}")
    print(f"  Seed:        {base_seed}")
    print(f"  Fine-tune:   warmup_epochs={warmup_epochs}, warmup_start_factor={warmup_start_factor}")
    if args.human_eval:
        diag_label = "on" if args.human_eval_value_diagnostics else "off"
        print(f"  Human eval:  enabled (dir={args.human_eval_dir}, value_diag={diag_label})")
    print()

    for i in range(args.iterations):
        gen = start_gen + i
        gen_dir = os.path.join(raw_dir, f"nn_gen{gen}")
        iter_start = time.time()

        # Determine training side for alternating mode
        if args.alternating:
            # Odd iterations train Black, even train White
            train_side = "black" if (i % 2 == 0) else "white"
            side_label = train_side.upper()
        else:
            train_side = "both"
            side_label = "BOTH"

        print(f"\n{'#'*60}")
        print(f"  ITERATION {i+1}/{args.iterations}  -  Generation {gen}  -  Training: {side_label}")
        print(f"{'#'*60}")

        # Step 1a: Generate normal games
        opponent_pool_count = 0
        def _sim_bounds(base_sims):
            if selfplay_sims_jitter_pct <= 0:
                return None
            min_sims = max(1, int(round(base_sims * (1.0 - selfplay_sims_jitter_pct))))
            max_sims = max(min_sims, int(round(base_sims * (1.0 + selfplay_sims_jitter_pct))))
            return min_sims, max_sims

        gen_cmd = [
            sys.executable, "data_generation.py",
            "--num-games", str(args.games),
            "--simulations", str(args.simulations),
            "--output-dir", gen_dir,
            "--use-model", model_path,
            "--seed", str(base_seed + gen * 10 + 1),
        ]
        normal_sim_bounds = _sim_bounds(args.simulations)
        if normal_sim_bounds is not None:
            gen_cmd.extend([
                "--simulations-min", str(normal_sim_bounds[0]),
                "--simulations-max", str(normal_sim_bounds[1]),
            ])
        if args.keep_check_positions:
            gen_cmd.append("--keep-check-positions")

        if args.alternating:
            gen_cmd.extend(["--train-side", train_side])
            gen_cmd.extend(["--opponent-sims", str(opponent_sims)])
            archived_models = _list_archive_models(archive_dir)
            opponent_pool_count = min(len(archived_models), args.pool_size) if args.pool_size > 0 else 0
            if args.pool_size > 0 and archived_models:
                gen_cmd.extend(["--opponent-pool-dir", archive_dir])
                gen_cmd.extend(["--opponent-pool-size", str(args.pool_size)])
            # Fallback: single frozen model if pool is unavailable
            elif os.path.exists(frozen_path):
                gen_cmd.extend(["--opponent-model", frozen_path])

        t_gen = run(gen_cmd,
            f"[{i+1}/{args.iterations}] Generating {args.games} normal games "
            f"(gen {gen}, training {side_label})")

        print(f"\n  --- Normal Games ---")
        normal_summary = summarize_generation(gen_dir)

        # Step 1b: Generate extra black-focused games from black-advantage starts
        t_bf = 0.0
        black_focus_summary = None
        if args.black_focus_games > 0 and args.alternating and train_side == "black":
            bf_dir = os.path.join(raw_dir, f"nn_gen{gen}_blackfocus")
            bf_cmd = [
                sys.executable, "data_generation.py",
                "--num-games", str(args.black_focus_games),
                "--simulations", str(args.black_focus_simulations),
                "--output-dir", bf_dir,
                "--use-model", model_path,
                "--curriculum",
                "--curriculum-live-results",
                "--train-side", "black",
                "--opponent-sims", str(opponent_sims),
                "--seed", str(base_seed + gen * 10 + 5),
            ]
            blackfocus_sim_bounds = _sim_bounds(args.black_focus_simulations)
            if blackfocus_sim_bounds is not None:
                bf_cmd.extend([
                    "--simulations-min", str(blackfocus_sim_bounds[0]),
                    "--simulations-max", str(blackfocus_sim_bounds[1]),
                ])
            if args.keep_check_positions:
                bf_cmd.append("--keep-check-positions")
            if args.pool_size > 0 and archived_models:
                bf_cmd.extend(["--opponent-pool-dir", archive_dir])
                bf_cmd.extend(["--opponent-pool-size", str(args.pool_size)])
            elif os.path.exists(frozen_path):
                bf_cmd.extend(["--opponent-model", frozen_path])

            t_bf = run(
                bf_cmd,
                f"[{i+1}/{args.iterations}] Generating {args.black_focus_games} black-focus games "
                f"@ {args.black_focus_simulations} sims (gen {gen}, live outcomes)"
            )
            print(f"\n  --- Black-Focus Games ---")
            black_focus_summary = summarize_generation(bf_dir)
            t_gen += t_bf

        # Step 1c: Generate curriculum endgame games
        t_cur = 0.0
        if args.curriculum_games > 0:
            cur_dir = os.path.join(raw_dir, f"nn_gen{gen}_curriculum")
            t_cur = run([
                sys.executable, "data_generation.py",
                "--num-games", str(args.curriculum_games),
                "--simulations", str(args.curriculum_simulations),
                "--output-dir", cur_dir,
                "--use-model", model_path,
                "--curriculum",
                "--scripted-black",
                "--seed", str(base_seed + gen * 10 + 2),
                *(
                    [
                        "--simulations-min", str(_sim_bounds(args.curriculum_simulations)[0]),
                        "--simulations-max", str(_sim_bounds(args.curriculum_simulations)[1]),
                    ] if _sim_bounds(args.curriculum_simulations) is not None else []
                ),
                *(["--keep-check-positions"] if args.keep_check_positions else []),
            ], f"[{i+1}/{args.iterations}] Generating {args.curriculum_games} curriculum games @ {args.curriculum_simulations} sims (gen {gen}, tiered values)")

            print(f"\n  --- Curriculum Games ---")
            curriculum_summary = summarize_generation(cur_dir)
            t_gen += t_cur
        else:
            curriculum_summary = None

        # Step 2: Reprocess data (sliding window)
        total_games, total_pos = count_data(raw_dir)
        proc_cmd = [
            sys.executable, "data_processor.py",
            "--raw-dir", raw_dir,
            "--output-dir", processed_dir,
            "--seed", str(base_seed + gen * 10 + 3),
            *(["--exclude-human-games"] if args.exclude_human_games else []),
        ]
        if position_budget is not None:
            proc_cmd.extend(["--position-budget", str(position_budget)])
            proc_desc = (
                f"[{i+1}/{args.iterations}] Reprocessing data "
                f"(position_budget={position_budget}, total on disk: {total_games} games)"
            )
        else:
            proc_cmd.extend(["--keep-generations", str(keep_gens)])
            proc_desc = (
                f"[{i+1}/{args.iterations}] Reprocessing data "
                f"(window={keep_gens}, total on disk: {total_games} games)"
            )
        t_proc = run(proc_cmd, proc_desc)

        # Step 3: Train candidate model (fine-tune from incumbent)
        candidate_dir = os.path.join(model_dir, "candidates", f"gen_{gen:04d}")
        candidate_path = os.path.join(candidate_dir, "best_value_net.pt")
        if os.path.exists(candidate_dir):
            shutil.rmtree(candidate_dir)
        os.makedirs(candidate_dir, exist_ok=True)
        train_cmd = [
            sys.executable, "train.py",
            "--target", "mcts_value",
            "--epochs", str(args.epochs),
            "--data-dir", processed_dir,
            "--model-dir", candidate_dir,
            "--seed", str(base_seed + gen * 10 + 4),
            "--warmup-epochs", str(warmup_epochs),
            "--warmup-start-factor", str(warmup_start_factor),
        ]
        if os.path.exists(model_path):
            train_cmd.extend(["--resume-from", model_path])
        t_train = run(train_cmd, f"[{i+1}/{args.iterations}] Training candidate model")

        # Step 4: Gate candidate vs incumbent and promote if accepted
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"Candidate model missing after training: {candidate_path}")
        if args.no_gating:
            gate_info = {
                "enabled": False,
                "accepted": True,
                "score": None,
                "threshold": args.gate_threshold,
                "min_other_side": args.gate_min_other_side,
                "total_games": 0,
                "decision_mode": "disabled",
            }
            accepted = True
        else:
            gate_info = _run_arena(
                candidate_model=candidate_path,
                incumbent_model=model_path,
                gen=gen,
                model_dir=model_dir,
                arena_games=args.arena_games,
                arena_sims=args.arena_sims,
                arena_workers=args.arena_workers,
                base_seed=base_seed,
            )
            gate_info["enabled"] = True
            gate_info["threshold"] = args.gate_threshold
            gate_info["min_other_side"] = args.gate_min_other_side
            if args.alternating:
                primary_side = train_side
                other_side = "white" if train_side == "black" else "black"
                primary_info = gate_info[f"candidate_{primary_side}"]
                other_info = gate_info[f"candidate_{other_side}"]
                gate_info.update({
                    "decision_mode": "side_aware",
                    "primary_side": primary_side,
                    "other_side": other_side,
                    "primary_score": primary_info["score"],
                    "other_score": other_info["score"],
                    "primary_games": primary_info["games"],
                    "other_games": other_info["games"],
                })
                accepted = (
                    gate_info["total_games"] > 0
                    and gate_info["primary_games"] > 0
                    and gate_info["other_games"] > 0
                    and gate_info["primary_score"] >= args.gate_threshold
                    and gate_info["other_score"] >= args.gate_min_other_side
                )
            else:
                gate_info.update({
                    "decision_mode": "overall",
                    "primary_side": "overall",
                    "other_side": None,
                    "primary_score": gate_info["score"],
                    "other_score": None,
                    "primary_games": gate_info["total_games"],
                    "other_games": 0,
                })
                accepted = gate_info["score"] >= args.gate_threshold and gate_info["total_games"] > 0
            gate_info["accepted"] = accepted

        if accepted:
            shutil.copy2(candidate_path, model_path)
            archive_path = os.path.join(archive_dir, f"gen_{gen:04d}.pt")
            shutil.copy2(candidate_path, archive_path)
            print(f"\n  Candidate ACCEPTED and promoted -> {model_path}")
            print(f"  Archived accepted model -> {archive_path}")
            if args.alternating:
                shutil.copy2(model_path, frozen_path)
                print(f"  Frozen fallback updated -> {frozen_path}")
        else:
            if gate_info.get("decision_mode") == "side_aware":
                print(
                    f"\n  Candidate REJECTED "
                    f"(trained-{gate_info['primary_side']} score={gate_info['primary_score']:.3f} "
                    f"vs {args.gate_threshold:.3f}, other-{gate_info['other_side']} "
                    f"score={gate_info['other_score']:.3f} vs {args.gate_min_other_side:.3f}); "
                    f"keeping incumbent"
                )
            else:
                print(f"\n  Candidate REJECTED (score={gate_info['score']:.3f} < {args.gate_threshold:.3f}); keeping incumbent")

        _append_jsonl(archive_manifest, {
            "event": "gate_result",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generation": gen,
            "candidate_path": candidate_path,
            "incumbent_path": model_path,
            "accepted": accepted,
            "gate": gate_info,
        })

        iter_elapsed = time.time() - iter_start
        total_elapsed = time.time() - loop_start
        remaining = (args.iterations - i - 1) * iter_elapsed
        run_metadata["iterations"].append({
            "iteration": i + 1,
            "generation": gen,
            "train_side": train_side,
            "opponent_pool_count": opponent_pool_count,
            "normal_generation": normal_summary,
            "black_focus_generation": black_focus_summary,
            "curriculum_generation": curriculum_summary,
            "candidate_model_path": candidate_path,
            "accepted": accepted,
            "gate": gate_info,
            "timings_seconds": {
                "generate_total": float(t_gen),
                "process": float(t_proc),
                "train": float(t_train),
                "iteration_total": float(iter_elapsed),
            },
        })

        print(f"\n{'-'*60}")
        gate_label = "ACCEPTED" if accepted else "REJECTED"
        print(f"  Iteration {i+1}/{args.iterations} complete (trained {side_label}, candidate {gate_label})")
        print(f"  Time:  generate {format_time(t_gen)}  |  process {format_time(t_proc)}  |  train {format_time(t_train)}  |  total {format_time(iter_elapsed)}")
        if gate_info.get("enabled"):
            if gate_info.get("decision_mode") == "side_aware":
                print(
                    f"  Gate:  trained-{gate_info['primary_side']}={gate_info['primary_score']:.3f} "
                    f"(>={args.gate_threshold:.3f})  other-{gate_info['other_side']}={gate_info['other_score']:.3f} "
                    f"(>={args.gate_min_other_side:.3f})  overall={gate_info['score']:.3f}  games={gate_info['total_games']}"
                )
            else:
                print(f"  Gate:  score={gate_info['score']:.3f}  threshold={args.gate_threshold:.3f}  games={gate_info['total_games']}")
        print(f"  Clock: {format_time(total_elapsed)} elapsed  |  ~{format_time(remaining)} remaining")
        print(f"{'-'*60}")

    total_elapsed = time.time() - loop_start
    total_games, total_pos = count_data(raw_dir)
    print(f"\n{'#'*60}")
    print(f"  ALL DONE - {args.iterations} iterations in {format_time(total_elapsed)}")
    print(f"  Total data: {total_games} games, {total_pos} positions")
    print(f"  Model: {model_path}")
    print(f"{'#'*60}\n")


    run_metadata["end_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_metadata["total_elapsed_seconds"] = float(total_elapsed)
    run_metadata["final_data_totals"] = {
        "games": int(total_games),
        "positions": int(total_pos),
    }
    if args.human_eval:
        try:
            from human_eval import evaluate_human_games

            human_eval_summary = evaluate_human_games(
                human_dir=args.human_eval_dir,
                model_path=model_path if args.human_eval_value_diagnostics else None,
                include_value_diagnostics=args.human_eval_value_diagnostics,
            )
            run_metadata["final_human_eval"] = human_eval_summary

            ai_black = human_eval_summary["by_ai_side"]["black"]
            human_black = human_eval_summary["by_human_side"]["black"]
            print("\n  --- Human Eval ---")
            print(
                f"  AI as Black: games={ai_black['games']}  "
                f"W/L/D={ai_black['wins']}/{ai_black['losses']}/{ai_black['draws']}  "
                f"score={ai_black['score']:.3f}"
            )
            print(
                f"  Human as Black vs AI-White: games={human_black['games']}  "
                f"W/L/D={human_black['wins']}/{human_black['losses']}/{human_black['draws']}  "
                f"score={human_black['score']:.3f}"
            )
        except Exception as e:
            run_metadata["final_human_eval"] = {"error": str(e)}
            print(f"\n  WARNING: human evaluation failed: {e}")
    os.makedirs(model_dir, exist_ok=True)
    run_meta_path = os.path.join(model_dir, f"iterate_run_{run_id}.json")
    with open(run_meta_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Iteration metadata saved to {run_meta_path}")

if __name__ == "__main__":
    main()

