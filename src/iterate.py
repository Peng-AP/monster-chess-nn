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
               arena_sims, arena_workers, base_seed, arena_tag="standard",
               curriculum=False, curriculum_live_results=False,
               curriculum_tier_min=None, curriculum_tier_max=None):
    """Evaluate candidate vs incumbent with color swap. Returns score dict."""
    arena_root = os.path.join(model_dir, "arena_runs", f"gen_{gen:04d}", arena_tag)
    white_dir = os.path.join(arena_root, "candidate_white")
    black_dir = os.path.join(arena_root, "candidate_black")
    if os.path.exists(arena_root):
        shutil.rmtree(arena_root)
    os.makedirs(white_dir, exist_ok=True)
    os.makedirs(black_dir, exist_ok=True)

    n_white = arena_games // 2
    n_black = arena_games - n_white

    if n_white > 0:
        white_cmd = [
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
        ]
        if curriculum:
            white_cmd.append("--curriculum")
            if curriculum_live_results:
                white_cmd.append("--curriculum-live-results")
            if curriculum_tier_min is not None:
                white_cmd.extend(["--curriculum-tier-min", str(curriculum_tier_min)])
            if curriculum_tier_max is not None:
                white_cmd.extend(["--curriculum-tier-max", str(curriculum_tier_max)])
        run(
            white_cmd,
            f"Arena[{arena_tag}] gen {gen}: candidate as White vs incumbent ({n_white} games)",
        )

    if n_black > 0:
        black_cmd = [
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
        ]
        if curriculum:
            black_cmd.append("--curriculum")
            if curriculum_live_results:
                black_cmd.append("--curriculum-live-results")
            if curriculum_tier_min is not None:
                black_cmd.extend(["--curriculum-tier-min", str(curriculum_tier_min)])
            if curriculum_tier_max is not None:
                black_cmd.extend(["--curriculum-tier-max", str(curriculum_tier_max)])
        run(
            black_cmd,
            f"Arena[{arena_tag}] gen {gen}: candidate as Black vs incumbent ({n_black} games)",
        )

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


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _compute_adaptive_mix(base_normal_games, base_curriculum_games, base_black_focus_games,
                          train_side, alternating, adaptive_enabled, side_scales,
                          min_normal_games):
    """Compute effective per-iteration game mix under adaptive curriculum scheduling."""
    if not adaptive_enabled:
        black_focus = base_black_focus_games if (alternating and train_side == "black") else 0
        return {
            "scale_key": None,
            "scale_value": 1.0,
            "normal_games": base_normal_games,
            "curriculum_games": base_curriculum_games,
            "black_focus_games": black_focus,
            "base_total_games": base_normal_games + base_curriculum_games + black_focus,
            "effective_total_games": base_normal_games + base_curriculum_games + black_focus,
        }

    scale_key = train_side if alternating else "both"
    scale_value = side_scales.get(scale_key, 1.0)

    curriculum_games = int(round(base_curriculum_games * scale_value))
    black_focus_games = (
        int(round(base_black_focus_games * scale_value))
        if (alternating and train_side == "black")
        else 0
    )

    base_total = base_normal_games + base_curriculum_games + (
        base_black_focus_games if (alternating and train_side == "black") else 0
    )
    normal_games = max(0, base_total - curriculum_games - black_focus_games)

    # Keep a floor of normal games to preserve opening/midgame diversity.
    normal_floor = max(0, min_normal_games)
    if base_total >= normal_floor and normal_games < normal_floor:
        deficit = normal_floor - normal_games
        normal_games = normal_floor
        while deficit > 0 and (curriculum_games > 0 or black_focus_games > 0):
            if black_focus_games >= curriculum_games and black_focus_games > 0:
                black_focus_games -= 1
            elif curriculum_games > 0:
                curriculum_games -= 1
            else:
                break
            deficit -= 1

    return {
        "scale_key": scale_key,
        "scale_value": float(scale_value),
        "normal_games": int(normal_games),
        "curriculum_games": int(curriculum_games),
        "black_focus_games": int(black_focus_games),
        "base_total_games": int(base_total),
        "effective_total_games": int(normal_games + curriculum_games + black_focus_games),
    }


def _update_adaptive_scale(current_scale, accepted, gate_info,
                           up_factor, down_factor, min_scale, max_scale):
    """Update side scale from gate outcome; returns (new_scale, reason)."""
    if accepted:
        proposed = current_scale * down_factor
        reason = "accepted_reduce_aux"
    else:
        # If non-trained side collapsed, reduce aux emphasis; otherwise increase.
        if gate_info.get("decision_mode") in ("side_aware", "side_aware_black_focus"):
            other_score = gate_info.get("other_score")
            other_floor = gate_info.get("min_other_side")
            if (
                other_score is not None
                and other_floor is not None
                and other_score < other_floor
            ):
                proposed = current_scale * down_factor
                reason = "other_side_under_floor_reduce_aux"
            else:
                proposed = current_scale * up_factor
                reason = "trained_side_miss_increase_aux"
        else:
            proposed = current_scale * up_factor
            reason = "rejected_increase_aux"

    new_scale = _clamp(proposed, min_scale, max_scale)
    return float(new_scale), reason


def main():
    parser = argparse.ArgumentParser(description="Iterate self-play training loop")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--curriculum-games", type=int, default=200,
                        help="Curriculum endgame games per iteration (0 to disable)")
    parser.add_argument("--black-focus-games", type=int, default=0,
                        help="Extra black-focused games from black-advantage starts (black training side only)")
    parser.add_argument("--human-seed-games", type=int, default=0,
                        help="Extra games from human-recorded start positions each iteration")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--curriculum-simulations", type=int, default=50,
                        help="MCTS simulations for curriculum games (lower = faster)")
    parser.add_argument("--black-focus-simulations", type=int, default=120,
                        help="MCTS simulations for black-focused start games")
    parser.add_argument("--black-focus-tier-min", type=int, default=None,
                        help="Curriculum tier lower bound for black-focus generation (1-indexed)")
    parser.add_argument("--black-focus-tier-max", type=int, default=None,
                        help="Curriculum tier upper bound for black-focus generation (1-indexed)")
    parser.add_argument("--human-seed-simulations", type=int, default=None,
                        help="MCTS simulations for human-seeded games (default: normal sims)")
    parser.add_argument("--black-train-sims-mult", type=float, default=1.0,
                        help="Multiplier on train-side simulations when training Black in alternating mode")
    parser.add_argument("--black-opponent-sims-mult", type=float, default=1.0,
                        help="Multiplier on opponent simulations when training Black in alternating mode")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--primary-no-resume", action="store_true",
                        help="Do not initialize primary training from incumbent checkpoint")
    parser.add_argument("--train-target", type=str, default="mcts_value",
                        choices=["mcts_value", "game_result", "blend"],
                        help="Training target passed to train.py")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Warmup epochs when fine-tuning (default: from config)")
    parser.add_argument("--warmup-start-factor", type=float, default=None,
                        help="Warmup start LR factor when fine-tuning (default: from config)")
    parser.add_argument("--keep-generations", type=int, default=None,
                        help="Sliding window: keep last N generations (default: from config)")
    parser.add_argument("--position-budget", type=int, default=None,
                        help="Position budget window: include enough recent generations to hit N raw positions")
    parser.add_argument("--position-budget-max", type=int, default=None,
                        help="Optional max-cap for position budget window (requires --position-budget)")
    parser.add_argument("--alternating", action="store_true",
                        help="Use frozen-opponent alternating training")
    parser.add_argument(
        "--alternating-pattern",
        type=str,
        default="alternate",
        choices=["alternate", "black_only", "white_only"],
        help="Alternating mode side schedule: alternate each iteration, or lock to one side",
    )
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
    parser.add_argument("--gate-min-other-side-white", type=float, default=None,
                        help="Alternating White training: min non-trained-side score (default: --gate-min-other-side)")
    parser.add_argument("--gate-min-other-side-black", type=float, default=None,
                        help="Alternating Black training: min non-trained-side score (default: --gate-min-other-side)")
    parser.add_argument("--black-focus-arena-games", type=int, default=0,
                        help="Extra black-focus arena games (curriculum live starts) for Black-side gating")
    parser.add_argument("--black-focus-arena-sims", type=int, default=None,
                        help="MCTS simulations per move for black-focus arena games (default: --arena-sims)")
    parser.add_argument("--black-focus-gate-threshold", type=float, default=0.40,
                        help="Alternating Black mode: accept if black-focus arena score >= this threshold")
    parser.add_argument("--black-focus-arena-tier-min", type=int, default=None,
                        help="Curriculum tier lower bound for black-focus arena gating (1-indexed)")
    parser.add_argument("--black-focus-arena-tier-max", type=int, default=None,
                        help="Curriculum tier upper bound for black-focus arena gating (1-indexed)")
    parser.add_argument("--no-gating", action="store_true",
                        help="Disable arena gating and always promote candidate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (default: from config)")
    parser.add_argument("--human-eval", action="store_true",
                        help="Evaluate final incumbent on recorded human games")
    parser.add_argument("--human-eval-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "raw", "human_games"),
                        help="Directory of human game JSONL files for post-run evaluation")
    parser.add_argument("--human-seed-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "raw", "human_games"),
                        help="Directory of human game JSONL files used as start-position source")
    parser.add_argument("--human-seed-side", type=str, default="auto",
                        choices=["auto", "any", "white", "black"],
                        help="Human-seed start side filter (auto=training side in alternating mode)")
    parser.add_argument("--human-seed-max-positions", type=int, default=2000,
                        help="Cap loaded human start positions per generation (0 = no cap)")
    parser.add_argument("--human-eval-value-diagnostics", action="store_true",
                        help="Include model value diagnostics on human positions (slower)")
    parser.add_argument("--exclude-human-games", action="store_true",
                        help="Exclude data/raw/human_games from training data processing")
    parser.add_argument("--min-blackfocus-plies", type=int, default=0,
                        help="Drop non-human _blackfocus games shorter than this many plies during processing")
    parser.add_argument("--keep-check-positions", action="store_true",
                        help="Keep in-check positions during self-play data generation")
    parser.add_argument("--selfplay-sims-jitter-pct", type=float, default=None,
                        help="Randomize self-play per-game simulations by +/- this fraction (e.g. 0.20)")
    parser.add_argument("--adaptive-curriculum", action="store_true",
                        help="Dynamically rebalance normal/curriculum/black-focus mix from recent gate outcomes")
    parser.add_argument("--adaptive-min-scale", type=float, default=0.70,
                        help="Lower bound for adaptive curriculum scaling")
    parser.add_argument("--adaptive-max-scale", type=float, default=1.80,
                        help="Upper bound for adaptive curriculum scaling")
    parser.add_argument("--adaptive-up-factor", type=float, default=1.20,
                        help="Scale multiplier after rejected candidate (increase aux focus)")
    parser.add_argument("--adaptive-down-factor", type=float, default=0.92,
                        help="Scale multiplier after accepted candidate (decrease aux focus)")
    parser.add_argument("--adaptive-min-normal-games", type=int, default=80,
                        help="Minimum normal games per iteration when adaptive curriculum is enabled")
    parser.add_argument("--use-se-blocks", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable SE modules in training backbone (default: from config)")
    parser.add_argument("--se-reduction", type=int, default=None,
                        help="SE channel reduction ratio for training backbone (default: from config)")
    parser.add_argument("--use-side-specialized-heads", action=argparse.BooleanOptionalAction, default=None,
                        help="Use side-specialized value/policy heads in training model (default: from config)")
    parser.add_argument("--consolidation-epochs", type=int, default=2,
                        help="Post-train consolidation epochs on balanced mixed-side data (0 to disable)")
    parser.add_argument("--consolidation-lr-factor", type=float, default=0.35,
                        help="Multiplier on base LR for consolidation phase")
    parser.add_argument("--consolidation-batch-size", type=int, default=192,
                        help="Batch size for consolidation phase")
    parser.add_argument("--consolidation-balance-sides", action=argparse.BooleanOptionalAction, default=True,
                        help="Use side-balanced sampling during consolidation phase")
    parser.add_argument("--consolidation-balanced-black-ratio", type=float, default=0.5,
                        help="Black-to-move fraction when consolidation side balancing is enabled")
    parser.add_argument("--consolidation-distill-value-weight", type=float, default=0.20,
                        help="Value distillation weight in consolidation phase")
    parser.add_argument("--consolidation-distill-policy-weight", type=float, default=0.60,
                        help="Policy distillation weight in consolidation phase")
    parser.add_argument("--consolidation-distill-temperature", type=float, default=1.0,
                        help="Distillation temperature in consolidation phase")
    parser.add_argument("--primary-balance-sides", action=argparse.BooleanOptionalAction, default=False,
                        help="Use side-balanced sampling during primary training phase")
    parser.add_argument("--primary-balanced-black-ratio", type=float, default=0.5,
                        help="Black-to-move fraction when primary side balancing is enabled")
    parser.add_argument("--primary-train-only-side", type=str, default="auto",
                        choices=["auto", "none", "white", "black"],
                        help="Restrict primary training data to one side-to-move (auto=training side in alternating)")
    parser.add_argument("--consolidation-train-only-side", type=str, default="auto",
                        choices=["auto", "none", "white", "black"],
                        help="Restrict consolidation training data to one side-to-move (auto=training side in alternating)")
    args = parser.parse_args()

    from config import (
        SLIDING_WINDOW, POSITION_BUDGET, POSITION_BUDGET_MAX, OPPONENT_SIMULATIONS, RANDOM_SEED,
        WARMUP_EPOCHS, WARMUP_START_FACTOR, SELFPLAY_SIMS_JITTER_PCT,
        LEARNING_RATE,
        USE_SE_BLOCKS, SE_REDUCTION, USE_SIDE_SPECIALIZED_HEADS,
    )
    base_seed = args.seed if args.seed is not None else RANDOM_SEED
    set_seed(base_seed)
    keep_gens = args.keep_generations if args.keep_generations is not None else SLIDING_WINDOW
    position_budget = args.position_budget if args.position_budget is not None else POSITION_BUDGET
    position_budget_max = (
        args.position_budget_max if args.position_budget_max is not None else POSITION_BUDGET_MAX
    )
    selfplay_sims_jitter_pct = (
        args.selfplay_sims_jitter_pct
        if args.selfplay_sims_jitter_pct is not None else SELFPLAY_SIMS_JITTER_PCT
    )
    use_se_blocks = args.use_se_blocks if args.use_se_blocks is not None else USE_SE_BLOCKS
    se_reduction = args.se_reduction if args.se_reduction is not None else SE_REDUCTION
    use_side_specialized_heads = (
        args.use_side_specialized_heads
        if args.use_side_specialized_heads is not None else USE_SIDE_SPECIALIZED_HEADS
    )
    if position_budget is not None and position_budget <= 0:
        position_budget = None
    if position_budget_max is not None and position_budget_max <= 0:
        position_budget_max = None
    opponent_sims = args.opponent_sims if args.opponent_sims is not None else OPPONENT_SIMULATIONS
    black_focus_arena_sims = (
        args.black_focus_arena_sims
        if args.black_focus_arena_sims is not None else args.arena_sims
    )
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else WARMUP_EPOCHS
    warmup_start_factor = (
        args.warmup_start_factor
        if args.warmup_start_factor is not None else WARMUP_START_FACTOR
    )
    gate_min_other_side_white = (
        args.gate_min_other_side_white
        if args.gate_min_other_side_white is not None else args.gate_min_other_side
    )
    gate_min_other_side_black = (
        args.gate_min_other_side_black
        if args.gate_min_other_side_black is not None else args.gate_min_other_side
    )
    if not (0.0 <= args.gate_threshold <= 1.0):
        raise ValueError("--gate-threshold must be in [0, 1]")
    if not (0.0 <= args.gate_min_other_side <= 1.0):
        raise ValueError("--gate-min-other-side must be in [0, 1]")
    if not (0.0 <= gate_min_other_side_white <= 1.0):
        raise ValueError("--gate-min-other-side-white must be in [0, 1]")
    if not (0.0 <= gate_min_other_side_black <= 1.0):
        raise ValueError("--gate-min-other-side-black must be in [0, 1]")
    if not (0.0 <= args.black_focus_gate_threshold <= 1.0):
        raise ValueError("--black-focus-gate-threshold must be in [0, 1]")
    if args.pool_size < 0:
        raise ValueError("--pool-size must be >= 0")
    if args.black_focus_games < 0:
        raise ValueError("--black-focus-games must be >= 0")
    if args.black_focus_simulations <= 0:
        raise ValueError("--black-focus-simulations must be > 0")
    if args.black_focus_tier_min is not None and args.black_focus_tier_min < 1:
        raise ValueError("--black-focus-tier-min must be >= 1")
    if args.black_focus_tier_max is not None and args.black_focus_tier_max < 1:
        raise ValueError("--black-focus-tier-max must be >= 1")
    if (
        args.black_focus_tier_min is not None
        and args.black_focus_tier_max is not None
        and args.black_focus_tier_min > args.black_focus_tier_max
    ):
        raise ValueError("--black-focus-tier-min must be <= --black-focus-tier-max")
    if args.human_seed_games < 0:
        raise ValueError("--human-seed-games must be >= 0")
    if args.human_seed_simulations is not None and args.human_seed_simulations <= 0:
        raise ValueError("--human-seed-simulations must be > 0")
    if args.human_seed_max_positions < 0:
        raise ValueError("--human-seed-max-positions must be >= 0")
    if args.black_focus_arena_games < 0:
        raise ValueError("--black-focus-arena-games must be >= 0")
    if black_focus_arena_sims <= 0:
        raise ValueError("--black-focus-arena-sims must be > 0")
    if args.black_focus_arena_tier_min is not None and args.black_focus_arena_tier_min < 1:
        raise ValueError("--black-focus-arena-tier-min must be >= 1")
    if args.black_focus_arena_tier_max is not None and args.black_focus_arena_tier_max < 1:
        raise ValueError("--black-focus-arena-tier-max must be >= 1")
    if (
        args.black_focus_arena_tier_min is not None
        and args.black_focus_arena_tier_max is not None
        and args.black_focus_arena_tier_min > args.black_focus_arena_tier_max
    ):
        raise ValueError("--black-focus-arena-tier-min must be <= --black-focus-arena-tier-max")
    if args.black_train_sims_mult <= 0:
        raise ValueError("--black-train-sims-mult must be > 0")
    if args.black_opponent_sims_mult <= 0:
        raise ValueError("--black-opponent-sims-mult must be > 0")
    if selfplay_sims_jitter_pct < 0.0 or selfplay_sims_jitter_pct >= 1.0:
        raise ValueError("--selfplay-sims-jitter-pct must be in [0.0, 1.0)")
    if args.adaptive_min_scale <= 0:
        raise ValueError("--adaptive-min-scale must be > 0")
    if args.adaptive_max_scale < args.adaptive_min_scale:
        raise ValueError("--adaptive-max-scale must be >= --adaptive-min-scale")
    if args.adaptive_up_factor <= 0:
        raise ValueError("--adaptive-up-factor must be > 0")
    if args.adaptive_down_factor <= 0:
        raise ValueError("--adaptive-down-factor must be > 0")
    if args.adaptive_min_normal_games < 0:
        raise ValueError("--adaptive-min-normal-games must be >= 0")
    if args.min_blackfocus_plies < 0:
        raise ValueError("--min-blackfocus-plies must be >= 0")
    if args.human_seed_games > 0 and not os.path.isdir(args.human_seed_dir):
        raise FileNotFoundError(f"--human-seed-dir not found: {args.human_seed_dir}")
    if se_reduction <= 0:
        raise ValueError("--se-reduction must be > 0")
    if args.consolidation_epochs < 0:
        raise ValueError("--consolidation-epochs must be >= 0")
    if args.consolidation_lr_factor <= 0:
        raise ValueError("--consolidation-lr-factor must be > 0")
    if args.consolidation_batch_size <= 0:
        raise ValueError("--consolidation-batch-size must be > 0")
    if args.consolidation_distill_value_weight < 0:
        raise ValueError("--consolidation-distill-value-weight must be >= 0")
    if args.consolidation_distill_policy_weight < 0:
        raise ValueError("--consolidation-distill-policy-weight must be >= 0")
    if args.consolidation_distill_temperature <= 0:
        raise ValueError("--consolidation-distill-temperature must be > 0")
    if not (0.0 < args.primary_balanced_black_ratio < 1.0):
        raise ValueError("--primary-balanced-black-ratio must be in (0, 1)")
    if not (0.0 < args.consolidation_balanced_black_ratio < 1.0):
        raise ValueError("--consolidation-balanced-black-ratio must be in (0, 1)")
    if args.alternating and not args.no_gating and args.arena_games < 2:
        raise ValueError("--arena-games must be >= 2 for alternating side-aware gating")
    if (not args.alternating) and args.alternating_pattern != "alternate":
        raise ValueError("--alternating-pattern requires --alternating")
    if args.keep_generations is not None and args.position_budget is not None:
        raise ValueError("Specify only one of --keep-generations or --position-budget")
    if args.position_budget_max is not None and args.position_budget is None:
        raise ValueError("--position-budget-max requires --position-budget")
    if position_budget_max is not None and position_budget is None:
        raise ValueError("position budget max-cap requires an active position budget")
    if (
        position_budget is not None
        and position_budget_max is not None
        and position_budget_max < position_budget
    ):
        raise ValueError("position budget max-cap must be >= position budget")
    consolidation_lr = LEARNING_RATE * args.consolidation_lr_factor

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
            "--min-blackfocus-plies", str(args.min_blackfocus_plies),
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
            "--se-reduction", str(se_reduction),
            *(
                ["--use-side-specialized-heads"]
                if use_side_specialized_heads
                else ["--no-use-side-specialized-heads"]
            ),
            *(["--use-se-blocks"] if use_se_blocks else ["--no-use-se-blocks"]),
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
        "gate_min_other_side_white_effective": gate_min_other_side_white,
        "gate_min_other_side_black_effective": gate_min_other_side_black,
        "arena_games": args.arena_games,
        "arena_sims": args.arena_sims,
        "black_focus_arena_games": args.black_focus_arena_games,
        "black_focus_arena_sims": black_focus_arena_sims,
        "black_focus_gate_threshold": args.black_focus_gate_threshold,
        "black_focus_games": args.black_focus_games,
        "black_focus_simulations": args.black_focus_simulations,
        "black_focus_tier_min": args.black_focus_tier_min,
        "black_focus_tier_max": args.black_focus_tier_max,
        "black_focus_arena_tier_min": args.black_focus_arena_tier_min,
        "black_focus_arena_tier_max": args.black_focus_arena_tier_max,
        "human_seed_games": args.human_seed_games,
        "human_seed_simulations": args.human_seed_simulations,
        "human_seed_dir": args.human_seed_dir,
        "human_seed_side": args.human_seed_side,
        "human_seed_max_positions": args.human_seed_max_positions,
        "black_train_sims_mult": float(args.black_train_sims_mult),
        "black_opponent_sims_mult": float(args.black_opponent_sims_mult),
        "primary_no_resume": bool(args.primary_no_resume),
        "opponent_pool_size": args.pool_size,
        "position_budget_effective": position_budget,
        "position_budget_max_effective": position_budget_max,
        "selfplay_sims_jitter_pct": selfplay_sims_jitter_pct,
        "adaptive_curriculum_enabled": bool(args.adaptive_curriculum),
        "adaptive_settings": {
            "min_scale": float(args.adaptive_min_scale),
            "max_scale": float(args.adaptive_max_scale),
            "up_factor": float(args.adaptive_up_factor),
            "down_factor": float(args.adaptive_down_factor),
            "min_normal_games": int(args.adaptive_min_normal_games),
        },
        "se_blocks_enabled": bool(use_se_blocks),
        "se_reduction": int(se_reduction),
        "side_specialized_heads_enabled": bool(use_side_specialized_heads),
        "consolidation": {
            "enabled": bool(args.consolidation_epochs > 0),
            "epochs": int(args.consolidation_epochs),
            "lr_factor": float(args.consolidation_lr_factor),
            "lr": float(consolidation_lr),
            "batch_size": int(args.consolidation_batch_size),
            "balance_sides": bool(args.consolidation_balance_sides),
            "balanced_black_ratio": float(args.consolidation_balanced_black_ratio),
            "distill_value_weight": float(args.consolidation_distill_value_weight),
            "distill_policy_weight": float(args.consolidation_distill_policy_weight),
            "distill_temperature": float(args.consolidation_distill_temperature),
        },
        "primary_balance_sides": bool(args.primary_balance_sides),
        "primary_balanced_black_ratio": float(args.primary_balanced_black_ratio),
        "primary_train_only_side": args.primary_train_only_side,
        "consolidation_train_only_side": args.consolidation_train_only_side,
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
    adaptive_scales = {"white": 1.0, "black": 1.0, "both": 1.0}
    run_metadata["adaptive_scales_initial"] = {k: float(v) for k, v in adaptive_scales.items()}

    mode_str = "ALTERNATING" if args.alternating else "STANDARD"

    print(f"\n{'#'*60}")
    print(f"  SELF-PLAY ITERATION LOOP ({mode_str})")
    print(f"{'#'*60}")
    print(f"  Iterations:  {args.iterations}")
    human_seed_sims_label = (
        args.human_seed_simulations if args.human_seed_simulations is not None else args.simulations
    )
    print(
        f"  Games/iter:  {args.games} normal + {args.curriculum_games} curriculum + "
        f"{args.black_focus_games} black-focus + {args.human_seed_games} human-seed"
    )
    print(
        f"  Simulations: {args.simulations} normal, {args.curriculum_simulations} curriculum, "
        f"{args.black_focus_simulations} black-focus, {human_seed_sims_label} human-seed"
    )
    if args.black_focus_tier_min is not None or args.black_focus_tier_max is not None:
        bf_tmin = args.black_focus_tier_min if args.black_focus_tier_min is not None else 1
        bf_tmax = args.black_focus_tier_max if args.black_focus_tier_max is not None else "max"
        print(f"  BF tiers:    generation {bf_tmin}..{bf_tmax}")
    print(f"  Target:      {args.train_target}")
    if selfplay_sims_jitter_pct > 0:
        print(f"  Sim jitter:  +/- {100 * selfplay_sims_jitter_pct:.0f}% (self-play generation only)")
    if args.adaptive_curriculum:
        print(
            f"  Adaptive:    enabled (scale range {args.adaptive_min_scale:.2f}..{args.adaptive_max_scale:.2f}, "
            f"up={args.adaptive_up_factor:.2f}, down={args.adaptive_down_factor:.2f}, "
            f"min normal={args.adaptive_min_normal_games})"
        )
    print(f"  SE blocks:   {'on' if use_se_blocks else 'off'} (reduction={se_reduction})")
    print(f"  Side heads:  {'on' if use_side_specialized_heads else 'off'}")
    if args.consolidation_epochs > 0:
        print(
            f"  Consolidate: epochs={args.consolidation_epochs}, lr={consolidation_lr:.2e}, "
            f"batch={args.consolidation_batch_size}, balanced={args.consolidation_balance_sides}, "
            f"black_ratio={args.consolidation_balanced_black_ratio:.2f}, "
            f"distill(v={args.consolidation_distill_value_weight}, "
            f"p={args.consolidation_distill_policy_weight}, t={args.consolidation_distill_temperature})"
        )
    else:
        print("  Consolidate: disabled")
    print(
        f"  Primary bal: {args.primary_balance_sides} "
        f"(black_ratio={args.primary_balanced_black_ratio:.2f})"
    )
    print(
        f"  Side focus:  primary={args.primary_train_only_side}, "
        f"consolidation={args.consolidation_train_only_side}"
    )
    if args.alternating:
        print(f"  Opponent:    {opponent_sims} sims (pool/frozen fallback)")
        print(
            f"  Black tune:  train_sims x{args.black_train_sims_mult:.2f}, "
            f"opp_sims x{args.black_opponent_sims_mult:.2f}"
        )
        print(f"  Pattern:     {args.alternating_pattern}")
        print(f"  Pool size:   {args.pool_size} archived models")
    if args.no_gating:
        print("  Gating:      disabled (auto-promote candidates)")
    else:
        print(f"  Gating:      arena {args.arena_games} games @ {args.arena_sims} sims, threshold={args.gate_threshold:.2f}")
        if args.alternating:
            print(
                f"               alternating side floor non-trained: "
                f"white-train->{gate_min_other_side_white:.2f}, "
                f"black-train->{gate_min_other_side_black:.2f}"
            )
            if args.black_focus_arena_games > 0:
                print(
                    f"               black-focus arena {args.black_focus_arena_games} games @ "
                    f"{black_focus_arena_sims} sims, threshold={args.black_focus_gate_threshold:.2f} "
                    f"(Black training side only)"
                )
                if args.black_focus_arena_tier_min is not None or args.black_focus_arena_tier_max is not None:
                    bfa_tmin = (
                        args.black_focus_arena_tier_min
                        if args.black_focus_arena_tier_min is not None else 1
                    )
                    bfa_tmax = (
                        args.black_focus_arena_tier_max
                        if args.black_focus_arena_tier_max is not None else "max"
                    )
                    print(f"               black-focus arena tiers {bfa_tmin}..{bfa_tmax}")
    print(f"  Epochs:      {args.epochs}")
    if position_budget is not None:
        budget_human = "no human_games" if args.exclude_human_games else "with human_games"
        if position_budget_max is not None:
            budget_label = f"{position_budget}..{position_budget_max}"
        else:
            budget_label = f"{position_budget}"
        print(f"  Window:      position budget {budget_label} raw positions (+ curriculum_bootstrap, {budget_human})")
    else:
        if args.exclude_human_games:
            print(f"  Window:      last {keep_gens} generations (+ curriculum_bootstrap, no human_games)")
        else:
            print(f"  Window:      last {keep_gens} generations (+ curriculum_bootstrap + human_games)")
    print(f"  Check pos:   {'keep' if args.keep_check_positions else 'skip'}")
    print(f"  BF filter:   min_blackfocus_plies={args.min_blackfocus_plies}")
    print(f"  Starting at: generation {start_gen}")
    print(f"  Existing:    {total_games_existing} games, {total_pos_existing} positions")
    print(f"  Model:       {model_path}")
    print(f"  Seed:        {base_seed}")
    print(f"  Fine-tune:   warmup_epochs={warmup_epochs}, warmup_start_factor={warmup_start_factor}")
    print(f"  Resume:      {'disabled (primary cold start)' if args.primary_no_resume else 'from incumbent'}")
    if args.human_eval:
        diag_label = "on" if args.human_eval_value_diagnostics else "off"
        print(f"  Human eval:  enabled (dir={args.human_eval_dir}, value_diag={diag_label})")
    if args.human_seed_games > 0:
        print(
            f"  Human seed:  enabled (dir={args.human_seed_dir}, side={args.human_seed_side}, "
            f"max_positions={args.human_seed_max_positions})"
        )
    print()

    for i in range(args.iterations):
        gen = start_gen + i
        gen_dir = os.path.join(raw_dir, f"nn_gen{gen}")
        iter_start = time.time()

        # Determine training side for alternating mode
        if args.alternating:
            if args.alternating_pattern == "black_only":
                train_side = "black"
            elif args.alternating_pattern == "white_only":
                train_side = "white"
            else:
                train_side = "black" if (i % 2 == 0) else "white"
            side_label = train_side.upper()
        else:
            train_side = "both"
            side_label = "BOTH"

        primary_train_only_side = args.primary_train_only_side
        if primary_train_only_side == "auto":
            primary_train_only_side = train_side if args.alternating else "none"
        consolidation_train_only_side = args.consolidation_train_only_side
        if consolidation_train_only_side == "auto":
            consolidation_train_only_side = train_side if args.alternating else "none"

        print(f"\n{'#'*60}")
        print(f"  ITERATION {i+1}/{args.iterations}  -  Generation {gen}  -  Training: {side_label}")
        print(f"{'#'*60}")

        # Step 1a: Generate normal games
        opponent_pool_count = 0
        mix = _compute_adaptive_mix(
            base_normal_games=args.games,
            base_curriculum_games=args.curriculum_games,
            base_black_focus_games=args.black_focus_games,
            train_side=train_side,
            alternating=args.alternating,
            adaptive_enabled=args.adaptive_curriculum,
            side_scales=adaptive_scales,
            min_normal_games=args.adaptive_min_normal_games,
        )
        effective_normal_games = mix["normal_games"]
        effective_curriculum_games = mix["curriculum_games"]
        effective_black_focus_games = mix["black_focus_games"]
        adaptive_scale_key = mix["scale_key"]
        adaptive_scale_value = mix["scale_value"]
        effective_train_sims = args.simulations
        effective_opponent_sims = opponent_sims
        effective_black_focus_sims = args.black_focus_simulations
        effective_human_seed_games = args.human_seed_games
        human_seed_base_sims = (
            args.human_seed_simulations
            if args.human_seed_simulations is not None
            else args.simulations
        )
        effective_human_seed_sims = human_seed_base_sims
        if args.alternating and train_side == "black":
            effective_train_sims = max(1, int(round(args.simulations * args.black_train_sims_mult)))
            effective_opponent_sims = max(
                1, int(round(opponent_sims * args.black_opponent_sims_mult))
            )
            effective_black_focus_sims = max(
                1, int(round(args.black_focus_simulations * args.black_train_sims_mult))
            )
            effective_human_seed_sims = max(
                1, int(round(human_seed_base_sims * args.black_train_sims_mult))
            )
        print(
            f"  Mix:         normal={effective_normal_games}, curriculum={effective_curriculum_games}, "
            f"black-focus={effective_black_focus_games}, human-seed={effective_human_seed_games}"
            + (
                f" (adaptive {adaptive_scale_key} scale={adaptive_scale_value:.3f})"
                if args.adaptive_curriculum else ""
            )
        )

        def _sim_bounds(base_sims):
            if selfplay_sims_jitter_pct <= 0:
                return None
            min_sims = max(1, int(round(base_sims * (1.0 - selfplay_sims_jitter_pct))))
            max_sims = max(min_sims, int(round(base_sims * (1.0 + selfplay_sims_jitter_pct))))
            return min_sims, max_sims

        gen_cmd = [
            sys.executable, "data_generation.py",
            "--num-games", str(effective_normal_games),
            "--simulations", str(effective_train_sims),
            "--output-dir", gen_dir,
            "--use-model", model_path,
            "--seed", str(base_seed + gen * 10 + 1),
        ]
        normal_sim_bounds = _sim_bounds(effective_train_sims)
        if normal_sim_bounds is not None:
            gen_cmd.extend([
                "--simulations-min", str(normal_sim_bounds[0]),
                "--simulations-max", str(normal_sim_bounds[1]),
            ])
        if args.keep_check_positions:
            gen_cmd.append("--keep-check-positions")

        archived_models = []
        if args.alternating:
            gen_cmd.extend(["--train-side", train_side])
            gen_cmd.extend(["--opponent-sims", str(effective_opponent_sims)])
            archived_models = _list_archive_models(archive_dir)
            opponent_pool_count = min(len(archived_models), args.pool_size) if args.pool_size > 0 else 0
            if args.pool_size > 0 and archived_models:
                gen_cmd.extend(["--opponent-pool-dir", archive_dir])
                gen_cmd.extend(["--opponent-pool-size", str(args.pool_size)])
            # Fallback: single frozen model if pool is unavailable
            elif os.path.exists(frozen_path):
                gen_cmd.extend(["--opponent-model", frozen_path])

        t_gen = run(gen_cmd,
            f"[{i+1}/{args.iterations}] Generating {effective_normal_games} normal games "
            f"(gen {gen}, training {side_label})")

        print(f"\n  --- Normal Games ---")
        normal_summary = summarize_generation(gen_dir)

        # Step 1b: Generate extra games from human-recorded start positions
        t_hs = 0.0
        human_seed_summary = None
        if effective_human_seed_games > 0:
            hs_dir = os.path.join(raw_dir, f"nn_gen{gen}_humanseed")
            start_side = args.human_seed_side
            if start_side == "auto":
                start_side = train_side if args.alternating else "any"
            hs_cmd = [
                sys.executable, "data_generation.py",
                "--num-games", str(effective_human_seed_games),
                "--simulations", str(effective_human_seed_sims),
                "--output-dir", hs_dir,
                "--use-model", model_path,
                "--start-fen-dir", args.human_seed_dir,
                "--start-fen-max-positions", str(args.human_seed_max_positions),
                "--seed", str(base_seed + gen * 10 + 6),
            ]
            if start_side != "any":
                hs_cmd.extend(["--start-fen-side", start_side])
                if start_side == "black":
                    hs_cmd.append("--start-fen-convert-white-to-black")
            hs_sim_bounds = _sim_bounds(effective_human_seed_sims)
            if hs_sim_bounds is not None:
                hs_cmd.extend([
                    "--simulations-min", str(hs_sim_bounds[0]),
                    "--simulations-max", str(hs_sim_bounds[1]),
                ])
            if args.keep_check_positions:
                hs_cmd.append("--keep-check-positions")
            if args.alternating:
                hs_cmd.extend(["--train-side", train_side])
                hs_cmd.extend(["--opponent-sims", str(effective_opponent_sims)])
                if args.pool_size > 0 and archived_models:
                    hs_cmd.extend(["--opponent-pool-dir", archive_dir])
                    hs_cmd.extend(["--opponent-pool-size", str(args.pool_size)])
                elif os.path.exists(frozen_path):
                    hs_cmd.extend(["--opponent-model", frozen_path])

            t_hs = run(
                hs_cmd,
                f"[{i+1}/{args.iterations}] Generating {effective_human_seed_games} human-seed games "
                f"@ {effective_human_seed_sims} sims (gen {gen}, start-side={start_side})"
            )
            print(f"\n  --- Human-Seed Games ---")
            human_seed_summary = summarize_generation(hs_dir)
            t_gen += t_hs

        # Step 1c: Generate extra black-focused games from black-advantage starts
        t_bf = 0.0
        black_focus_summary = None
        if effective_black_focus_games > 0 and args.alternating and train_side == "black":
            bf_dir = os.path.join(raw_dir, f"nn_gen{gen}_blackfocus")
            bf_cmd = [
                sys.executable, "data_generation.py",
                "--num-games", str(effective_black_focus_games),
                "--simulations", str(effective_black_focus_sims),
                "--output-dir", bf_dir,
                "--use-model", model_path,
                "--curriculum",
                "--curriculum-live-results",
                "--train-side", "black",
                "--opponent-sims", str(effective_opponent_sims),
                "--seed", str(base_seed + gen * 10 + 5),
            ]
            if args.black_focus_tier_min is not None:
                bf_cmd.extend(["--curriculum-tier-min", str(args.black_focus_tier_min)])
            if args.black_focus_tier_max is not None:
                bf_cmd.extend(["--curriculum-tier-max", str(args.black_focus_tier_max)])
            blackfocus_sim_bounds = _sim_bounds(effective_black_focus_sims)
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
                f"[{i+1}/{args.iterations}] Generating {effective_black_focus_games} black-focus games "
                f"@ {effective_black_focus_sims} sims (gen {gen}, live outcomes)"
            )
            print(f"\n  --- Black-Focus Games ---")
            black_focus_summary = summarize_generation(bf_dir)
            t_gen += t_bf

        # Step 1d: Generate curriculum endgame games
        t_cur = 0.0
        if effective_curriculum_games > 0:
            cur_dir = os.path.join(raw_dir, f"nn_gen{gen}_curriculum")
            curriculum_sim_bounds = _sim_bounds(args.curriculum_simulations)
            cur_cmd = [
                sys.executable, "data_generation.py",
                "--num-games", str(effective_curriculum_games),
                "--simulations", str(args.curriculum_simulations),
                "--output-dir", cur_dir,
                "--use-model", model_path,
                "--curriculum",
                "--scripted-black",
                "--seed", str(base_seed + gen * 10 + 2),
            ]
            if curriculum_sim_bounds is not None:
                cur_cmd.extend([
                    "--simulations-min", str(curriculum_sim_bounds[0]),
                    "--simulations-max", str(curriculum_sim_bounds[1]),
                ])
            if args.keep_check_positions:
                cur_cmd.append("--keep-check-positions")
            t_cur = run(
                cur_cmd,
                f"[{i+1}/{args.iterations}] Generating {effective_curriculum_games} curriculum games @ {args.curriculum_simulations} sims (gen {gen}, tiered values)"
            )

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
            "--min-blackfocus-plies", str(args.min_blackfocus_plies),
            *(["--exclude-human-games"] if args.exclude_human_games else []),
        ]
        if position_budget is not None:
            proc_cmd.extend(["--position-budget", str(position_budget)])
            if position_budget_max is not None:
                proc_cmd.extend(["--position-budget-max", str(position_budget_max)])
            proc_desc = (
                f"[{i+1}/{args.iterations}] Reprocessing data "
                f"(position_budget={position_budget}"
                + (f", cap={position_budget_max}" if position_budget_max is not None else "")
                + f", total on disk: {total_games} games)"
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
            "--target", args.train_target,
            "--epochs", str(args.epochs),
            "--data-dir", processed_dir,
            "--model-dir", candidate_dir,
            "--seed", str(base_seed + gen * 10 + 4),
            "--warmup-epochs", str(warmup_epochs),
            "--warmup-start-factor", str(warmup_start_factor),
            "--se-reduction", str(se_reduction),
        ]
        train_cmd.append(
            "--balanced-sides-train" if args.primary_balance_sides else "--no-balanced-sides-train"
        )
        train_cmd.extend(["--balanced-black-ratio", str(args.primary_balanced_black_ratio)])
        train_cmd.append("--use-se-blocks" if use_se_blocks else "--no-use-se-blocks")
        train_cmd.append(
            "--use-side-specialized-heads"
            if use_side_specialized_heads else "--no-use-side-specialized-heads"
        )
        if primary_train_only_side != "none":
            train_cmd.extend(["--train-only-side", primary_train_only_side])
        if os.path.exists(model_path) and not args.primary_no_resume:
            train_cmd.extend(["--resume-from", model_path])
        t_train_primary = run(train_cmd, f"[{i+1}/{args.iterations}] Training candidate model")
        t_train_consolidation = 0.0
        if args.consolidation_epochs > 0:
            if not os.path.exists(candidate_path):
                raise FileNotFoundError(
                    f"Candidate model missing before consolidation: {candidate_path}"
                )
            consolidation_cmd = [
                sys.executable, "train.py",
                "--target", args.train_target,
                "--epochs", str(args.consolidation_epochs),
                "--data-dir", processed_dir,
                "--model-dir", candidate_dir,
                "--seed", str(base_seed + gen * 10 + 5),
                "--batch-size", str(args.consolidation_batch_size),
                "--lr", str(consolidation_lr),
                "--warmup-epochs", "0",
                "--resume-from", candidate_path,
                "--se-reduction", str(se_reduction),
                "--distill-value-weight", str(args.consolidation_distill_value_weight),
                "--distill-policy-weight", str(args.consolidation_distill_policy_weight),
                "--distill-temperature", str(args.consolidation_distill_temperature),
            ]
            consolidation_cmd.append("--use-se-blocks" if use_se_blocks else "--no-use-se-blocks")
            consolidation_cmd.append(
                "--use-side-specialized-heads"
                if use_side_specialized_heads else "--no-use-side-specialized-heads"
            )
            consolidation_cmd.append(
                "--balanced-sides-train" if args.consolidation_balance_sides else "--no-balanced-sides-train"
            )
            consolidation_cmd.extend(
                ["--balanced-black-ratio", str(args.consolidation_balanced_black_ratio)]
            )
            if consolidation_train_only_side != "none":
                consolidation_cmd.extend(["--train-only-side", consolidation_train_only_side])
            if (
                args.consolidation_distill_value_weight > 0
                or args.consolidation_distill_policy_weight > 0
            ):
                consolidation_cmd.extend(["--distill-from", model_path])
            t_train_consolidation = run(
                consolidation_cmd,
                f"[{i+1}/{args.iterations}] Consolidation pass"
            )
        t_train = t_train_primary + t_train_consolidation

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
                arena_tag="standard",
            )
            gate_info["enabled"] = True
            gate_info["threshold"] = args.gate_threshold
            gate_info["min_other_side"] = args.gate_min_other_side
            if args.alternating:
                primary_side = train_side
                other_side = "white" if train_side == "black" else "black"
                min_other_side = (
                    gate_min_other_side_black
                    if train_side == "black"
                    else gate_min_other_side_white
                )
                gate_info["min_other_side"] = min_other_side
                primary_info = gate_info[f"candidate_{primary_side}"]
                other_info = gate_info[f"candidate_{other_side}"]
                black_focus_gate = None
                if train_side == "black" and args.black_focus_arena_games > 0:
                    black_focus_gate = _run_arena(
                        candidate_model=candidate_path,
                        incumbent_model=model_path,
                        gen=gen,
                        model_dir=model_dir,
                        arena_games=args.black_focus_arena_games,
                        arena_sims=black_focus_arena_sims,
                        arena_workers=args.arena_workers,
                        base_seed=base_seed + 700000,
                        arena_tag="black_focus",
                        curriculum=True,
                        curriculum_live_results=True,
                        curriculum_tier_min=args.black_focus_arena_tier_min,
                        curriculum_tier_max=args.black_focus_arena_tier_max,
                    )
                    gate_info["black_focus_arena"] = black_focus_gate
                    gate_info["black_focus_threshold"] = args.black_focus_gate_threshold
                gate_info.update({
                    "decision_mode": "side_aware",
                    "primary_side": primary_side,
                    "other_side": other_side,
                    "primary_score": primary_info["score"],
                    "other_score": other_info["score"],
                    "primary_games": primary_info["games"],
                    "other_games": other_info["games"],
                })
                base_side_pass = (
                    gate_info["total_games"] > 0
                    and gate_info["primary_games"] > 0
                    and gate_info["other_games"] > 0
                    and gate_info["primary_score"] >= args.gate_threshold
                    and gate_info["other_score"] >= min_other_side
                )
                if black_focus_gate is None:
                    accepted = base_side_pass
                else:
                    focus_primary = black_focus_gate["candidate_black"]["score"]
                    focus_games = black_focus_gate["candidate_black"]["games"]
                    gate_info["black_focus_primary_score"] = focus_primary
                    gate_info["black_focus_primary_games"] = focus_games
                    gate_info["decision_mode"] = "side_aware_black_focus"
                    accepted = (
                        gate_info["total_games"] > 0
                        and gate_info["primary_games"] > 0
                        and gate_info["other_games"] > 0
                        and gate_info["other_score"] >= min_other_side
                        and (
                            gate_info["primary_score"] >= args.gate_threshold
                            or (
                                focus_games > 0
                                and focus_primary >= args.black_focus_gate_threshold
                            )
                        )
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
            if gate_info.get("decision_mode") == "side_aware_black_focus":
                print(
                    f"\n  Candidate REJECTED "
                    f"(trained-{gate_info['primary_side']} opening score={gate_info['primary_score']:.3f} "
                    f"vs {args.gate_threshold:.3f}, black-focus score={gate_info.get('black_focus_primary_score', 0.0):.3f} "
                    f"vs {args.black_focus_gate_threshold:.3f}, other-{gate_info['other_side']} "
                    f"score={gate_info['other_score']:.3f} vs {gate_info.get('min_other_side', args.gate_min_other_side):.3f}); "
                    f"keeping incumbent"
                )
            elif gate_info.get("decision_mode") == "side_aware":
                print(
                    f"\n  Candidate REJECTED "
                    f"(trained-{gate_info['primary_side']} score={gate_info['primary_score']:.3f} "
                    f"vs {args.gate_threshold:.3f}, other-{gate_info['other_side']} "
                    f"score={gate_info['other_score']:.3f} vs {gate_info.get('min_other_side', args.gate_min_other_side):.3f}); "
                    f"keeping incumbent"
                )
            else:
                print(f"\n  Candidate REJECTED (score={gate_info['score']:.3f} < {args.gate_threshold:.3f}); keeping incumbent")

        adaptive_update = None
        if args.adaptive_curriculum and adaptive_scale_key is not None:
            prev_scale = adaptive_scales.get(adaptive_scale_key, 1.0)
            new_scale, reason = _update_adaptive_scale(
                current_scale=prev_scale,
                accepted=accepted,
                gate_info=gate_info,
                up_factor=args.adaptive_up_factor,
                down_factor=args.adaptive_down_factor,
                min_scale=args.adaptive_min_scale,
                max_scale=args.adaptive_max_scale,
            )
            adaptive_scales[adaptive_scale_key] = new_scale
            adaptive_update = {
                "scale_key": adaptive_scale_key,
                "previous_scale": float(prev_scale),
                "new_scale": float(new_scale),
                "reason": reason,
            }
            print(
                f"  Adaptive:   {adaptive_scale_key} scale {prev_scale:.3f} -> {new_scale:.3f} "
                f"({reason})"
            )

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
            "primary_train_only_side_effective": primary_train_only_side,
            "consolidation_train_only_side_effective": consolidation_train_only_side,
            "effective_mix": {
                "normal_games": int(effective_normal_games),
                "curriculum_games": int(effective_curriculum_games),
                "black_focus_games": int(effective_black_focus_games),
                "human_seed_games": int(effective_human_seed_games),
                "normal_simulations": int(effective_train_sims),
                "black_focus_simulations": int(effective_black_focus_sims),
                "human_seed_simulations": int(effective_human_seed_sims),
                "opponent_simulations": int(effective_opponent_sims),
                "base_total_games": int(mix["base_total_games"]),
                "effective_total_games": int(mix["effective_total_games"] + effective_human_seed_games),
                "adaptive_scale_key": adaptive_scale_key,
                "adaptive_scale_value": float(adaptive_scale_value),
            },
            "adaptive_update": adaptive_update,
            "opponent_pool_count": opponent_pool_count,
            "normal_generation": normal_summary,
            "human_seed_generation": human_seed_summary,
            "black_focus_generation": black_focus_summary,
            "curriculum_generation": curriculum_summary,
            "candidate_model_path": candidate_path,
            "accepted": accepted,
            "gate": gate_info,
            "timings_seconds": {
                "generate_total": float(t_gen),
                "generate_normal": float(t_gen - t_hs - t_bf - t_cur),
                "generate_human_seed": float(t_hs),
                "generate_black_focus": float(t_bf),
                "generate_curriculum": float(t_cur),
                "process": float(t_proc),
                "train_primary": float(t_train_primary),
                "train_consolidation": float(t_train_consolidation),
                "train": float(t_train),
                "iteration_total": float(iter_elapsed),
            },
        })

        print(f"\n{'-'*60}")
        gate_label = "ACCEPTED" if accepted else "REJECTED"
        print(f"  Iteration {i+1}/{args.iterations} complete (trained {side_label}, candidate {gate_label})")
        print(f"  Time:  generate {format_time(t_gen)}  |  process {format_time(t_proc)}  |  train {format_time(t_train)}  |  total {format_time(iter_elapsed)}")
        if gate_info.get("enabled"):
            if gate_info.get("decision_mode") == "side_aware_black_focus":
                print(
                    f"  Gate:  trained-{gate_info['primary_side']} opening={gate_info['primary_score']:.3f} "
                    f"(>={args.gate_threshold:.3f})  black-focus={gate_info.get('black_focus_primary_score', 0.0):.3f} "
                    f"(>={args.black_focus_gate_threshold:.3f})  other-{gate_info['other_side']}={gate_info['other_score']:.3f} "
                    f"(>={gate_info.get('min_other_side', args.gate_min_other_side):.3f})  overall={gate_info['score']:.3f}  games={gate_info['total_games']}"
                )
            elif gate_info.get("decision_mode") == "side_aware":
                print(
                    f"  Gate:  trained-{gate_info['primary_side']}={gate_info['primary_score']:.3f} "
                    f"(>={args.gate_threshold:.3f})  other-{gate_info['other_side']}={gate_info['other_score']:.3f} "
                    f"(>={gate_info.get('min_other_side', args.gate_min_other_side):.3f})  overall={gate_info['score']:.3f}  games={gate_info['total_games']}"
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
    run_metadata["adaptive_scales_final"] = {k: float(v) for k, v in adaptive_scales.items()}
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

