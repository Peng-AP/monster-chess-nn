"""Continuous self-play iteration loop.

Repeats: generate NN-guided games -> reprocess all data -> retrain model.
Each generation is saved to data/raw/nn_genN/ so nothing is overwritten.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

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
    results = {1: 0, -1: 0, 0: 0}
    game_lengths = []
    files = sorted(glob.glob(os.path.join(gen_dir, "*.jsonl")))
    for path in files:
        with open(path) as f:
            lines = f.readlines()
        if lines:
            last = json.loads(lines[-1])
            r = last.get("game_result", 0)
            results[r] = results.get(r, 0) + 1
            game_lengths.append(len(lines))

    n = len(files)
    if n == 0:
        return

    avg_len = sum(game_lengths) / n
    min_len = min(game_lengths)
    max_len = max(game_lengths)
    w, b, d = results.get(1, 0), results.get(-1, 0), results.get(0, 0)
    total_pos = sum(game_lengths)

    print(f"\n  --- Generation Summary ---")
    print(f"  Games:     {n}")
    print(f"  Positions: {total_pos}")
    print(f"  Results:   White {w} ({100*w/n:.0f}%)  |  Black {b} ({100*b/n:.0f}%)  |  Draw {d} ({100*d/n:.0f}%)")
    print(f"  Length:    avg {avg_len:.0f}  |  min {min_len}  |  max {max_len}")


def main():
    parser = argparse.ArgumentParser(description="Iterate self-play training loop")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--curriculum-games", type=int, default=100,
                        help="Extra curriculum endgame games per iteration (0 to disable)")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    model_dir = os.path.join(PROJECT_ROOT, "models")
    model_path = os.path.join(model_dir, "best_value_net.pt")

    if not os.path.exists(model_path):
        print(f"ERROR: No model found at {model_path}")
        print("Run the initial training first (pipeline steps 1-3).")
        sys.exit(1)

    start_gen = find_next_gen(raw_dir)
    total_games_existing, total_pos_existing = count_data(raw_dir)
    loop_start = time.time()

    print(f"\n{'#'*60}")
    print(f"  SELF-PLAY ITERATION LOOP")
    print(f"{'#'*60}")
    print(f"  Iterations:  {args.iterations}")
    print(f"  Games/iter:  {args.games} normal + {args.curriculum_games} curriculum")
    print(f"  Simulations: {args.simulations}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Starting at: generation {start_gen}")
    print(f"  Existing:    {total_games_existing} games, {total_pos_existing} positions")
    print(f"  Model:       {model_path}")
    print()

    for i in range(args.iterations):
        gen = start_gen + i
        gen_dir = os.path.join(raw_dir, f"nn_gen{gen}")
        iter_start = time.time()

        print(f"\n{'#'*60}")
        print(f"  ITERATION {i+1}/{args.iterations}  —  Generation {gen}")
        print(f"{'#'*60}")

        # Step 1a: Generate normal games with current model
        t_gen = run([
            sys.executable, "data_generation.py",
            "--num-games", str(args.games),
            "--simulations", str(args.simulations),
            "--output-dir", gen_dir,
            "--use-model", model_path,
        ], f"[{i+1}/{args.iterations}] Generating {args.games} normal games (gen {gen})")

        print(f"\n  --- Normal Games ---")
        summarize_generation(gen_dir)

        # Step 1b: Generate curriculum endgame games
        t_cur = 0.0
        if args.curriculum_games > 0:
            cur_dir = os.path.join(raw_dir, f"nn_gen{gen}_curriculum")
            t_cur = run([
                sys.executable, "data_generation.py",
                "--num-games", str(args.curriculum_games),
                "--simulations", str(args.simulations),
                "--output-dir", cur_dir,
                "--use-model", model_path,
                "--curriculum",
                "--scripted-black",
            ], f"[{i+1}/{args.iterations}] Generating {args.curriculum_games} curriculum games (gen {gen})")

            print(f"\n  --- Curriculum Games ---")
            summarize_generation(cur_dir)
            t_gen += t_cur

        # Step 2: Reprocess all data
        total_games, total_pos = count_data(raw_dir)
        t_proc = run([
            sys.executable, "data_processor.py",
            "--raw-dir", raw_dir,
            "--output-dir", processed_dir,
        ], f"[{i+1}/{args.iterations}] Reprocessing all data ({total_games} games, {total_pos} positions)")

        # Step 3: Retrain
        t_train = run([
            sys.executable, "train.py",
            "--target", "blend",
            "--epochs", str(args.epochs),
            "--data-dir", processed_dir,
            "--model-dir", model_dir,
        ], f"[{i+1}/{args.iterations}] Retraining model")

        iter_elapsed = time.time() - iter_start
        total_elapsed = time.time() - loop_start
        remaining = (args.iterations - i - 1) * iter_elapsed

        print(f"\n{'─'*60}")
        print(f"  Iteration {i+1}/{args.iterations} complete")
        print(f"  Time:  generate {format_time(t_gen)}  |  process {format_time(t_proc)}  |  train {format_time(t_train)}  |  total {format_time(iter_elapsed)}")
        print(f"  Clock: {format_time(total_elapsed)} elapsed  |  ~{format_time(remaining)} remaining")
        print(f"{'─'*60}")

    total_elapsed = time.time() - loop_start
    total_games, total_pos = count_data(raw_dir)
    print(f"\n{'#'*60}")
    print(f"  ALL DONE — {args.iterations} iterations in {format_time(total_elapsed)}")
    print(f"  Total data: {total_games} games, {total_pos} positions")
    print(f"  Model: {model_path}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
