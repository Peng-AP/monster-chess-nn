"""Continuous self-play iteration loop.

Repeats: generate NN-guided games -> reprocess all data -> retrain model.
Each generation is saved to data/raw/nn_genN/ so nothing is overwritten.
"""
import argparse
import os
import subprocess
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=SRC_DIR)
    if result.returncode != 0:
        print(f"FAILED: {desc} (exit code {result.returncode})")
        sys.exit(1)


def find_next_gen(raw_dir):
    """Find the next generation number by scanning existing nn_gen* dirs."""
    gen = 1
    while os.path.exists(os.path.join(raw_dir, f"nn_gen{gen}")):
        gen += 1
    return gen


def main():
    parser = argparse.ArgumentParser(description="Iterate self-play training loop")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games", type=int, default=300)
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

    for i in range(args.iterations):
        gen = start_gen + i
        gen_dir = os.path.join(raw_dir, f"nn_gen{gen}")

        print(f"\n{'#'*60}")
        print(f"  ITERATION {i+1}/{args.iterations}  —  Generation {gen}")
        print(f"{'#'*60}")

        # Step 1: Generate games with current model
        run([
            sys.executable, "data_generation.py",
            "--num-games", str(args.games),
            "--simulations", str(args.simulations),
            "--output-dir", gen_dir,
            "--use-model", model_path,
        ], f"Generating {args.games} games (gen {gen})")

        # Step 2: Reprocess all data
        run([
            sys.executable, "data_processor.py",
            "--raw-dir", raw_dir,
            "--output-dir", processed_dir,
        ], "Reprocessing all data")

        # Step 3: Retrain
        run([
            sys.executable, "train.py",
            "--target", "blend",
            "--epochs", str(args.epochs),
            "--data-dir", processed_dir,
            "--model-dir", model_dir,
        ], f"Retraining (iteration {i+1})")

        print(f"\n  Iteration {i+1} complete. Model updated.\n")


if __name__ == "__main__":
    main()
