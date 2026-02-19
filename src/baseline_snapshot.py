"""Stage 0: Capture baseline metrics snapshot.

Records current model performance, data statistics, config values,
and self-play result distribution. Saves to baselines/baseline_<timestamp>.json.

No behavior changes — read-only against existing data and model.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch

# Add src/ to path for imports
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

from config import (
    TENSOR_SHAPE, POLICY_SIZE, POLICY_LOSS_WEIGHT,
    BATCH_SIZE, VALUE_TARGET, BLEND_WEIGHT,
    VALUE_LOSS_EXPONENT, LR_GAMMA, LEARNING_RATE, EPOCHS,
    MCTS_SIMULATIONS, SLIDING_WINDOW, HUMAN_DATA_WEIGHT,
    OPPONENT_SIMULATIONS, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON,
    EXPLORATION_CONSTANT, MAX_GAME_TURNS,
    PROCESSED_DATA_DIR, MODEL_DIR, RAW_DATA_DIR,
)
from train import load_model_for_inference, _make_loader, _eval_epoch


def model_checksum(path):
    """SHA256 of the model file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    """Best-effort short git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def scan_raw_results(raw_dir):
    """Scan all raw JSONL files for game result distribution."""
    results_by_dir = {}
    for dirpath, _dirnames, filenames in os.walk(raw_dir):
        dirname = os.path.basename(dirpath)
        jsonl_files = [f for f in filenames if f.endswith(".jsonl")]
        if not jsonl_files:
            continue
        white_wins = 0
        black_wins = 0
        draws = 0
        game_lengths = []
        counted_games = 0
        empty_files = 0
        for fname in jsonl_files:
            path = os.path.join(dirpath, fname)
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
        n = counted_games
        results_by_dir[dirname] = {
            "games": n,
            "files": len(jsonl_files),
            "empty_files": empty_files,
            "positions": sum(game_lengths),
            "white_wins": white_wins,
            "black_wins": black_wins,
            "draws": draws,
            "avg_length": round(sum(game_lengths) / n, 1) if n else 0,
        }
    return results_by_dir


def evaluate_model(model, positions, mcts_values, policies, split_indices,
                   split_name, device):
    """Evaluate model on a data split, returning metrics dict."""
    idx = split_indices
    targets = mcts_values[idx]  # always use mcts_value for baseline

    if policies is not None:
        pol_split = policies[idx]
    else:
        # No policy data -> create dummy zeros (policy CE won't be meaningful)
        pol_split = np.zeros((len(idx), POLICY_SIZE), dtype=np.float32)

    loader = _make_loader(positions[idx], targets, pol_split, BATCH_SIZE, shuffle=False)
    total_loss, val_loss, pol_loss, mae, mse = _eval_epoch(
        model, loader, device, POLICY_LOSS_WEIGHT,
    )

    # Sign accuracy on non-draw positions
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for X_b, yv_b, _ in loader:
            vp, _ = model(X_b.to(device))
            all_preds.append(vp.cpu().numpy())
            all_true.append(yv_b.numpy())
    preds = np.concatenate(all_preds).flatten()
    true = np.concatenate(all_true).flatten()
    non_draw = true != 0
    sign_acc = float(np.mean(np.sign(preds[non_draw]) == np.sign(true[non_draw]))) if non_draw.sum() > 0 else None

    return {
        "split": split_name,
        "n_positions": len(idx),
        "total_loss": round(total_loss, 6),
        "power_loss_2.5": round(val_loss, 6),
        "true_mse": round(mse, 6),
        "mae": round(mae, 6),
        "policy_ce": round(pol_loss, 6),
        "sign_accuracy": round(sign_acc, 4) if sign_acc is not None else None,
        "pred_mean": round(float(preds.mean()), 4),
        "pred_std": round(float(preds.std()), 4),
        "target_mean": round(float(true.mean()), 4),
        "target_std": round(float(true.std()), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 0: Capture baseline metrics snapshot")
    parser.add_argument("--data-dir", type=str, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--raw-dir", type=str, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "baselines"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.model_dir, "best_value_net.pt")

    if not os.path.exists(model_path):
        print(f"ERROR: No model found at {model_path}")
        return

    print(f"Device: {device}")
    print(f"Model:  {model_path}")
    print(f"Data:   {args.data_dir}")
    print(f"Raw:    {args.raw_dir}")

    # Load model
    model, policy_head_channels = load_model_for_inference(model_path, device)

    total_params = sum(p.numel() for p in model.parameters())
    checksum = model_checksum(model_path)
    print(f"Params: {total_params:,}")
    print(f"SHA256: {checksum[:16]}...")

    # Load processed data (handle missing policies.npy gracefully)
    positions = np.load(os.path.join(args.data_dir, "positions.npy"))
    mcts_values = np.load(os.path.join(args.data_dir, "mcts_values.npy"))
    splits = np.load(os.path.join(args.data_dir, "splits.npz"))
    policy_path = os.path.join(args.data_dir, "policies.npy")
    policies = np.load(policy_path) if os.path.exists(policy_path) else None
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    print(f"\nData: {len(positions)} total positions")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Evaluate on val and test
    print("\nEvaluating on val split...")
    val_metrics = evaluate_model(model, positions, mcts_values, policies,
                                 val_idx, "val", device)
    print(f"  Power loss: {val_metrics['power_loss_2.5']:.4f}")
    print(f"  True MSE:   {val_metrics['true_mse']:.4f}")
    print(f"  MAE:        {val_metrics['mae']:.4f}")
    print(f"  Policy CE:  {val_metrics['policy_ce']:.4f}")
    print(f"  Sign acc:   {val_metrics['sign_accuracy']}")

    print("\nEvaluating on test split...")
    test_metrics = evaluate_model(model, positions, mcts_values, policies,
                                  test_idx, "test", device)
    print(f"  Power loss: {test_metrics['power_loss_2.5']:.4f}")
    print(f"  True MSE:   {test_metrics['true_mse']:.4f}")
    print(f"  MAE:        {test_metrics['mae']:.4f}")
    print(f"  Policy CE:  {test_metrics['policy_ce']:.4f}")
    print(f"  Sign acc:   {test_metrics['sign_accuracy']}")

    # Scan raw data for result distribution
    print("\nScanning raw data...")
    raw_results = scan_raw_results(args.raw_dir)

    # Aggregate totals
    total_games = sum(d["games"] for d in raw_results.values())
    total_white = sum(d["white_wins"] for d in raw_results.values())
    total_black = sum(d["black_wins"] for d in raw_results.values())
    total_draws = sum(d["draws"] for d in raw_results.values())
    print(f"  Total games: {total_games}")
    print(f"  White wins:  {total_white} ({100*total_white/total_games:.0f}%)" if total_games else "")
    print(f"  Black wins:  {total_black} ({100*total_black/total_games:.0f}%)" if total_games else "")
    print(f"  Draws:       {total_draws} ({100*total_draws/total_games:.0f}%)" if total_games else "")

    # Value distribution of training data
    all_values = mcts_values
    value_stats = {
        "mean": round(float(all_values.mean()), 4),
        "std": round(float(all_values.std()), 4),
        "min": round(float(all_values.min()), 4),
        "max": round(float(all_values.max()), 4),
        "pct_positive": round(float((all_values > 0).mean()), 4),
        "pct_negative": round(float((all_values < 0).mean()), 4),
        "pct_zero": round(float((all_values == 0).mean()), 4),
    }

    # Config snapshot
    config_snapshot = {
        "MCTS_SIMULATIONS": MCTS_SIMULATIONS,
        "SLIDING_WINDOW": SLIDING_WINDOW,
        "HUMAN_DATA_WEIGHT": HUMAN_DATA_WEIGHT,
        "OPPONENT_SIMULATIONS": OPPONENT_SIMULATIONS,
        "C_PUCT": C_PUCT,
        "DIRICHLET_ALPHA": DIRICHLET_ALPHA,
        "DIRICHLET_EPSILON": DIRICHLET_EPSILON,
        "EXPLORATION_CONSTANT": EXPLORATION_CONSTANT,
        "MAX_GAME_TURNS": MAX_GAME_TURNS,
        "VALUE_TARGET": VALUE_TARGET,
        "BLEND_WEIGHT": BLEND_WEIGHT,
        "VALUE_LOSS_EXPONENT": VALUE_LOSS_EXPONENT,
        "LR_GAMMA": LR_GAMMA,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "POLICY_SIZE": POLICY_SIZE,
        "POLICY_LOSS_WEIGHT": POLICY_LOSS_WEIGHT,
    }

    # Build artifact
    artifact = {
        "stage": 0,
        "description": "Baseline snapshot before improvement plan",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": {
            "path": model_path,
            "sha256": checksum,
            "total_params": total_params,
            "policy_head_channels": int(policy_head_channels),
            "git_commit": get_git_commit(),
        },
        "data": {
            "processed_dir": args.data_dir,
            "total_positions": len(positions),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "value_distribution": value_stats,
        },
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "self_play_results": raw_results,
        "self_play_totals": {
            "games": total_games,
            "white_wins": total_white,
            "black_wins": total_black,
            "draws": total_draws,
        },
        "config": config_snapshot,
    }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"baseline_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nBaseline artifact saved to {out_path}")
    print("Stage 0 complete.")


if __name__ == "__main__":
    main()

