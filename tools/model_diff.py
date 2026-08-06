"""Offline candidate-vs-incumbent differential on identical positions.

The cheap gate that runs BEFORE any match burns hours: both models evaluate
the same held-out positions and are compared on the metrics that actually
predict playing strength (WDL-v18 lesson — aggregate validation loss and soft
policy CE both looked fine while policy top-1 fell six points and Black play
collapsed):

  - policy top-1 accuracy over enabled policy teachers,
  - winner-sign accuracy over non-draw outcomes,
  - both split by side to move.

Models with different encoding widths are compared fairly: positions are
converted between the legacy 15-plane and current 17-plane layouts (planes
0-13 are shared; legacy plane 14 == 17-plane White-pawn-progress plane 15;
the rank-coordinate and Black-progress planes are recomputable from the
board planes).

With --enforce the candidate must stay within --margin of the incumbent on
top-1 and sign accuracy, overall and per side. Exit code 1 on failure so
overnight drivers abort before spending match time on a regressed model.

    py -3 tools/model_diff.py --candidate models/candidates/X/best_value_net.pt \\
        --incumbent models/fresh_start_v17/best_value_net.pt \\
        --data-dir data/processed/combined_v15 --enforce
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import (
    TURN_LAYER, LEGACY_TENSOR_CHANNELS, TENSOR_SHAPE,
    RANK_COORD_LAYER, WHITE_PAWN_PROGRESS_LAYER, BLACK_PAWN_PROGRESS_LAYER,
    LEGACY_PAWN_ADVANCEMENT_LAYER,
)

BLACK_PAWN_PLANE = 6  # encoding.PIECE_TO_LAYER[(PAWN, BLACK)]
WHITE_PAWN_PLANE = 0  # encoding.PIECE_TO_LAYER[(PAWN, WHITE)]
BLACK_KING_PLANE = 11  # last Black piece plane; 6..11 inclusive = Black's army

# Planes 0-11 are the piece planes and are IDENTICAL in the 15- and 17-plane
# layouts, so the opening filter below needs no channel conversion.
OPENING_MIN_WHITE_PAWNS = 4   # White has lost no pawn yet
OPENING_MIN_BLACK_MEN = 14    # Black has lost at most two men


def opening_mask(positions, idx, chunk=8192,
                 min_white_pawns=OPENING_MIN_WHITE_PAWNS,
                 min_black_men=OPENING_MIN_BLACK_MEN):
    """Boolean mask over ``idx`` selecting opening positions.

    Read in chunks so a memory-mapped corpus is never materialized whole.
    """
    keep = np.zeros(len(idx), dtype=bool)
    for i in range(0, len(idx), chunk):
        block = np.asarray(positions[idx[i:i + chunk]])
        white_pawns = block[..., WHITE_PAWN_PLANE].sum(axis=(1, 2))
        black_men = block[..., BLACK_PAWN_PLANE:BLACK_KING_PLANE + 1].sum(axis=(1, 2, 3))
        keep[i:i + chunk] = (white_pawns >= min_white_pawns) & (black_men >= min_black_men)
    return keep


def convert_channels(x, target_channels):
    """Convert a (N, 8, 8, C) batch between the 15- and 17-plane encodings.

    Axis 1 is the rank. Planes 0-13 are identical in both layouts.
    """
    source_channels = x.shape[3]
    target_channels = int(target_channels)
    if source_channels == target_channels:
        return x
    if (source_channels, target_channels) == (TENSOR_SHAPE[2], LEGACY_TENSOR_CHANNELS):
        out = np.empty(x.shape[:3] + (LEGACY_TENSOR_CHANNELS,), dtype=x.dtype)
        out[..., :LEGACY_PAWN_ADVANCEMENT_LAYER] = x[..., :LEGACY_PAWN_ADVANCEMENT_LAYER]
        out[..., LEGACY_PAWN_ADVANCEMENT_LAYER] = x[..., WHITE_PAWN_PROGRESS_LAYER]
        return out
    if (source_channels, target_channels) == (LEGACY_TENSOR_CHANNELS, TENSOR_SHAPE[2]):
        out = np.empty(x.shape[:3] + (TENSOR_SHAPE[2],), dtype=x.dtype)
        out[..., :RANK_COORD_LAYER] = x[..., :RANK_COORD_LAYER]
        ranks = np.arange(8, dtype=x.dtype)
        out[..., RANK_COORD_LAYER] = ((ranks - 3.5) / 3.5)[None, :, None]
        out[..., WHITE_PAWN_PROGRESS_LAYER] = x[..., LEGACY_PAWN_ADVANCEMENT_LAYER]
        black_progress = np.clip((6.0 - ranks) / 6.0, 0.0, 1.0)[None, :, None]
        out[..., BLACK_PAWN_PROGRESS_LAYER] = x[..., BLACK_PAWN_PLANE] * black_progress
        return out
    raise ValueError(
        f"No conversion from {source_channels} to {target_channels} channels")


def evaluate_model(model_path, positions, policies, policy_weights,
                   results_side, white_turn, batch_size=512):
    """Metrics for one model on shared positions (converted to its encoding)."""
    import torch
    from train import load_model_for_inference

    model, _ = load_model_for_inference(model_path, torch.device("cpu"))
    channels = model.input_channels

    pol_enabled = (policy_weights > 0) & (policies.sum(axis=1) > 0)
    target_idx = policies.argmax(axis=1)
    non_draw = results_side != 0

    top1_correct = np.zeros(len(positions), dtype=bool)
    sign_correct = np.zeros(len(positions), dtype=bool)
    ce_sum, ce_n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            sl = slice(i, i + batch_size)
            x = convert_channels(np.ascontiguousarray(positions[sl]), channels)
            xt = torch.from_numpy(x).float().permute(0, 3, 1, 2)
            value, logits = model(xt)
            pred_v = value.squeeze(-1).numpy()
            pred_idx = logits.argmax(dim=1).numpy()
            top1_correct[sl] = pred_idx == target_idx[sl]
            sign_correct[sl] = np.sign(pred_v) == np.sign(results_side[sl])
            en = pol_enabled[sl]
            if en.any():
                logp = torch.log_softmax(logits[torch.from_numpy(en)], dim=1)
                tgt = torch.from_numpy(policies[sl][en]).float()
                ce_sum += float(-(tgt * logp).sum())
                ce_n += int(en.sum())

    def _acc(correct, mask):
        return float(correct[mask].mean()) if mask.any() else None

    metrics = {
        "channels": int(channels),
        "policy_ce": ce_sum / ce_n if ce_n else None,
        "policy_top1": _acc(top1_correct, pol_enabled),
        "policy_top1_white": _acc(top1_correct, pol_enabled & white_turn),
        "policy_top1_black": _acc(top1_correct, pol_enabled & ~white_turn),
        "sign_acc": _acc(sign_correct, non_draw),
        "sign_acc_white": _acc(sign_correct, non_draw & white_turn),
        "sign_acc_black": _acc(sign_correct, non_draw & ~white_turn),
    }
    return metrics


GATED_METRICS = (
    "policy_top1", "policy_top1_white", "policy_top1_black",
    "sign_acc", "sign_acc_white", "sign_acc_black",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--data-dir", required=True,
                    help="processed corpus dir (positions.npy etc.)")
    ap.add_argument("--split", default="test", choices=["test", "val", "all"],
                    help="'all' = every position; use on a corpus unseen by "
                         "BOTH models (leakage-clean cross-evaluation)")
    ap.add_argument("--max-positions", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--enforce", action="store_true",
                    help="exit 1 if the candidate regresses beyond --margin")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="allowed drop vs incumbent on each gated metric")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    ap.add_argument("--report-path", default=None,
                    help="write the report to this exact path")
    ap.add_argument("--opening-only", action="store_true",
                    help=f"restrict to opening positions (>= "
                         f"{OPENING_MIN_WHITE_PAWNS} White pawns and >= "
                         f"{OPENING_MIN_BLACK_MEN} Black men). Applied BEFORE "
                         f"--max-positions so the sample is not decimated first")
    args = ap.parse_args()

    positions = np.load(os.path.join(args.data_dir, "positions.npy"), mmap_mode="r")
    results = np.load(os.path.join(args.data_dir, "game_results.npy"))
    policies_all = np.load(os.path.join(args.data_dir, "policies.npy"), mmap_mode="r")
    pw_path = os.path.join(args.data_dir, "policy_weights.npy")
    weights_all = (np.load(pw_path) if os.path.exists(pw_path)
                   else np.ones((len(results),), dtype=np.float32))
    with np.load(os.path.join(args.data_dir, "splits.npz")) as f:
        if args.split == "all":
            idx = np.sort(np.concatenate([f[k] for k in f.files]))
        else:
            idx = f[args.split]
    # Filter BEFORE decimating: subsampling first would leave a handful of
    # opening positions and produce a noise reading that looks like a result.
    if args.opening_only:
        before = len(idx)
        idx = idx[opening_mask(positions, idx)]
        if len(idx) == 0:
            raise SystemExit(
                f"--opening-only matched 0 of {before} positions in "
                f"{args.data_dir}; nothing to compare")
        print(f"--opening-only: {len(idx)} of {before} positions kept")

    if len(idx) > args.max_positions:
        idx = idx[np.linspace(0, len(idx) - 1, args.max_positions).astype(np.int64)]

    pos = np.ascontiguousarray(positions[idx])
    pol = np.ascontiguousarray(policies_all[idx])
    pw = weights_all[idx]
    white_turn = pos[:, 0, 0, TURN_LAYER] > 0
    side_sign = np.where(white_turn, 1.0, -1.0).astype(np.float32)
    results_side = results[idx] * side_sign  # side-to-move perspective

    rows = {}
    for name, path in (("candidate", args.candidate), ("incumbent", args.incumbent)):
        rows[name] = evaluate_model(
            path, pos, pol, pw, results_side, white_turn, args.batch_size)

    print(f"=== model_diff: {len(idx)} {args.split} positions from {args.data_dir} ===")
    print(f"{'metric':22s} {'candidate':>10s} {'incumbent':>10s} {'delta':>8s}")
    failures = []
    for key in ("policy_ce",) + GATED_METRICS:
        c, inc = rows["candidate"][key], rows["incumbent"][key]
        if c is None or inc is None:
            print(f"{key:22s} {'n/a':>10s} {'n/a':>10s}")
            continue
        delta = c - inc
        print(f"{key:22s} {c:10.4f} {inc:10.4f} {delta:+8.4f}")
        # policy_ce is informational (lower is better, but soft CE hid the
        # WDL-v18 regression); only the decisive metrics gate.
        if key in GATED_METRICS and delta < -args.margin:
            failures.append(f"{key}: {c:.4f} vs incumbent {inc:.4f} "
                            f"(drop {-delta:.4f} > margin {args.margin})")

    out = {
        "candidate": args.candidate,
        "incumbent": args.incumbent,
        "data_dir": args.data_dir,
        "split": args.split,
        "opening_only": bool(args.opening_only),
        "positions": int(len(idx)),
        "margin": args.margin,
        "metrics": rows,
        "failures": failures,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.report_path:
        path = os.path.abspath(args.report_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(
            args.out_dir, f"model_diff_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {path}")

    for line in failures:
        print("  FAIL " + line)
    if failures and args.enforce:
        print("MODEL DIFF: FAIL — candidate regresses vs incumbent; "
              "do not spend match time on it.")
        sys.exit(1)
    print("MODEL DIFF: " + ("FAIL (informational)" if failures else "PASS"))


if __name__ == "__main__":
    main()
