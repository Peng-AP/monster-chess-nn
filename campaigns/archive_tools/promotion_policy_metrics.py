"""Measure the distinct-promotion head on held-out promotion decisions.

The incumbent is lifted into the candidate's extended policy ABI with a zero
delta, which exactly reproduces its source/destination logits.  Metrics are
reported only on enabled rows carrying promotion target mass; ordinary rows
would otherwise hide nearly the whole signal.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import POLICY_SIZE, PROMOTION_AWARE_POLICY_SIZE  # noqa: E402
from train import build_model, load_model_for_inference, load_state_dict_flexible  # noqa: E402


def _lift_incumbent(path, candidate, device):
    model = build_model(
        input_channels=candidate.input_channels,
        policy_head_channels=candidate.policy_head_channels,
        policy_head_type=candidate.policy_head_type,
        policy_attention_channels=candidate.policy_attention_channels,
        side_policy_adapters=candidate.side_policy_adapters,
        promotion_policy=True,
        stem_channels=candidate.stem_channels,
        residual_block_channels=candidate.residual_block_channels,
        use_se_blocks=candidate.use_se_blocks,
        se_reduction=candidate.se_reduction,
        use_wdl_head=candidate.use_wdl_head,
        value_head_mode=candidate.value_head_mode,
        spatial_value_head=candidate.spatial_value_head,
        value_head_conv_channels=candidate.value_head_conv_channels,
        use_moves_left_head=candidate.use_moves_left_head,
        moves_left_head_channels=candidate.moves_left_head_channels,
    ).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    loaded, skipped = load_state_dict_flexible(model, state)
    if skipped:
        raise ValueError(f"incumbent architecture differs: skipped {skipped}")
    if loaded != len(state):
        raise ValueError(f"loaded {loaded} of {len(state)} incumbent tensors")
    model.eval()
    return model


def _score(model, positions, targets, legal_masks, rows, device, batch_size):
    totals = {"full_ce": 0.0, "full_top1": 0,
              "promotion_choice_ce": 0.0, "promotion_choice_top1": 0}
    for start in range(0, len(rows), batch_size):
        idx = rows[start:start + batch_size]
        x = torch.from_numpy(
            np.asarray(positions[idx]).transpose(0, 3, 1, 2).copy()).to(device)
        y = torch.from_numpy(np.asarray(targets[idx]).copy()).to(device)
        packed = np.asarray(legal_masks[idx])
        legal = torch.from_numpy(np.unpackbits(
            packed, axis=1, count=PROMOTION_AWARE_POLICY_SIZE).astype(bool)
        ).to(device)
        with torch.no_grad():
            _, logits = model(x)
        logp = torch.log_softmax(logits, dim=1)
        totals["full_ce"] += float((-(y * logp).sum(dim=1)).sum())
        totals["full_top1"] += int((logits.argmax(1) == y.argmax(1)).sum())

        promo_y = y[:, POLICY_SIZE:]
        promo_y = promo_y / promo_y.sum(dim=1, keepdim=True)
        promo_logits = logits[:, POLICY_SIZE:].masked_fill(
            ~legal[:, POLICY_SIZE:], -torch.inf)
        promo_logp = torch.log_softmax(promo_logits, dim=1)
        conditional_ce = torch.where(
            promo_y > 0, -promo_y * promo_logp, torch.zeros_like(promo_y))
        totals["promotion_choice_ce"] += float(conditional_ce.sum())
        totals["promotion_choice_top1"] += int(
            (promo_logits.argmax(1) == promo_y.argmax(1)).sum())
    n = len(rows)
    return {
        "rows": n,
        "full_ce": totals["full_ce"] / n,
        "full_top1": totals["full_top1"] / n,
        "promotion_choice_ce": totals["promotion_choice_ce"] / n,
        "promotion_choice_top1": totals["promotion_choice_top1"] / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=("val", "test"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate, _ = load_model_for_inference(args.candidate, device)
    if not candidate.promotion_policy:
        raise ValueError("candidate does not have a distinct-promotion head")
    incumbent = _lift_incumbent(args.incumbent, candidate, device)

    positions = np.load(os.path.join(args.data_dir, "positions.npy"), mmap_mode="r")
    targets = np.load(os.path.join(args.data_dir, "policies.npy"), mmap_mode="r")
    weights = np.load(os.path.join(args.data_dir, "policy_weights.npy"), mmap_mode="r")
    legal = np.load(os.path.join(args.data_dir, "legal_masks_packed.npy"), mmap_mode="r")
    if targets.shape[1] != PROMOTION_AWARE_POLICY_SIZE:
        raise ValueError(f"expected {PROMOTION_AWARE_POLICY_SIZE}-wide targets")
    with np.load(os.path.join(args.data_dir, "splits.npz")) as split_file:
        split_indices = {name: split_file[name] for name in args.splits}

    report = {
        "candidate": args.candidate,
        "incumbent": args.incumbent,
        "data_dir": args.data_dir,
        "device": str(device),
        "splits": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for name, indices in split_indices.items():
        promotion_mass = np.asarray(targets[indices, POLICY_SIZE:]).sum(axis=1)
        rows = indices[(promotion_mass > 0) & (np.asarray(weights[indices]) > 0)]
        report["splits"][name] = {
            "incumbent_zero_lift": _score(
                incumbent, positions, targets, legal, rows, device, args.batch_size),
            "candidate": _score(
                candidate, positions, targets, legal, rows, device, args.batch_size),
        }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(
        args.out_dir,
        f"promotion_policy_metrics_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
