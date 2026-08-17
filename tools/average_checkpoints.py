"""Average several checkpoints into one model (SWA), with BatchNorm repair.

Measured 2026-08-07 on a from-scratch run: epoch 12 beats epoch 4 by 52 Elo
head-to-head, yet against a common opponent (v21) the two land within 0.002 of
each other overall while differing sharply by colour -- e4 is the better Black
(+0.069 over v21, 100 Black wins), e12 the better White (+0.031). They are
complementary rather than ranked, which is the classic case for averaging.

**BatchNorm is the trap.** This network has 18 BN layers. Averaging weights
leaves their running statistics describing activations none of the averaged
weights produce, and the result can be badly miscalibrated. `--recalibrate`
re-estimates those statistics with forward passes over real positions, which
is what makes the average usable rather than merely plausible.

    py -3 tools/average_checkpoints.py --out avg.pt --recalibrate \\
        --checkpoint models/candidates/run/selected_epoch_012.pt \\
        --checkpoint models/candidates/run/selected_epoch_004.pt
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj


def average_states(paths):
    """Mean of float tensors; integer buffers (BN num_batches) take the first."""
    states = [load_state(p) for p in paths]
    keys = set(states[0])
    for s in states[1:]:
        if set(s) != keys:
            raise SystemExit("checkpoints have different parameter sets")
    out = {}
    for k in states[0]:
        first = states[0][k]
        if torch.is_tensor(first) and first.is_floating_point():
            out[k] = torch.stack([s[k].float() for s in states]).mean(0).to(first.dtype)
        else:
            out[k] = first.clone() if torch.is_tensor(first) else first
    return out


def recalibrate_bn(model, data_dir, batches, batch_size, device):
    """Re-estimate BatchNorm statistics for the averaged weights.

    Momentum is set to None so each layer accumulates a true running average
    over the passes rather than an exponentially weighted one -- the standard
    SWA procedure.
    """
    positions = np.load(os.path.join(data_dir, "positions.npy"), mmap_mode="r")
    n = len(positions)
    bns = [m for m in model.modules()
           if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    if not bns:
        return 0
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(batches):
            idx = np.sort(rng.choice(n, size=min(batch_size, n), replace=False))
            batch = np.asarray(positions[idx], dtype=np.float32)
            # stored (N,8,8,C) -> model wants (N,C,8,8)
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)
            model(tensor.half() if next(model.parameters()).dtype == torch.half
                  else tensor)
    model.eval()
    return len(bns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="path to average (repeatable, 2 or more)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recalibrate", action="store_true",
                    help="re-estimate BatchNorm statistics (strongly advised)")
    ap.add_argument("--data-dir",
                    default="data/processed/bootstrap_replay_main_gen_0005_teacher3200_full")
    ap.add_argument("--bn-batches", type=int, default=100)
    ap.add_argument("--bn-batch-size", type=int, default=256)
    args = ap.parse_args()

    paths = [p if os.path.isabs(p) else os.path.join(ROOT, p) for p in args.checkpoint]
    if len(paths) < 2:
        raise SystemExit("need at least two checkpoints to average")
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"missing checkpoint: {p}")
    print(f"averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"   {os.path.relpath(p, ROOT)}")

    averaged = average_states(paths)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if args.recalibrate:
        from train import load_model_for_inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tmp = out + ".pre_bn.pt"
        torch.save(averaged, tmp)
        model, _meta = load_model_for_inference(tmp, device)
        model = model.float()
        n = recalibrate_bn(model, os.path.join(ROOT, args.data_dir),
                           args.bn_batches, args.bn_batch_size, device)
        print(f"recalibrated {n} BatchNorm layers over "
              f"{args.bn_batches * args.bn_batch_size} positions")
        averaged = {k: v.cpu() for k, v in model.state_dict().items()}
        os.remove(tmp)

    torch.save(averaged, out)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
