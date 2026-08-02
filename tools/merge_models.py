"""Average two or more checkpoints into one ("model soup").

    py -3 tools/merge_models.py --model models/candidates/v19_K/best_value_net.pt \
        --model models/candidates/v19_B/best_value_net.pt \
        --out models/candidates/v19_KB/best_value_net.pt

Why this is a reasonable thing to do *here* and usually is not: averaging
weights only helps when the models sit in the same loss basin, which normally
means they were fine-tuned from a shared starting point. v19_K and v19_B were
trained from the same seed -- same initialisation, same per-epoch shuffle
order, same architecture -- and differ only in whether ps_monster's records
carried value gradient. Measured cosine similarity of their flattened
parameters is 0.932. Two independently-initialised networks would be near 0,
and averaging them would produce noise.

Integer buffers (num_batches_tracked) are taken from the first model rather
than averaged. BatchNorm running statistics ARE averaged, which is the usual
soup treatment and is approximate -- the honest check is the gate, not a
plausibility argument.

The output is a candidate like any other: it has to clear the bar twice, and
promotion still needs the owner's playtest.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402


def soup(paths, weights=None):
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    keys = set(states[0])
    for s, p in zip(states[1:], paths[1:]):
        if set(s) != keys:
            raise SystemExit(f"key mismatch between {paths[0]} and {p}")
        for k in keys:
            if s[k].shape != states[0][k].shape:
                raise SystemExit(f"shape mismatch for {k} in {p}")

    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    total = sum(weights)
    weights = [w / total for w in weights]

    out = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            acc = torch.zeros_like(states[0][k], dtype=torch.float64)
            for s, w in zip(states, weights):
                acc += s[k].to(torch.float64) * w
            out[k] = acc.to(states[0][k].dtype)
        else:
            # Counters, not parameters; averaging them is meaningless.
            out[k] = states[0][k].clone()
    return out, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="PATH")
    ap.add_argument("--weight", action="append", type=float, default=None,
                    help="blend weight per --model (default: equal)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.model) < 2:
        ap.error("need at least two --model paths")
    if args.weight and len(args.weight) != len(args.model):
        ap.error("--weight must be given once per --model")
    for p in args.model:
        if not os.path.exists(p):
            ap.error(f"no such model: {p}")
    if os.path.exists(args.out):
        ap.error(f"{args.out} exists -- refusing to overwrite a checkpoint")

    state, weights = soup(args.model, args.weight)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(state, args.out)

    for p, w in zip(args.model, weights):
        print(f"  {w:.3f}  {os.path.relpath(p, ROOT)}")
    print(f"-> {os.path.relpath(args.out, ROOT)}")

    # Load it back through the real loader: a soup that cannot be loaded for
    # inference is worse than useless, and this catches it immediately.
    from train import load_model_for_inference
    model, _ = load_model_for_inference(args.out, torch.device("cpu"))
    n = sum(q.numel() for q in model.parameters())
    print(f"   loads for inference, {n:,} params")


if __name__ == "__main__":
    main()
