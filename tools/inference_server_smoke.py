"""Smoke and benchmark for the stage-2 inference server (DIRECTIVE E4).

Must be run as a file, not piped to the interpreter: the server uses the
`spawn` start method (the only one on Windows), which re-imports `__main__` in
the child. A heredoc has no importable `__main__`, so `spawn` fails with
`OSError: Invalid argument: '<stdin>'`. Every caller therefore needs a real
module and an `if __name__ == "__main__"` guard.

    py -3 tools/inference_server_smoke.py
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "native"))

MODEL = os.path.join("models", "fresh_start_v19", "best_value_net.pt")
FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import monster_native as mn
    from inference_server import InferenceServer

    print("starting server...", flush=True)
    server = InferenceServer(args.model, num_workers=args.workers, max_batch=256)
    channels = server.input_channels
    print(f"  ready, input_channels={channels}", flush=True)

    fn = server.eval_fn_for(0)

    # Shape contract.
    buf = np.zeros((4, channels, 8, 8), dtype=np.float32).tobytes()
    values, policies = fn(buf, 4, channels)
    assert len(values) // 4 == 4, len(values)
    assert len(policies) // 4 == 4 * 4096, len(policies)
    print("  shape contract ok (4 values, 4x4096 logits)", flush=True)

    # A real search driven entirely through the server.
    tree = mn.Tree(FEN)
    tree.run_batched_puct(64, fn, batch_size=16, channels=channels,
                          allow_early_stop=False)  # warm
    tree = mn.Tree(FEN)
    started = time.perf_counter()
    tree.run_batched_puct(args.sims, fn, batch_size=16, channels=channels,
                          allow_early_stop=False)
    elapsed = (time.perf_counter() - started) * 1000
    top = sorted(tree.root_visits(), key=lambda kv: -kv[1])[:3]
    print(f"  {args.sims}-sim search via server: {elapsed:.0f} ms, top {top}",
          flush=True)

    print("stopping...", flush=True)
    clean = server.stop()
    print(f"  clean stop: {clean}", flush=True)
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
