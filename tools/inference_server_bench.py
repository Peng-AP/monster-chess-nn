"""Does the stage-2 server actually pay? (DIRECTIVE E4)

With a single worker it cannot: IPC latency has nothing to amortise against,
and the smoke run measures 485 ms against ~180 ms for the in-process bridge.
The entire claim is *cross-worker* batching — 8 workers each submitting 16
leaves become one 128-leaf forward instead of eight 16-leaf ones, and measured
per-position forward cost is 0.294 ms at batch 16 against 0.037 ms at 256.

So this compares the two architectures at the concurrency they will actually
run at:

  stage 1  N worker processes, each with its own model, own forwards
  stage 2  N worker processes, no model, one shared server

Reported as wall-clock for the same total work. A stage-2 number that is not
clearly better at N=8 means the server is not worth its complexity and should
not be wired into generation.

    py -3 tools/inference_server_bench.py --workers 8 --games 4 --sims 200
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "native"))

MODEL = os.path.join("models", "fresh_start_v19", "best_value_net.pt")
FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def _search_with(eval_fn, channels, games, sims, batch):
    import monster_native as mn
    decisions = 0
    for _ in range(games):
        tree = mn.Tree(FEN)
        tree.run_batched_puct(sims, eval_fn, batch_size=batch, channels=channels,
                              allow_early_stop=False)
        decisions += 1
    return decisions


def _stage1_worker(args):
    """Own model, own forwards — the current architecture."""
    model, games, sims, batch = args
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, os.path.join(ROOT, "native"))
    from native_mcts import make_bridge
    from evaluation import NNEvaluator
    evaluator = NNEvaluator(model)
    eval_fn, channels = make_bridge(evaluator)
    _search_with(eval_fn, channels, 1, 32, batch)  # warm
    started = time.perf_counter()
    _search_with(eval_fn, channels, games, sims, batch)
    return time.perf_counter() - started


_SHARED = {}


def _stage2_init(requests, responses, channels):
    """Queues can only reach a spawned child by inheritance, which means the
    pool initializer -- passing them as map() arguments raises
    "Queue objects should only be shared between processes through
    inheritance"."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, os.path.join(ROOT, "native"))
    _SHARED["requests"] = requests
    _SHARED["responses"] = responses
    _SHARED["channels"] = channels


def _stage2_worker(args):
    """No model at all — leaves go to the shared server."""
    worker_id, games, sims, batch = args
    requests = _SHARED["requests"]
    response = _SHARED["responses"][worker_id]
    channels = _SHARED["channels"]

    def eval_fn(buf, n, _channels):
        requests.put((worker_id, bytes(buf), n))
        return response.get()

    _search_with(eval_fn, channels, 1, 32, batch)  # warm
    started = time.perf_counter()
    _search_with(eval_fn, channels, games, sims, batch)
    return time.perf_counter() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    ctx = mp.get_context("spawn")

    # Compare the SEARCH work, not the setup. Stage 1 loads a model inside
    # every worker; the stage-2 server loads once before timing starts. Wall
    # clock would therefore credit the server with 8 model loads it simply
    # moved earlier. Each worker times its own post-warm work and returns it;
    # the slowest worker is what the batch actually waits for.
    print(f"stage 1: {args.workers} workers, each with its own model", flush=True)
    started = time.perf_counter()
    with ctx.Pool(args.workers) as pool:
        durations = pool.map(_stage1_worker,
                             [(args.model, args.games, args.sims, args.batch)]
                             * args.workers)
    stage1_wall = time.perf_counter() - started
    stage1 = max(durations)
    print(f"  search {stage1:.2f}s (slowest worker) | wall {stage1_wall:.1f}s "
          f"incl. {args.workers} model loads", flush=True)

    print(f"stage 2: {args.workers} workers, one shared server", flush=True)
    from inference_server import InferenceServer
    server = InferenceServer(args.model, num_workers=args.workers, max_batch=256)
    started = time.perf_counter()
    with ctx.Pool(args.workers, initializer=_stage2_init,
                  initargs=(server.requests, server.responses,
                            server.input_channels)) as pool:
        durations = pool.map(_stage2_worker,
                             [(i, args.games, args.sims, args.batch)
                              for i in range(args.workers)])
    stage2_wall = time.perf_counter() - started
    stage2 = max(durations)
    server.stop()
    print(f"  search {stage2:.2f}s (slowest worker) | wall {stage2_wall:.1f}s "
          f"incl. 1 model load", flush=True)

    print()
    print(f"search work:  stage2/stage1 = {stage2 / stage1:.2f}x "
          f"({'server wins' if stage2 < stage1 else 'server LOSES'})")
    print(f"GPU memory:   1 model resident instead of {args.workers}")


if __name__ == "__main__":
    main()
