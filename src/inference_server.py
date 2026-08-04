"""D3 stage 2: one inference server batching leaves across all worker games.

Stage 1 gave each worker its own model and its own forwards. That leaves the
GPU running many small batches: measured per-position forward cost is 0.294 ms
at batch 16 against 0.037 ms at batch 256 — an 8x efficiency gap that no amount
of native search speed recovers, because it is the network's time, not the
tree's.

This server takes the leaves that 8 workers would have submitted separately and
runs them as one forward. It is the *replacement* for per-worker inference, not
a wrapper around it (§4 risk row): workers hold no model at all, so GPU memory
holds one copy rather than eight.

Shape:

    worker --(worker_id, batch bytes)--> requests: one shared Queue
    server --(values, policies bytes)--> responses[worker_id]: one Queue each

The server drains the request queue up to `max_batch` leaves or `linger`
seconds, whichever comes first. Lingering is what buys the cross-game batching:
without it the server simply serves whoever arrived first and reproduces stage 1.

**Shutdown is the part that bites.** A worker blocked on `responses[id].get()`
while the server has already exited hangs forever, which is the same failure
mode as the post-timeout pool hang. So the server always replies — even when
shutting down — and `stop()` joins with a bounded timeout and escalates, the
same discipline as `data_generation.terminate_pool`.
"""
import multiprocessing as mp
import os
import queue
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

SENTINEL = None
DEFAULT_MAX_BATCH = 256
DEFAULT_LINGER = 0.002  # 2 ms: long enough to gather peers, short enough to hide


def _server_loop(model_path, requests, responses, max_batch, linger, ready):
    """Hold the model; serve batched forwards until the sentinel arrives."""
    import numpy as np
    from evaluation import NNEvaluator

    evaluator = NNEvaluator(model_path)
    torch = evaluator.torch
    channels = evaluator.input_channels
    ready.put(channels)

    def reply(pending):
        """Answer every request in `pending`, one forward for all of them."""
        if not pending:
            return
        arrays = [np.frombuffer(buf, dtype=np.float32).reshape(n, channels, 8, 8)
                  for _wid, buf, n in pending]
        batch = np.concatenate(arrays, axis=0)
        tensor = torch.from_numpy(batch).to(evaluator.device)
        if getattr(evaluator, "_half", False):
            tensor = tensor.half()
        with torch.no_grad():
            value, policy = evaluator.model(tensor)
        values = value.reshape(-1).float().cpu().numpy().astype(np.float32)
        policies = policy.reshape(batch.shape[0], -1).float().cpu().numpy().astype(np.float32)
        offset = 0
        for wid, _buf, n in pending:
            responses[wid].put((values[offset:offset + n].tobytes(),
                                policies[offset:offset + n].tobytes()))
            offset += n

    stopping = False
    while not stopping:
        pending = []
        total = 0
        try:
            item = requests.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is SENTINEL:
            break
        pending.append(item)
        total += item[2]

        # Linger briefly to gather the other workers' leaves — this is the
        # whole point of the server.
        deadline = time.perf_counter() + linger
        while total < max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = requests.get(timeout=remaining)
            except queue.Empty:
                break
            if item is SENTINEL:
                stopping = True
                break
            pending.append(item)
            total += item[2]

        reply(pending)

    # Drain anything still queued so no worker is left blocked on a reply.
    while True:
        try:
            item = requests.get_nowait()
        except queue.Empty:
            break
        if item is SENTINEL:
            continue
        wid, _buf, n = item
        responses[wid].put((b"", b""))


class InferenceServer:
    """Owns the server process and hands out per-worker eval callbacks."""

    def __init__(self, model_path, num_workers, max_batch=DEFAULT_MAX_BATCH,
                 linger=DEFAULT_LINGER, ctx=None):
        self.ctx = ctx or mp.get_context("spawn")
        self.requests = self.ctx.Queue()
        self.responses = [self.ctx.Queue() for _ in range(num_workers)]
        ready = self.ctx.Queue()
        self.process = self.ctx.Process(
            target=_server_loop,
            args=(model_path, self.requests, self.responses, max_batch, linger, ready),
            daemon=True)
        self.process.start()
        # Block until the model is loaded, so the first decision is not timed
        # against CUDA initialisation.
        self.input_channels = ready.get()

    def eval_fn_for(self, worker_id):
        """A drop-in for the stage-1 bridge, backed by the shared server."""
        requests = self.requests
        response = self.responses[worker_id]

        def eval_fn(buf, n, _channels):
            requests.put((worker_id, bytes(buf), n))
            return response.get()

        return eval_fn

    def stop(self, timeout=10):
        try:
            self.requests.put(SENTINEL)
        except Exception:
            pass
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            # Same discipline as terminate_pool: never wait unbounded on a
            # process that may be blocked inside a CUDA call.
            self.process.terminate()
            self.process.join(timeout=timeout)
        return not self.process.is_alive()
