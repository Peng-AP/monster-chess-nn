"""D3 stage 1: the Python side of the native search's NN bridge.

The native search fills an (N, C, 8, 8) f32 buffer and hands it over as raw
bytes; this returns values and policy logits the same way. Bytes rather than
Python lists because a batch of 16 is ~17k floats and list marshalling would
cost more than the search it serves.

The model, the checkpoint loading and the FP16 path are untouched — that is the
point of stage 1. Only the *caller* moved.

Note the native side applies the pre-NN clamps itself (king absence, the
side-to-move capture scan), exactly as `NNEvaluator._batch_impl` does, so
decided positions never reach this function.

**The value returned here is in the SIDE-TO-MOVE perspective**, exactly as the
model emits it. The native search converts to White's perspective itself,
because it is what knows each leaf's side. Do not convert here: doing it in
both places restores the bug it fixes.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def make_eval_fn(model_path):
    """Return (eval_fn, input_channels) for `monster_native.Tree.run_batched_puct`."""
    from evaluation import NNEvaluator

    evaluator = NNEvaluator(model_path)
    torch = evaluator.torch
    channels = evaluator.input_channels

    def eval_fn(buf, n, chans):
        array = np.frombuffer(buf, dtype=np.float32).reshape(n, chans, 8, 8)
        tensor = torch.from_numpy(array.copy()).to(evaluator.device)
        if getattr(evaluator, "_half", False):
            tensor = tensor.half()
        with torch.no_grad():
            value, policy = evaluator.model(tensor)
        values = value.reshape(-1).float().cpu().numpy().astype(np.float32)
        policies = policy.reshape(n, -1).float().cpu().numpy().astype(np.float32)
        return values.tobytes(), policies.tobytes()

    return eval_fn, channels
