"""Warm persistent inference profiling with explicit CPU and CUDA activities."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from match_evidence import atomic_json, model_identity, runtime_identity
from worker_lease import worker_lease


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import torch
    import numpy as np
    from evaluation import NNEvaluator
    from native_mcts import make_bridge
    from encoding import fen_to_tensor
    from monster_chess import MonsterChessGame
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA profiling requires the production GPU')
    model = 'models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
    torch.set_num_threads(1)
    with worker_lease():
        evaluator = NNEvaluator(model)
        bridge, channels = make_bridge(evaluator, graph_width=16)
        board = fen_to_tensor(MonsterChessGame().fen(), input_channels=channels)
        # Native callback layout is NCHW, not the encoder's HWC.
        buffers = {n:np.repeat(board.transpose(2,0,1)[None], n, axis=0).astype(np.float32).tobytes() for n in (1,4,8,16)}
        for n, buf in buffers.items():
            for _ in range(5):
                bridge(buf,n,channels)
        timings = {}
        for n, buf in buffers.items():
            t = time.perf_counter()
            for _ in range(100):
                bridge(buf,n,channels)
            timings[n] = (time.perf_counter()-t)/100
        activities = [torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            for n, buf in buffers.items():
                for _ in range(4):
                    with torch.profiler.record_function(f'bridge_batch_{n}'):
                        bridge(buf,n,channels)
        output.parent.mkdir(parents=True,exist_ok=True)
        profiler.export_chrome_trace(str(output.with_suffix('.trace.json')))
        averages = profiler.key_averages()
        entries = [dict(name=e.key,count=e.count,cpu_us=e.cpu_time_total,
                        self_cpu_us=e.self_cpu_time_total,
                        device_us=getattr(e,'device_time_total',0),
                        self_device_us=getattr(e,'self_device_time_total',0)) for e in averages]
        atomic_json(output,dict(model=model_identity(model),runtime=runtime_identity(),
                                steady_seconds_per_callback=timings,events=entries,
                                note='Single-worker warm inference; compare with multiworker wall measurements, not as a prediction of campaign speed.'))
        print(json.dumps(timings,indent=2))
        print(averages.table(sort_by='self_cpu_time_total',row_limit=20))


if __name__ == '__main__':
    main()
