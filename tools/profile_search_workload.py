"""Representative fixed-state production profiler; measurements never train models."""
import argparse
from collections import Counter
import concurrent.futures as futures
import cProfile
import io
import json
import multiprocessing as mp
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from match_evidence import atomic_json, file_hash, model_identity, runtime_identity
from worker_lease import worker_lease

_model = None
_load_seconds = 0


def initialize(model):
    global _model, _load_seconds
    import torch
    from evaluation import NNEvaluator
    torch.set_num_threads(1)
    started = time.perf_counter()
    _model = NNEvaluator(model)
    _load_seconds = time.perf_counter() - started


def workload(task):
    import native_mcts as native
    from stateful_generation import restore_state
    import torch
    metrics = Counter()
    batches = Counter()
    original_capture = native._GraphedForward._capture
    def capture(graph, n):
        t = time.perf_counter()
        result = original_capture(graph, n)
        metrics['capture_seconds'] += time.perf_counter() - t
        metrics['captures'] += 1
        return result
    native._GraphedForward._capture = capture
    profiler = cProfile.Profile()
    profiler.enable()
    decisions = []
    started = time.perf_counter()
    try:
        for sims in task['sims']:
            t = time.perf_counter()
            engine = native.NativeMCTS(num_simulations=sims, eval_fn=_model,
                                       allow_early_stop=False, root_noise=False, seed=task['seed'])
            metrics['engine_create_seconds'] += time.perf_counter()-t
            bridge = engine._bridge
            def timed_bridge(buf, n, channels):
                t = time.perf_counter()
                result = bridge(buf, n, channels)
                metrics['callback_seconds'] += time.perf_counter()-t
                batches[n] += 1
                return result
            engine._bridge = timed_bridge
            for label, record in task['positions']:
                t = time.perf_counter()
                game, _ = restore_state(record['state'])
                metrics['restore_seconds'] += time.perf_counter()-t
                engine._reuse_tree = None
                engine._reuse_key = None
                t = time.perf_counter()
                move, policy, value = engine.get_best_action(game, temperature=0)
                duration = time.perf_counter()-t
                metrics['search_seconds'] += duration
                decisions.append(dict(phase=label, sims=sims, action=move.uci() if move else None,
                                      policy=policy, value=value, seconds=duration))
    finally:
        native._GraphedForward._capture = original_capture
        profiler.disable()
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats('cumulative').print_stats(30)
    return dict(source=task['source'], model_load_seconds=_load_seconds,
                total_seconds=time.perf_counter()-started, metrics=dict(metrics),
                batches=dict(batches), decisions=decisions, profile=output.getvalue(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='models/candidates/bootstrap_main_gen_0047/arena_selected.pt')
    ap.add_argument('--source', default='iterations/gen_0047/raw/selfplay')
    ap.add_argument('--games', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--sims', type=int, nargs='+', default=[700,3200])
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if not 1 <= args.workers <= 8 or args.games <= 0:
        raise ValueError('workers1..8 and positive games required')
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    tasks = []
    for path in sorted(Path(args.source).glob('*.jsonl'))[:args.games]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) < 3:
            continue
        positions = [(label, rows[index]) for label,index in [('early',0),('middle',len(rows)//2),('late',max(0,len(rows)-8))]]
        tasks.append(dict(source=str(path), positions=positions, sims=args.sims, seed=12345+len(tasks)))
    if not tasks:
        raise ValueError('no stateful source games')
    manifest = dict(config=vars(args), model=model_identity(args.model), runtime=runtime_identity(),
                    implementation=file_hash(__file__), sources={t['source']:file_hash(t['source']) for t in tasks})
    started = time.perf_counter()
    results = []
    with worker_lease():
        pool = futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                          initializer=initialize, initargs=(args.model,))
        pending = {pool.submit(workload, task) for task in tasks}
        try:
            while pending:
                done, pending = futures.wait(pending, timeout=600, return_when=futures.FIRST_COMPLETED)
                if not done:
                    raise TimeoutError('profile stalled')
                for future in done:
                    result = future.result()
                    results.append(result)
                    print(f"{len(results)}/{len(tasks)} {result['source']}: {result['metrics']}", flush=True)
        except BaseException:
            from data_generation import terminate_pool
            terminate_pool(pool)
            raise
        else:
            pool.shutdown(wait=True)
    aggregate = Counter()
    for result in results:
        aggregate.update(result['metrics'])
    atomic_json(args.output, dict(manifest=manifest, elapsed_seconds=time.perf_counter()-started,
                                 totals=dict(aggregate), results=results,
                                 note='CPU wall measurements include GPU synchronization/contending waits; cProfile adds overhead.'))
    print(json.dumps(dict(aggregate), indent=2))


if __name__ == '__main__':
    mp.freeze_support()
    main()
