"""Paired whole-game throughput and exact-record parity for graph reuse."""
import argparse
import concurrent.futures as futures
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from match_evidence import atomic_json, digest, file_hash, runtime_identity
from worker_lease import worker_lease
from stateful_generation import init_worker, play_task


def one(task):
    started = time.perf_counter()
    rows = play_task(task)
    return dict(id=task['id'], seconds=time.perf_counter()-started, records=len(rows),
                sha256=digest(rows), result=rows[0]['game_result'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--games', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--sims', type=int, default=700)
    ap.add_argument('--output', required=True)
    ap.add_argument('--feature', choices=('graph-cache','pinned-input'), default='graph-cache')
    args = ap.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if not 1 <= args.workers <= 8 or args.games < 1:
        raise ValueError('invalid worker/game count')
    model = 'models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
    tasks = [dict(id=f'parity_{i}', kind='selfplay', model=model, sims=args.sims, seed=1800000000+i) for i in range(args.games)]
    report = dict(config=vars(args), model_sha256=file_hash(model), runtime=runtime_identity(),
                  implementation=file_hash(__file__), runs=[])
    feature_env = 'MONSTER_CUDA_GRAPH_CACHE' if args.feature == 'graph-cache' else 'MONSTER_PINNED_INPUT'
    previous = os.environ.get(feature_env)
    try:
        with worker_lease():
            for mode in ('0','1','1','0'):
                os.environ[feature_env] = mode
                started = time.perf_counter()
                pool = futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                                  initializer=init_worker, initargs=([model],))
                pending = {pool.submit(one, task) for task in tasks}
                rows = []
                try:
                    while pending:
                        done, pending = futures.wait(pending, timeout=600, return_when=futures.FIRST_COMPLETED)
                        if not done:
                            raise TimeoutError('whole-game validation stalled')
                        rows.extend(f.result() for f in done)
                        print(f"cache={mode} {len(rows)}/{args.games}", flush=True)
                except BaseException:
                    from data_generation import terminate_pool
                    terminate_pool(pool)
                    raise
                else:
                    pool.shutdown(wait=True)
                report['runs'].append(dict(mode=mode, elapsed_seconds=time.perf_counter()-started,
                                           games=sorted(rows, key=lambda r:r['id'])))
                atomic_json(args.output, report)
        reference = {r['id']:r['sha256'] for r in report['runs'][0]['games']}
        report['exact_record_parity'] = all({r['id']:r['sha256'] for r in run['games']} == reference for run in report['runs'])
        times = {mode:sum(r['elapsed_seconds'] for r in report['runs'] if r['mode']==mode)/2 for mode in ('0','1')}
        report['mean_elapsed_seconds'] = times
        report['speedup'] = times['0']/times['1']
        atomic_json(args.output, report)
        print(json.dumps({k:report[k] for k in ('exact_record_parity','mean_elapsed_seconds','speedup')}, indent=2))
        if not report['exact_record_parity']:
            raise RuntimeError('graph reuse changed whole-game records')
    finally:
        if previous is None:
            os.environ.pop(feature_env, None)
        else:
            os.environ[feature_env] = previous


if __name__ == '__main__':
    mp.freeze_support()
    main()
