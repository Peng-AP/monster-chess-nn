"""Isolated transfer experiments; no production switch or search changes."""
import argparse
import concurrent.futures as futures
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from match_evidence import atomic_json, runtime_identity
from worker_lease import worker_lease

_barrier = None


def initialize(barrier):
    global _barrier
    _barrier = barrier


def variant(evaluator, pinned=False, packed=False):
    import numpy as np
    import torch
    from native_mcts import _GraphedForward
    from config import POLICY_TEMPERATURE
    channels = evaluator.input_channels
    graph = _GraphedForward(torch,evaluator.model,evaluator.device,16,channels,True)
    buffers = {}
    def call(buf,n,chans):
        array = np.frombuffer(buf,dtype=np.float32).reshape(n,chans,8,8)
        if pinned:
            if n not in buffers:
                host = torch.empty((n,chans,8,8),dtype=torch.float32,pin_memory=True)
                gpu = torch.empty_like(host,device=evaluator.device)
                buffers[n] = host,host.numpy(),gpu
            host,view,gpu = buffers[n]
            np.copyto(view,array)
            gpu.copy_(host,non_blocking=True)
            tensor = gpu.half()
        else:
            tensor = torch.from_numpy(array.copy()).to(evaluator.device).half()
        value,policy,_ = graph(tensor,n)
        policy = policy.reshape(n,-1).float()/POLICY_TEMPERATURE
        value = value.reshape(-1).float()
        if packed:
            merged = torch.cat((value,policy.reshape(-1))).cpu().numpy()
            return merged[:n].tobytes(),merged[n:].tobytes()
        return value.cpu().numpy().astype(np.float32).tobytes(),policy.cpu().numpy().astype(np.float32).tobytes()
    return call


def worker(index):
    import numpy as np
    import torch
    from evaluation import NNEvaluator
    from native_mcts import make_bridge
    from encoding import fen_to_tensor
    from monster_chess import MonsterChessGame
    torch.set_num_threads(1)
    evaluator=NNEvaluator('models/candidates/bootstrap_main_gen_0047/arena_selected.pt')
    reference,channels=make_bridge(evaluator,graph_width=16)
    board=fen_to_tensor(MonsterChessGame().fen(),input_channels=channels).transpose(2,0,1)
    buffers={n:np.repeat(board[None],n,axis=0).astype(np.float32).tobytes() for n in (1,4,8,16)}
    callbacks={'reference':reference,'pinned':variant(evaluator,pinned=True),
               'packed':variant(evaluator,packed=True),'both':variant(evaluator,pinned=True,packed=True)}
    expected={n:reference(buf,n,channels) for n,buf in buffers.items()}
    for callback in callbacks.values():
        for n,buf in buffers.items():
            for _ in range(3):
                if callback(buf,n,channels)!=expected[n]:
                    raise ValueError('variant differs from production output bytes')
    results=[]
    for mode in ('reference','pinned','packed','both','both','packed','pinned','reference'):
        _barrier.wait(timeout=120)
        t=time.perf_counter()
        for _ in range(20):
            for n,buf in buffers.items():
                if callbacks[mode](buf,n,channels)!=expected[n]:
                    raise ValueError('variant output changed during timing')
        results.append(dict(mode=mode,seconds=time.perf_counter()-t,callbacks=80))
    return dict(worker=index,results=results,peak_reserved=torch.cuda.max_memory_reserved())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    if not 1<=args.workers<=8 or Path(args.output).exists():
        raise ValueError('invalid workers or output already exists')
    ctx=mp.get_context('spawn')
    with worker_lease():
        pool=futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx,
                                         initializer=initialize,initargs=(ctx.Barrier(args.workers),))
        pending=[pool.submit(worker,i) for i in range(args.workers)]
        try:
            results=[f.result(timeout=600) for f in pending]
        except BaseException:
            from data_generation import terminate_pool
            terminate_pool(pool)
            raise
        else:
            pool.shutdown(wait=True)
    atomic_json(args.output,dict(runtime=runtime_identity(),workers=args.workers,
                                exact_bytes=True,results=results,
                                caveat='Fixed initial-position inference batches, not a full-game speedup.'))
    for mode in ('reference','pinned','packed','both'):
        elapsed=sum(r['seconds'] for w in results for r in w['results'] if r['mode']==mode)
        print(mode,elapsed/(2*args.workers))


if __name__=='__main__':
    mp.freeze_support()
    main()
