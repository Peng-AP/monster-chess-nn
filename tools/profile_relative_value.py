"""Trained-model numerical parity and same-state evaluator/search cost."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'native')]
import monster_native as native
from encoding import fen_to_tensor
from train_search_value import model
from search_value_features import dense_features
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--absolute',type=Path,required=True)
    ap.add_argument('--relative',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    torch.set_num_threads(1)
    source=ROOT/'benchmarks/search_first_20260911/cpu_eval_after.json'
    states=json.loads(source.read_text())['states']
    positions=np.stack([fen_to_tensor(f,f.split()[1]=='w',h,24,c) for f,h,c in states])
    report=dict(complete=False,runtime=file_hash(ROOT/'native/monster_native.pyd'),
                state_source=file_hash(source),states=len(states),arms={})
    with worker_lease():
        for label,directory,relative in [('absolute',args.absolute,False),('relative',args.relative,True)]:
            receipt=json.loads((directory/'complete.json').read_text())
            path=directory/f"epoch_{receipt['best_epoch']:03}.bin"
            if file_hash(path)!=receipt['model_sha256']:raise ValueError('Model changed')
            net=model(512,32,6240 if relative else 840).eval()
            net.load_state_dict(torch.load(path.with_suffix('.pt'),weights_only=True,map_location='cpu'))
            evaluator=native.CheapValue(str(path))
            with torch.no_grad():expected=net(torch.from_numpy(dense_features(positions,relative))).numpy().ravel()
            actual=np.array([evaluator.evaluate(*state) for state in states])
            error=float(np.max(np.abs(actual-expected)))
            if error>1e-5:raise ValueError(f'Native/Torch parity failure: {error}')
            times=[]
            for _ in range(5):
                started=time.perf_counter()
                for _ in range(10):
                    for state in states:evaluator.evaluate(*state)
                times.append((time.perf_counter()-started)/(10*len(states)))
            searches=[]
            for fen,pending,count in states[::10][:12]:
                r=native.alphabeta_search(fen,pending=pending,turn_count=count,seconds=.3,evaluator=evaluator)
                searches.append(dict(depth=r.completed_depth,nodes=r.nodes,seconds=r.elapsed_seconds))
            row=dict(model=str(path),model_hash=file_hash(path),max_error=error,
                     median_eval_seconds=float(np.median(times)),searches=searches)
            report['arms'][label]=row
            print(label,error,row['median_eval_seconds'],flush=True)
    ratio=report['arms']['relative']['median_eval_seconds']/report['arms']['absolute']['median_eval_seconds']
    report.update(complete=True,inference_cost_ratio=ratio,cost_target_met=ratio<=1.5)
    atomic_json(args.out,report)


if __name__=='__main__':main()
