"""Export existing search targets in fixed White perspective, without relabeling.

Processed mcts_values are SIDE-TO-MOVE; game_results are WHITE. This distinct
control uses recorded searches by v27/gen47-epoch11, not gen47 raw evaluations.
"""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from match_evidence import atomic_json,file_hash


def white_targets(values, side):
    if not np.isin(side,[-1,1]).all():
        raise ValueError('Turn plane must be exactly +/-1')
    if values.shape!=side.shape or not np.isfinite(values).all() or np.abs(values).max()>1.00001:
        raise ValueError('Invalid aligned search values')
    return (values*side).astype(np.float32)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'iterations/b2_001/processed24')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    p=np.load(args.data/'positions.npy',mmap_mode='r')
    values=np.load(args.data/'mcts_values.npy')
    weights=np.load(args.data/'value_weights.npy')
    if weights.shape!=values.shape or not np.isfinite(weights).all() or weights.min()<0:
        raise ValueError('Invalid value weights')
    target=white_targets(values,np.asarray(p[:,0,0,12]))
    output=args.out/'targets.npy'
    np.save(output,target)
    report=dict(complete=True,rows=len(target),output_hash=file_hash(output),
        source_hashes={name:file_hash(args.data/name) for name in
            ['positions.npy','mcts_values.npy','value_weights.npy','splits.npz','split_game_ids.json']},
        generation_manifest=file_hash(ROOT/'iterations/b2_001/manifest.json'),
        implementation=file_hash(__file__),
        conversion='side-to-move mcts_value multiplied by turn plane; no other shaping',
        caveat='mixed historical v27/gen47-epoch11 searches; estimates, not solved targets',
        weighted_rows=int(np.count_nonzero(weights)),
        zero_search_value_weighted_rows=int(np.count_nonzero((weights>0)&(values==0))),
        white_rows=int(np.count_nonzero(p[:,0,0,12]>0)),
        black_rows=int(np.count_nonzero(p[:,0,0,12]<0)))
    atomic_json(args.out/'complete.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
