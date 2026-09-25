"""Measure unsupported input activations without changing models or game rules."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from encoding import fen_to_tensor
from search_value_features import sparse_features,MAX_ACTIVE
from match_evidence import atomic_json,file_hash


def active_counts(p,relative):
    ids,weights=sparse_features(p,relative)
    valid=weights!=0
    return np.bincount(ids[valid],minlength=6240 if relative else 840)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    data=ROOT/'iterations/b2_001/processed24'
    positions=np.load(data/'positions.npy',mmap_mode='r')
    train=np.load(data/'splits.npz')['train']
    samples=json.loads((args.audit/'samples.json').read_text())['rows']
    report={'source':file_hash(args.audit/'samples.json'),'training_rows':len(train),'models':{}}
    for relative,name in [(False,'absolute'),(True,'relative')]:
        n=6240 if relative else 840
        counts=np.zeros(n,dtype=np.int64)
        for offset in range(0,len(train),8192):
            counts+=active_counts(positions[train[offset:offset+8192]],relative)
        groups={}
        for kind in ('stored_root','runtime_root','leaf'):
            rows=[r for r in samples if r['kind']==kind]
            if not rows:continue
            p=np.stack([fen_to_tensor(r['fen'],r['fen'].split()[1]=='w',r['pending'],24,r['turn_count']) for r in rows])
            ids,weights=sparse_features(p,relative)
            padded=np.pad(counts,(0,MAX_ACTIVE))
            unsupported=(padded[ids]==0)&(weights!=0)
            groups[kind]={'rows':len(rows),'rows_with_unseen_features':int(unsupported.any(axis=1).sum()),
                'unseen_feature_ids':np.unique(ids[unsupported]).tolist()}
        report['models'][name]={'never_active_in_train':np.flatnonzero(counts==0).tolist(),
                               'train_counts':counts.tolist(),'groups':groups}
    atomic_json(args.out,report)
    print(json.dumps({k:v['groups'] for k,v in report['models'].items()},indent=2))


if __name__=='__main__':main()
