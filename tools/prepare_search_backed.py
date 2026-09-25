"""Deduplicate target inputs, enforce split isolation, construct sibling ranking pairs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from encoding import fen_to_tensor
from train_search_value import features
from train_search_leaves import signatures
from match_evidence import atomic_json,file_hash
from generate_search_targets import DATA,TEACHER

def make_pairs(groups,index,backed):
    pairs={}
    for sign,children in groups:
        children=[(index[k],v) for k,v in children if k in index]
        if len(children)<2:continue
        best,value=max(children,key=lambda p:sign*p[1])
        for other,v in children:
            gap=sign*(value-v)
            # Deduplicated value targets must still agree with this local ranking.
            if other==best or gap<.05 or sign*(backed[best]-backed[other])<.05:continue
            key=(best,other,sign);pairs[key]=min(.25,gap)
    return np.array([(a,b,s,m) for (a,b,s),m in pairs.items()],dtype=np.float32).reshape(-1,4)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--corpus',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    receipt=json.loads((args.corpus/'complete.json').read_text())
    if not receipt['complete'] or file_hash(args.corpus/'groups.jsonl')!=receipt['groups_hash']:
        raise ValueError('Invalid generation receipt')
    manifest=json.loads((args.corpus/'manifest.json').read_text())
    if any(file_hash(p)!=h for p,h in manifest['hashes'].items()):raise ValueError('Generation source changed')
    args.out.mkdir(parents=True,exist_ok=False)
    originals={'train':set(),'held':set()};p=np.load(DATA/'positions.npy',mmap_mode='r')
    splits=dict(np.load(DATA/'splits.npz'));is_train=np.zeros(len(p),bool);is_train[splits['train']]=True
    for offset in range(0,len(p),8192):
        f=features(p[offset:offset+8192])
        for i,key in enumerate(signatures(f),offset):originals['train' if is_train[i] else 'held'].add(key)
    collected={'train':{},'val':{}};groups={'train':[],'val':[]};excluded=duplicates=0
    for line in (args.corpus/'groups.jsonl').read_text().splitlines():
        group=json.loads(line)
        if 'skip' in group:continue
        split=group['split'];children=[]
        if split not in collected:raise ValueError('Invalid split')
        for row in group['records']:
            if row['pending']:raise ValueError('Pending White is not a static-value training state')
            if any(not np.isfinite(row[k]) or abs(row[k])>1.00001 for k in ('raw','backed')):raise ValueError('Invalid target')
            f=features(fen_to_tensor(row['fen'],row['fen'].split()[1]=='w',False,24,row['turn_count'])[None])[0]
            key=signatures(f[None])[0]
            if key in originals['held' if split=='train' else 'train']:excluded+=1;continue
            if key not in collected[split]:
                collected[split][key]=dict(feature=f,raw=[],backed=[],origins=set())
            else:duplicates+=1
            item=collected[split][key];item['raw'].append(row['raw']);item['backed'].append(row['backed'])
            item['origins'].add(group['root'])
            if not row['is_root']:children.append((key,row['backed']))
        groups[split].append((group['sign'],children))
    shared=collected['train'].keys() & collected['val'].keys()
    for key in shared:del collected['train'][key]
    report=dict(complete=False,corpus_hash=receipt['groups_hash'],duplicate_rows=duplicates,
        opposite_original_removed=excluded,cross_new_removed=len(shared),splits={},
        source_manifest=file_hash(args.corpus/'manifest.json'),implementation=file_hash(__file__))
    provenance={}
    for split,items in collected.items():
        if len(items)<16:raise ValueError('Insufficient clean targets')
        values=list(items.values());index={k:i for i,k in enumerate(items)}
        x=np.stack([r['feature'] for r in values]);raw=np.array([np.mean(r['raw']) for r in values],np.float32)
        backed=np.array([np.mean(r['backed']) for r in values],np.float32)
        pairs=make_pairs(groups[split],index,backed)
        if len(pairs)<4:raise ValueError('Insufficient nontrivial sibling comparisons')
        np.savez(args.out/(split+'.npz'),features=x,raw=raw,backed=backed,pairs=pairs)
        provenance[split]=[sorted(r['origins']) for r in values]
        spans=[max(r['backed'])-min(r['backed']) for r in values]
        report['splits'][split]=dict(rows=len(x),white=int(x[:,768].sum()),black=int(x[:,770].sum()),
            pairs=len(pairs),pair_white=int((pairs[:,2]>0).sum()),pair_black=int((pairs[:,2]<0).sum()),
            raw_backed_mse=float(np.mean((raw-backed)**2)),raw_backed_mae=float(np.mean(abs(raw-backed))),
            repeated_input_span_gt_025=sum(s>.25 for s in spans),max_repeated_input_span=max(spans),
            hash=file_hash(args.out/(split+'.npz')))
    atomic_json(args.out/'provenance.json',provenance)
    report.update(complete=True,provenance_hash=file_hash(args.out/'provenance.json'),
        note='One target per exact input; repeated labels averaged, history-conditioned conflicts reported; original replay unchanged')
    atomic_json(args.out/'complete.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
