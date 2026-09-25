"""Diagnostic only: actual NN-leaf compression error versus training roots.

Uses train/validation roots and full recorded repetition history. Never consumes
test roots or human/gate positions as new training. Requires opt-in native leaf
sampling, installed only AFTER the frozen representation campaign completes.
"""
import argparse
import bisect
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'native')]
import chess
import monster_native as native
from encoding import fen_to_tensor
from train import load_model_for_inference
from match_evidence import atomic_json,file_hash
from stateful_generation import restore_state
from search_first_match import prior_keys
from distill_search_value import teacher_input
from worker_lease import worker_lease


def root_map(audit):
    ends=[];entries=[];offset=0
    for split,paths in audit['splits'].items():
        for path in paths:
            size=audit['row_counts'][path]
            entries.append((split,path,offset))
            offset+=size;ends.append(offset)
    return ends,entries


def error_summary(rows,model):
    if not rows:return None
    errors=np.array([r[model]-r['teacher'] for r in rows])
    return dict(rows=len(rows),mse=float(np.mean(errors**2)),mae=float(np.mean(abs(errors))),
                mean_error=float(np.mean(errors)),p90_absolute_error=float(np.quantile(abs(errors),.9)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--roots-per-split',type=int,default=128)
    ap.add_argument('--val-roots',type=int,help='Optional independent validation root count')
    ap.add_argument('--nodes',type=int,default=100000)
    ap.add_argument('--leaves',type=int,default=64)
    ap.add_argument('--seed',type=int,default=9175)
    args=ap.parse_args()
    if not 1<=args.leaves<=4096 or args.roots_per_split<1 or args.nodes<1:
        raise ValueError('Positive bounded sample limits required')
    campaign=ROOT/'benchmarks/search_first_relative_20260911'
    if not json.loads((campaign/'summary.json').read_text())['complete']:
        raise ValueError('Frozen campaign must finish before this diagnostic')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    data=ROOT/'iterations/b2_001/processed24'
    raw=ROOT/'iterations/b2_001/raw'
    audit=json.loads((data/'b2_audit.json').read_text())
    ends,entries=root_map(audit)
    splits=dict(np.load(data/'splits.npz'))
    positions=np.load(data/'positions.npy',mmap_mode='r')
    if ends[-1]!=len(positions):raise ValueError('Raw/processed row count mismatch')
    labels=ROOT/'benchmarks/search_first_20260911/compression_control/labels/teacher_values.npy'
    label_receipt=json.loads(labels.with_name('complete.json').read_text())
    if file_hash(labels)!=label_receipt['labels'] or file_hash(data/'positions.npy')!=label_receipt['positions']:
        raise ValueError('Source corpus or teacher labels changed')
    cached_targets=np.load(labels,mmap_mode='r')
    models={}
    for name in ('absolute','relative'):
        directory=ROOT/'models/candidates'/f'search_relative_{name}_001'
        receipt=json.loads((directory/'complete.json').read_text())
        path=directory/f"epoch_{receipt['best_epoch']:03}.bin"
        if file_hash(path)!=receipt['model_sha256']:raise ValueError('Changed model')
        models[name]=path
    rng=np.random.default_rng(args.seed)
    selected={s:sorted(int(i) for i in rng.choice(splits[s],
              args.val_roots if s=='val' and args.val_roots is not None else args.roots_per_split,replace=False))
              for s in ('train','val')}
    manifest=dict(runtime=file_hash(ROOT/'native/monster_native.pyd'),
        models={k:dict(path=str(p),hash=file_hash(p)) for k,p in models.items()},
        roots=selected,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        sources={n:file_hash(data/n) for n in ('b2_audit.json','splits.npz','split_game_ids.json')},
        teacher_cache=label_receipt,implementation=file_hash(__file__),
        note='diagnostic only; fixed-node sampler; no new training and no test/human/gate roots')
    atomic_json(args.out/'manifest.json',manifest)
    rows=[];roots=[]
    started=time.monotonic()
    with worker_lease():
        evaluators={k:native.CheapValue(str(p)) for k,p in models.items()}
        for split,indices in selected.items():
            last_path=None
            for index in indices:
                slot=bisect.bisect_right(ends,index)
                actual_split,path,start=entries[slot]
                if actual_split!=split:raise ValueError('Split mapping mismatch')
                source=raw/path
                if source!=last_path:
                    expected=audit['sources'].get(str(Path(path)),audit['sources'].get(path))
                    if file_hash(source)!=expected:raise ValueError('Raw source changed')
                    records=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
                    last_path=source
                record=records[index-start]
                g,tracker=restore_state(record['state'])
                p=fen_to_tensor(record['fen'],g.is_white_turn,g.white_half_pending,24,g.turn_count)
                if not np.array_equal(p,positions[index]):raise ValueError('Processed/raw input mismatch')
                root=dict(index=index,split=split,source=path,row=index-start,state=record['state'])
                roots.append(root)
                if g.is_terminal() or tracker.fired_at is not None:
                    root['skipped']='terminal';continue
                raw_fen=g.board.fen(en_passant='fen')
                r=native.alphabeta_search(raw_fen,pending=g.white_half_pending,turn_count=g.turn_count,
                    evaluator=evaluators['absolute'],prior_positions=prior_keys(tracker,g),
                    node_limit=args.nodes,max_depth=8,seconds=60,
                    collect_leaves=args.leaves,leaf_seed=args.seed+index)
                if r.interrupted and r.nodes<args.nodes:raise TimeoutError('Sampler hit time rather than fixed node bound')
                root.update(nodes=r.nodes,depth=r.completed_depth,nn_evaluations=r.leaf_evaluations,
                            samples=len(r.leaf_samples))
                for kind,fen,target in [('stored_root',record['fen'],float(cached_targets[index])),
                                        ('runtime_root',raw_fen,float(cached_targets[index]))]:
                    row=dict(root=index,split=split,kind=kind,fen=fen,pending=g.white_half_pending,
                             turn_count=g.turn_count,teacher=target)
                    row.update({name:e.evaluate(fen,g.white_half_pending,g.turn_count) for name,e in evaluators.items()})
                    rows.append(row)
                for fen,pending,count,value in r.leaf_samples:
                    if pending:raise ValueError('Sampled a pending White leaf')
                    row=dict(root=index,split=split,kind='leaf',fen=fen,pending=pending,turn_count=count,
                        absolute=value,relative=evaluators['relative'].evaluate(fen,pending,count))
                    fields=fen.split();fields[3]=chess.Board(fen).fen().split()[3]
                    projected=' '.join(fields)
                    row['ep_projection_changed']=projected!=fen
                    row['absolute_ep_projected']=evaluators['absolute'].evaluate(projected,pending,count)
                    rows.append(row)
                if len(roots)%32==0:
                    atomic_json(args.out/'progress.json',dict(roots=len(roots),rows=len(rows),seconds=time.monotonic()-started))
                    print(f'{len(roots)} roots, {len(rows)} rows',flush=True)
        teacher=ROOT/'models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
        if file_hash(teacher)!=label_receipt['teacher']:raise ValueError('Teacher changed')
        net,_=load_model_for_inference(str(teacher),torch.device('cuda'));net.eval()
        leaves=[r for r in rows if r['kind']=='leaf']
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for offset in range(0,len(leaves),512):
                chunk=leaves[offset:offset+512]
                p=np.stack([fen_to_tensor(r['fen'],r['fen'].split()[1]=='w',r['pending'],24,r['turn_count']) for r in chunk])
                v,_=net(torch.from_numpy(teacher_input(p,net.input_channels)).cuda())
                v=v.squeeze(-1).float().cpu().numpy()*p[:,0,0,12]
                for row,value in zip(chunk,v):row['teacher']=float(value)
        atomic_json(args.out/'samples.json',dict(roots=roots,rows=rows))
        report=dict(complete=True,seconds=time.monotonic()-started,roots=len(roots),leaves=len(leaves),groups={})
        for split in ('train','val'):
            for kind in ('stored_root','runtime_root','leaf'):
                for side in ('w','b'):
                    subset=[r for r in rows if r['split']==split and r['kind']==kind and r['fen'].split()[1]==side
                            and not r['pending']]
                    report['groups'][f'{split}/{kind}/{side}']={m:error_summary(subset,m) for m in models}
        changed=[r for r in leaves if r['ep_projection_changed']]
        report['ep_projection']=dict(changed_leaves=len(changed),
            mean_absolute_prediction_change=float(np.mean([abs(r['absolute']-r['absolute_ep_projected']) for r in changed])) if changed else 0)
        atomic_json(args.out/'complete.json',report)
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
