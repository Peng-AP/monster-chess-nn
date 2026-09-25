"""Matched replay-only / 50% search-leaf fine-tuning of the absolute512 control.

Frozen gen47 raw targets, original family splits, no test/gate training. Both
arms have the same optimizer, steps and validation objective. New leaf states
are deduplicated and checked against opposite-split original input identities.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'native')]
from encoding import fen_to_tensor
from train_search_value import features,model,export
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease
import monster_native as native


def signatures(x):
    """Lossless identity for binary absolute features plus float32 budget."""
    if not np.isin(x[:,:839],[0,1]).all():raise ValueError('Nonbinary feature')
    bits=np.packbits(x[:,:839].astype(bool),axis=1)
    budget=np.ascontiguousarray(x[:,839],dtype=np.float32).view(np.uint8).reshape(-1,4)
    return [row.tobytes() for row in np.concatenate([bits,budget],axis=1)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arm',choices=['replay','leaf'],required=True)
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--seed',type=int,default=3273)
    args=ap.parse_args()
    if not json.loads((args.corpus/'complete.json').read_text())['complete']:raise ValueError('Incomplete corpus')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    data=ROOT/'iterations/b2_001/processed24'
    labels=ROOT/'benchmarks/search_first_20260911/compression_control/labels/teacher_values.npy'
    receipt=json.loads(labels.with_name('complete.json').read_text())
    corpus_manifest=json.loads((args.corpus/'manifest.json').read_text())
    if corpus_manifest['teacher_cache']!=receipt:raise ValueError('Corpus teacher provenance differs')
    for p,h in [(labels,receipt['labels']),(data/'positions.npy',receipt['positions']),
                (data/'splits.npz',receipt['split'])]:
        if file_hash(p)!=h:raise ValueError('Changed source')
    original=ROOT/'models/candidates/search_relative_absolute_001/epoch_007.pt'
    original_receipt=json.loads(original.with_name('complete.json').read_text())
    if file_hash(original.with_suffix('.bin'))!=original_receipt['model_sha256']:
        raise ValueError('Initialization binary changed')
    source_rows=json.loads((args.corpus/'samples.json').read_text())['rows']
    leaf_rows=[r for r in source_rows if r['kind']=='leaf']
    del source_rows
    with worker_lease():
        p=np.load(data/'positions.npy',mmap_mode='r')
        splits=dict(np.load(data/'splits.npz'))
        x=torch.empty((len(p),840),device='cuda')
        train_keys=set();held_keys=set()
        is_train=np.zeros(len(p),dtype=bool);is_train[splits['train']]=True
        for offset in range(0,len(p),8192):
            f=features(p[offset:offset+8192]);x[offset:offset+len(f)]=torch.from_numpy(f).cuda()
            for i,key in enumerate(signatures(f),offset):
                (train_keys if is_train[i] else held_keys).add(key)
        unique={'train':{},'val':{}}
        removed={'duplicate':0,'opposite_original':0,'cross_leaf':0}
        for row in leaf_rows:
            if row['split'] not in unique or row['pending']:raise ValueError('Invalid leaf split/phase')
            if not np.isfinite(row['teacher']) or abs(row['teacher'])>1.00001:
                raise ValueError('Invalid White teacher target')
            f=features(fen_to_tensor(row['fen'],row['fen'].split()[1]=='w',False,24,row['turn_count'])[None])[0]
            key=signatures(f[None])[0]
            forbidden=held_keys if row['split']=='train' else train_keys
            if key in forbidden:removed['opposite_original']+=1;continue
            if key in unique[row['split']]:removed['duplicate']+=1;continue
            unique[row['split']][key]=(f,float(row['teacher']))
        for key in unique['train'].keys() & unique['val'].keys():
            del unique['train'][key];removed['cross_leaf']+=1
        del train_keys,held_keys,leaf_rows
        leaf={}
        for split,rows in unique.items():
            if not rows:raise ValueError('Empty leaf split')
            values=list(rows.values())
            leaf[split]=(torch.from_numpy(np.stack([v[0] for v in values])).cuda(),
                         torch.tensor([v[1] for v in values],device='cuda'))
        del unique,values
        y=torch.from_numpy(np.load(labels)).cuda()
        w=torch.from_numpy(np.load(data/'value_weights.npy')).cuda()
        indices={k:torch.tensor(v,device='cuda') for k,v in splits.items()}
        net=model(512,32).cuda()
        net.load_state_dict(torch.load(original,weights_only=True,map_location='cuda'))
        optimizer=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
        manifest=dict(arm=args.arm,epochs=args.epochs,seed=args.seed,learning_rate=.0002,
            initialization=file_hash(original),corpus=file_hash(args.corpus/'samples.json'),
            labels=receipt,leaf_rows={s:len(v[1]) for s,v in leaf.items()},removed=removed,
            selection='minimum 0.5 original validation weighted MSE + 0.5 unique leaf validation MSE',
            recipe='2048 replay + 2048 random leaf rows per step; replay control substitutes random replay rows',
            implementation=file_hash(__file__),runtime=file_hash(ROOT/'native/monster_native.pyd'))
        atomic_json(args.out/'manifest.json',manifest)
        def evaluate():
            net.eval();num=mass=leaf_sum=0.
            with torch.no_grad():
                for ids in indices['val'].split(4096):
                    e=(net(x[ids]).ravel()-y[ids]).square()
                    num+=float((e*w[ids]).sum());mass+=float(w[ids].sum())
                lx,ly=leaf['val']
                for offset in range(0,len(ly),4096):
                    leaf_sum+=float((net(lx[offset:offset+4096]).ravel()-ly[offset:offset+4096]).square().sum())
            a=num/mass;b=leaf_sum/len(ly)
            return dict(root_mse=a,leaf_mse=b,selection=.5*(a+b))
        history=[dict(epoch=0,**evaluate())];best=float('inf')
        for epoch in range(1,args.epochs+1):
            net.train()
            order=indices['train'][torch.randperm(len(indices['train']),device='cuda')]
            for ids in order.split(2048):
                optimizer.zero_grad(set_to_none=True)
                if args.arm=='leaf':
                    lx,ly=leaf['train'];other=torch.randint(len(ly),(len(ids),),device='cuda')
                    bx,by,bw=lx[other],ly[other],None
                else:
                    other=indices['train'][torch.randint(len(indices['train']),(len(ids),),device='cuda')]
                    bx,by,bw=x[other],y[other],w[other]
                pred=net(torch.cat([x[ids],bx])).ravel()
                e=(pred[:len(ids)]-y[ids]).square()
                second=(pred[len(ids):]-by).square()
                loss=.5*(e*w[ids]).sum()/w[ids].sum()
                loss+=.5*(second.mean() if bw is None else (second*bw).sum()/bw.sum())
                loss.backward();optimizer.step()
            score=evaluate();history.append(dict(epoch=epoch,seconds=time.monotonic()-started,**score))
            torch.save(net.state_dict(),args.out/f'epoch_{epoch:03}.pt')
            export(net,args.out/f'epoch_{epoch:03}.bin')
            if score['selection']<best:best=score['selection'];best_epoch=epoch
            atomic_json(args.out/'progress.json',dict(history=history,best_epoch=best_epoch))
            print(json.dumps(history[-1]),flush=True)
        path=args.out/f'epoch_{best_epoch:03}.bin'
        net.load_state_dict(torch.load(path.with_suffix('.pt'),weights_only=True))
        evaluator=native.CheapValue(str(path))
        # Compare exported evaluation on saved diagnostic FENs before games.
        states=json.loads((ROOT/'benchmarks/search_first_20260911/cpu_eval_after.json').read_text())['states']
        f=features(np.stack([fen_to_tensor(f,f.split()[1]=='w',h,24,c) for f,h,c in states]))
        expected=net(torch.from_numpy(f).cuda()).ravel().detach().cpu().numpy()
        actual=np.array([evaluator.evaluate(*s) for s in states])
        parity=float(abs(actual-expected).max())
        if parity>1e-5:raise ValueError('Export parity failure')
        peak=torch.cuda.max_memory_allocated()
        if peak>12*1024**3:raise ValueError('VRAM limit exceeded')
        atomic_json(args.out/'complete.json',dict(complete=True,best_epoch=best_epoch,
            model_sha256=file_hash(path),selection=best,parity_max_error=parity,
            seconds=time.monotonic()-started,cuda_peak_bytes=peak,
            note='No test-based selection; both arms require actual play-testing'))


if __name__=='__main__':main()
