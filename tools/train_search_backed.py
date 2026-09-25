"""Matched raw / searched / searched+ranking targets, unchanged absolute512 evaluator."""
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
sys.path[:0]=[str(ROOT/'native'),str(ROOT/'src'),str(ROOT/'tools')]
import monster_native as native
from train_search_value import features,model,export
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease
from generate_search_targets import DATA,VALUE,TEACHER

def ranking_loss(pred_a,pred_b,sign,margin):
    return torch.relu(margin-sign*(pred_a-pred_b)).square().mean()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--arm',choices=['raw','backed','ranked'],required=True)
    ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--steps',type=int,help='Tiny rehearsal only')
    args=ap.parse_args()
    prepared=json.loads((args.data/'complete.json').read_text())
    if not prepared['complete']:raise ValueError('Missing prepared data receipt')
    labels=ROOT/'benchmarks/search_first_20260911/compression_control/labels/teacher_values.npy'
    label_receipt=json.loads(labels.with_name('complete.json').read_text())
    for path,key in ((labels,'labels'),(DATA/'positions.npy','positions'),(DATA/'splits.npz','split'),(TEACHER,'teacher')):
        if file_hash(path)!=label_receipt[key]:raise ValueError('Replay teacher provenance changed')
    initial=VALUE.with_suffix('.pt');old=json.loads(VALUE.with_name('complete.json').read_text())
    if file_hash(VALUE)!=old['model_sha256']:raise ValueError('CPU initialization changed')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(913327);np.random.seed(913327)
    torch.use_deterministic_algorithms(True);started=time.monotonic()
    with worker_lease():
        p=np.load(DATA/'positions.npy',mmap_mode='r');x=torch.empty((len(p),840),device='cuda')
        for offset in range(0,len(p),8192):
            f=features(p[offset:offset+8192]);x[offset:offset+len(f)]=torch.from_numpy(f).cuda()
        y=torch.from_numpy(np.load(labels)).cuda();w=torch.from_numpy(np.load(DATA/'value_weights.npy')).cuda()
        splits={s:torch.tensor(ids,device='cuda') for s,ids in dict(np.load(DATA/'splits.npz')).items()}
        data={};side_ids={}
        for split in ('train','val'):
            path=args.data/(split+'.npz')
            if file_hash(path)!=prepared['splits'][split]['hash']:raise ValueError('Prepared targets changed')
            data[split]={k:torch.from_numpy(v).cuda() for k,v in dict(np.load(path)).items()}
            side_ids[split]=[torch.where(data[split]['features'][:,i]>0)[0] for i in (768,770)]
            if any(len(ids)==0 for ids in side_ids[split]):raise ValueError('Missing side coverage')
        net=model(512,32).cuda();net.load_state_dict(torch.load(initial,weights_only=True,map_location='cuda'))
        native_initial=native.CheapValue(str(VALUE))
        states=json.loads((ROOT/'benchmarks/search_first_20260911/cpu_eval_after.json').read_text())['states']
        from encoding import fen_to_tensor
        probe=features(np.stack([fen_to_tensor(f,f.split()[1]=='w',h,24,c) for f,h,c in states]))
        def parity(path):
            evaluator=native.CheapValue(str(path));net.eval()
            with torch.no_grad():expected=net(torch.from_numpy(probe).cuda()).ravel().cpu().numpy()
            error=float(np.max(abs(expected-np.array([evaluator.evaluate(*s) for s in states]))))
            if error>1e-5:raise ValueError('Native model export parity failed')
            return error
        parity(VALUE)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.0001)
        atomic_json(args.out/'manifest.json',dict(arm=args.arm,epochs=args.epochs,steps_override=args.steps,
            seed=913327,learning_rate=.0001,weight_decay=.0001,ranking_coefficient=.1 if args.arm=='ranked' else 0,
            initialization=file_hash(initial),initial_binary=file_hash(VALUE),data=file_hash(args.data/'complete.json'),
            replay=label_receipt,runtime=file_hash(ROOT/'native/monster_native.pyd'),implementation=file_hash(__file__),
            recipe='2048 replay +2048 side-balanced new +1024 sibling pairs; identical forward shapes all arms',
            selection='0.5 replay weighted VAL MSE +0.5 side-balanced backed-target VAL MSE; no TEST'))
        def evaluate():
            net.eval();total=mass=0.
            with torch.no_grad():
                for ids in splits['val'].split(4096):
                    e=(net(x[ids]).ravel()-y[ids]).square();total+=float((e*w[ids]).sum());mass+=float(w[ids].sum())
                d=data['val'];prediction=torch.cat([net(chunk).ravel() for chunk in d['features'].split(4096)])
                by_side={name:float((prediction[ids]-d['backed'][ids]).square().mean())
                         for name,ids in zip(('white','black'),side_ids['val'])}
                pair=d['pairs'];a,b=pair[:,:2].long().T
                rank=float(ranking_loss(prediction[a],prediction[b],pair[:,2],pair[:,3]))
                agreement=float((pair[:,2]*(prediction[a]-prediction[b])>0).float().mean())
            replay=total/mass;backed=.5*sum(by_side.values())
            return dict(replay_mse=replay,backed_mse=backed,backed_by_side=by_side,ranking_loss=rank,
                        ranking_agreement=agreement,selection=.5*(replay+backed))
        history=[dict(epoch=0,**evaluate())];best=float('inf');d=data['train']
        for epoch in range(1,args.epochs+1):
            net.train();order=splits['train'][torch.randperm(len(splits['train']),device='cuda')]
            for step,ids in enumerate(order.split(2048)):
                if args.steps is not None and step>=args.steps:break
                selected=torch.cat([ids_side[torch.randint(len(ids_side),(len(ids)//2+(len(ids)%2 if side==0 else 0),),device='cuda')]
                    for side,ids_side in enumerate(side_ids['train'])])
                pair=d['pairs'][torch.randint(len(d['pairs']),(1024,),device='cuda')];a,b=pair[:,:2].long().T
                prediction=net(torch.cat([x[ids],d['features'][selected],d['features'][a],d['features'][b]])).ravel()
                n=len(ids);new_target=d['raw' if args.arm=='raw' else 'backed'][selected]
                replay=((prediction[:n]-y[ids]).square()*w[ids]).sum()/w[ids].sum()
                new=(prediction[n:2*n]-new_target).square().mean()
                rank=ranking_loss(prediction[2*n:2*n+1024],prediction[2*n+1024:],pair[:,2],pair[:,3])
                loss=.5*(replay+new)+(.1 if args.arm=='ranked' else 0.)*rank
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            score=evaluate();history.append(dict(epoch=epoch,seconds=time.monotonic()-started,**score))
            torch.save(net.state_dict(),args.out/f'epoch_{epoch:03}.pt');export(net,args.out/f'epoch_{epoch:03}.bin')
            if score['selection']<best:best=score['selection'];best_epoch=epoch
            atomic_json(args.out/'progress.json',dict(best_epoch=best_epoch,history=history));print(json.dumps(history[-1]),flush=True)
        path=args.out/f'epoch_{best_epoch:03}.bin';net.load_state_dict(torch.load(path.with_suffix('.pt'),weights_only=True))
        error=parity(path);peak=torch.cuda.max_memory_allocated()
        if peak>12*1024**3:raise ValueError('VRAM ceiling exceeded')
        atomic_json(args.out/'complete.json',dict(complete=True,best_epoch=best_epoch,selection=best,
            model_sha256=file_hash(path),parity_max_error=error,cuda_peak_bytes=peak,seconds=time.monotonic()-started,
            note='One validation-nominated checkpoint per arm; mandatory actual play-testing'))

if __name__=='__main__':main()
