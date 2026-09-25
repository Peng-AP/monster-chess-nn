"""Outcome data and frozen-backbone value-head fits; opt-in research only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from match_evidence import atomic_json,digest,file_hash
import mainline_study as study
from encoding import fen_to_tensor
from train import load_model_for_inference


def phase(state):
    return 'black' if state['fen'].split()[1]=='b' else ('white_second' if state['half'] else 'white_first')


PHASES=('black','white_first','white_second')


def encoded(state):
    return fen_to_tensor(state['fen'],state['fen'].split()[1]=='w',state['half'],15)


def fingerprint(x):
    return hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest()


def outcome_target(white_result,state):
    if white_result not in (-1.,0.,1.):raise ValueError('Not a completed captures-only outcome')
    return white_result if state['fen'].split()[1]=='w' else -white_result


def load_games(config_path,output):
    config=study.read(config_path)
    rows=[]
    for task in config['tasks']:
        result=study.load_result(Path(output)/'tasks'/(digest(task)+'.json'),task)
        rows.append((task,result))
    return rows


def raw_values(model,states):
    import torch
    values=[]
    with torch.no_grad():
        for start in range(0,len(states),128):
            x=np.stack([encoded(s) for s in states[start:start+128]])
            t=torch.from_numpy(x.transpose(0,3,1,2)).cuda()
            v,_=model(t)
            if not torch.isfinite(v).all():raise ValueError('Nonfinite raw value')
            values.extend(v.flatten().cpu().tolist())
    return values


def select(config,out):
    import torch
    torch.set_num_threads(1)
    games=load_games(config['parents_config'],config['parents_output'])
    rng=np.random.default_rng(26017)
    order=rng.permutation(len(games)); splits={}
    ntrain,nval=(8,2) if config['smoke'] else (144,24)
    for rank,index in enumerate(order):splits[int(index)]='train' if rank<ntrain else 'val' if rank<ntrain+nval else 'test'
    models=[load_model_for_inference(p,'cuda')[0] for p in (config['reference'],config['initial'])]
    roots=[]
    for index,(task,result) in enumerate(games):
        desired=('black','black','white_first','white_second')[index%4]
        trace=result['trajectory']; choices=[]
        for i,s in enumerate(trace[:-1]):
            if not 4<=s['absolute_ply']<=120 or phase(s)!=desired:continue
            decision=trace[i+1]
            if decision['finisher'] or decision['root_value'] is None:continue
            choices.append((i,s,decision['root_value']))
        if not choices:raise ValueError(f'Missing phase {desired} in parent {index}')
        values=[raw_values(m,[s for _,s,_ in choices]) for m in models]
        scored=[dict(index=i,score=abs(a-b)+abs(b-q),reference_value=a,initial_value=b,search_value=q)
                for (i,_,q),a,b in zip(choices,*values)]
        chosen=max(scored,key=lambda r:(r['score'],-r['index']))
        i=chosen['index']; s=trace[i]
        moves=task['state']['moves']+[r['action'] for r in trace[1:i+1]]
        state=dict(initial_fen=task['state']['initial_fen'],moves=moves,fen=s['fen'],
                   half=bool(s['half']),turn_count=s['turn_count'])
        study.checked_restore(state)
        roots.append(dict(id=f'family_{index:04d}',split=splits[index],phase=desired,state=state,
                          parent_task=digest(task),selection=chosen,candidates=scored))
    atomic_json(out,dict(complete=True,roots=roots,parents_config_sha256=file_hash(config['parents_config']),
                         parents_summary_sha256=file_hash(Path(config['parents_output'])/'summary.json')))
    print(f'SELECTED {len(roots)} disagreement roots',flush=True)


def clean_splits(groups):
    # Higher-priority held-out sets own exact encoded inputs. Merge duplicates
    # only within their surviving split, averaging observed outcome targets.
    occupied=set(); result={}; census={}
    for name in ('new_test','new_val','old_val','new_train','old_train'):
        grouped={}; excluded=0
        for row in groups[name]:
            key=row['key']
            if key in occupied:excluded+=1;continue
            grouped.setdefault(key,[]).append(row)
        merged=[]
        for key,rows in grouped.items():
            r=dict(rows[0]);r['y']=float(np.mean([v['y'] for v in rows]));r['families']=sorted({v['family'] for v in rows})
            merged.append(r)
        # Train sources may share inputs: they are both training, not leakage.
        if not name.endswith('train'):occupied.update(grouped)
        result[name]=merged
        census[name]=dict(input_rows=len(groups[name]),excluded=excluded,unique_rows=len(merged),
                          conflicting_outcomes=sum(len({r['y'] for r in rs})>1 for rs in grouped.values()))
    return result,census


def prepare(config,out):
    import torch
    torch.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    roots=study.read(config['roots'])['roots'];lookup={r['id']:r for r in roots}
    parent_lookup={r['parent_task']:r for r in roots}
    groups={n:[] for n in ('new_train','new_val','new_test','old_train','old_val')}
    for key in ('parents','continuations'):
        for task,result in load_games(config[key+'_config'],config[key+'_output']):
            root=parent_lookup[digest(task)] if key=='parents' else lookup[task['case_id']]
            trace=result['trajectory'][:-1]
            for p in PHASES:
                positions=[s for s in trace if phase(s)==p]
                indices=np.linspace(0,len(positions)-1,min(8,len(positions)),dtype=int) if positions else []
                for index in indices:
                    s=positions[index]; x=encoded(s)
                    groups['new_'+root['split']].append(dict(x=x,key=fingerprint(x),y=outcome_target(result['result_white'],s),
                                                           phase=p,family=root['id']))
    replay=Path(config['replay']);x=np.load(replay/'positions.npy',mmap_mode='r')
    for name,h in config['replay_hashes'].items():
        if file_hash(replay/name)!=h:raise ValueError('Replay source hash changed')
    y=np.load(replay/'capture_results.npy',mmap_mode='r');w=np.load(replay/'value_weights.npy',mmap_mode='r')
    split=dict(np.load(replay/'splits.npz'));rng=np.random.default_rng(26017)
    for name,count in [('train',512 if config['smoke'] else 32768),('val',128 if config['smoke'] else 4096)]:
        pool=split[name];pool=pool[w[pool]>0];ids=rng.choice(pool,min(count,len(pool)),replace=False)
        for i in ids:
            arr=np.asarray(x[i]);white=arr[0,0,12]>0;p='black' if not white else 'white_second' if arr[0,0,13]>0 else 'white_first'
            groups['old_'+name].append(dict(x=arr,key=fingerprint(arr),y=float(y[i])*(1 if white else -1),phase=p,family=f'replay_{int(i)}'))
    groups,census=clean_splits(groups)
    net,_=load_model_for_inference(config['initial'],'cuda');net.eval()
    if net.spatial_value_head or net.use_wdl_head:raise ValueError('Expected original GAP scalar value head')
    report=dict(complete=False,splits={},exclusions=census,initial=file_hash(config['initial']),
                config_sha256=digest(config),family_splits={r['id']:r['split'] for r in roots})
    for name,rows in groups.items():
        if not rows or any(not any(r['phase']==p for r in rows) for p in PHASES):
            raise ValueError(f'Missing coverage after split cleaning: {name}')
        feats=[];anchor=[]
        with torch.no_grad():
            for start in range(0,len(rows),128):
                arr=np.stack([r['x'] for r in rows[start:start+128]])
                t=torch.from_numpy(arr.transpose(0,3,1,2).copy()).cuda()
                backbone,_=net._forward_backbone(t);pooled=backbone.mean(dim=(2,3))
                actual,_=net(t);cached=net.value_head[2:](pooled)
                if not torch.allclose(actual,cached,atol=1e-6,rtol=1e-6):raise ValueError('Feature cache parity failed')
                feats.append(pooled.cpu().numpy());anchor.append(actual.flatten().cpu().numpy())
        path=out/(name+'.npz')
        # Data artifact publication; temporary file followed by same-directory replace.
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez(f,features=np.concatenate(feats),anchor=np.concatenate(anchor),
            target=np.array([r['y'] for r in rows],np.float32),phase=np.array([PHASES.index(r['phase']) for r in rows]),
            keys=np.array([r['key'] for r in rows]))
        tmp.replace(path)
        report['splits'][name]=dict(rows=len(rows),sha256=file_hash(path),
                                   phases={p:sum(r['phase']==p for r in rows) for p in PHASES})
        atomic_json(out/(name+'_families.json'),[r['families'] for r in rows])
        report['splits'][name]['families_sha256']=file_hash(out/(name+'_families.json'))
    for name,h in config['replay_hashes'].items():
        if file_hash(replay/name)!=h:raise ValueError('Replay changed during preparation')
    atomic_json(out/'complete.json',dict(report,complete=True))
    print('VALUE DATA COMPLETE '+json.dumps(report['exclusions']),flush=True)


def frozen_equal(initial,after):
    import torch
    return all(torch.equal(v,after[k]) for k,v in initial.items() if not k.startswith('value_head.'))


def train(config,out):
    import torch
    torch.set_num_threads(1);torch.manual_seed(26017)
    torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    data=Path(config['data']);receipt=study.read(data/'complete.json')
    if not receipt['complete']:raise ValueError('Data incomplete')
    pools={}
    for name,meta in receipt['splits'].items():
        p=data/(name+'.npz')
        if file_hash(p)!=meta['sha256']:raise ValueError('Changed cached features')
        a=dict(np.load(p));pools[name]={k:torch.from_numpy(a[k]).cuda() for k in ('features','target','anchor','phase')}
    net,_=load_model_for_inference(config['initial'],'cuda');net.eval()
    original={k:v.detach().clone() for k,v in net.state_dict().items()}
    for p in net.parameters():p.requires_grad_(False)
    head=copy.deepcopy(net.value_head[2:])
    for p in head.parameters():p.requires_grad_(True)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(26017)
    phase_ids={n:[torch.where(d['phase']==i)[0] for i in range(3)] for n,d in pools.items()}
    def evaluate(name):
        d=pools[name]
        with torch.no_grad():pred=head(d['features']).flatten();err=(pred-d['target']).square()
        return dict(mse=float(err.mean()),by_phase={p:float(err[ids].mean()) for p,ids in zip(PHASES,phase_ids[name])})
    def sample(name,n):
        d=pools[name];indices=[]
        for pool,count in zip(phase_ids[name],(n//2,n//4,n//4)):
            indices.append(pool[torch.tensor(rng.integers(0,len(pool),size=count),device='cuda')])
        ids=torch.cat(indices);return [d[k][ids] for k in ('features','target','anchor')]
    epochs=1 if config['smoke'] else 12;steps=2 if config['smoke'] else 128
    history=[];best=None;best_score=float('inf')
    for epoch in range(1,epochs+1):
        for _ in range(steps):
            if config['arm']=='continuation':
                left,right=sample('old_train',256),sample('new_train',256)
                features,target,anchor=[torch.cat([a,b]) for a,b in zip(left,right)]
            else:features,target,anchor=sample('old_train',512)
            pred=head(features).flatten()
            loss=(pred-target).square().mean()+.1*(pred-anchor).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
        metrics={n:evaluate(n) for n in ('old_val','new_val')}
        score=float(np.mean([v for d in metrics.values() for v in d['by_phase'].values()]))
        history.append(dict(epoch=epoch,selection_score=score,validation=metrics))
        if score<best_score:best_score=score;best=copy.deepcopy(head.state_dict());selected=epoch
        print(f'VALUE {config["arm"]} epoch{epoch} validation={score:.6f}',flush=True)
    head.load_state_dict(best);net.value_head[2:].load_state_dict(best)
    if not frozen_equal(original,net.state_dict()):raise ValueError('Frozen backbone/policy/buffers changed')
    # Also verify policy output bitwise against the original checkpoint.
    initial,_=load_model_for_inference(config['initial'],'cuda')
    x=torch.zeros((4,15,8,8),device='cuda');x[:,12]=torch.tensor([1,-1,1,-1],device='cuda')[:,None,None]
    with torch.no_grad():
        _,before=initial(x);_,after=net(x)
    if not torch.equal(before,after):raise ValueError('Policy logits changed')
    path=out/'candidate.pt';tmp=out/'candidate.tmp'
    torch.save({k:v.cpu() for k,v in net.state_dict().items()},tmp);tmp.replace(path)
    atomic_json(out/'complete.json',dict(complete=True,arm=config['arm'],selected_epoch=selected,
        history=history,test=evaluate('new_test'),model_sha256=file_hash(path),
        initial_sha256=file_hash(config['initial']),data_sha256=file_hash(data/'complete.json'),
        frozen_parameters_and_buffers_equal=True,policy_logits_equal=True,config=config))
    print(f'VALUE FIT COMPLETE {config["arm"]}',flush=True)


if __name__=='__main__':
    from worker_lease import worker_lease
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','prepare','train'])
    p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    with worker_lease():globals()[a.stage](study.read(a.config),a.out)
