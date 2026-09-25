"""Search-backed targets on isolated replay roots and actual lineage-preserved CPU leaves."""
import argparse
import bisect
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'native'),str(ROOT/'src'),str(ROOT/'tools')]
import monster_native as native
from encoding import fen_to_tensor
from evaluation import NNEvaluator
from stateful_generation import restore_state
from search_first_match import prior_keys
from search_leaf_audit import root_map
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease

BASE=ROOT/'benchmarks/search_targets_20260913'
DATA=ROOT/'iterations/b2_001/processed24'
VALUE=ROOT/'models/candidates/search_leaf_leaf_001/epoch_003.bin'
TEACHER=ROOT/'models/candidates/bootstrap_main_gen_0047/arena_selected.pt'

def infer_inputs(tree,nn,records=False):
    length=tree.record_count() if records else tree.frontier_count();values=[]
    with torch.inference_mode():
        for start in range(0,length,512):
            data,signs=tree.input_batch(start,512,nn.input_channels,records)
            p=np.frombuffer(data,dtype='<f4').reshape(-1,8,8,nn.input_channels)
            x=torch.from_numpy(p.transpose(0,3,1,2).copy()).to(nn.device)
            if nn._half:x=x.half()
            v,_=nn.model(x)
            values.extend((v.ravel().float().cpu().numpy()*signs).tolist())
    if len(values)!=length or not np.isfinite(values).all():raise ValueError('Invalid teacher batch')
    return values

def targets(game,tracker,nn,cheap,rng):
    tree=native.LabelTree(game.board.fen(en_passant='fen'),game.white_half_pending,game.turn_count,
                          prior_keys(tracker,game),depth=2,node_limit=250000)
    states=tree.record_states();backed=tree.solve(infer_inputs(tree,nn));raw=infer_inputs(tree,nn,True)
    rows=[]
    for i,(_,fen,pending,count,terminal,path) in enumerate(states):
        if terminal is not None or pending:continue
        rows.append(dict(fen=fen,pending=pending,turn_count=count,path=path,raw=raw[i],backed=backed[i],
                         cheap=cheap.evaluate(fen,pending,count),is_root=i==0))
    roots=[r for r in rows if r['is_root']];children=[r for r in rows if not r['is_root']]
    sign=1 if game.is_white_turn else -1
    if len(children)>8:
        ordered=sorted(range(len(children)),key=lambda i:-sign*children[i]['backed'])
        chosen=list(dict.fromkeys([ordered[0],ordered[1],max(range(len(children)),key=lambda i:sign*children[i]['cheap'])]))
        others=[i for i in range(len(children)) if i not in chosen];rng.shuffle(others)
        chosen=(chosen+others)[:8];children=[children[i] for i in chosen]
    return dict(records=roots+children,sign=sign,nodes=tree.node_count(),frontier=tree.frontier_count(),
                repetition_hits=tree.repetition_hits,cap_hits=tree.cap_hits)

def balanced_ids(positions,ids,count,rng):
    chosen={True:[],False:[]}
    for i in rng.permutation(ids):
        if positions[i,0,0,13]>0:continue
        side=bool(positions[i,0,0,12]>0)
        if len(chosen[side])<count//2:chosen[side].append(int(i))
        if all(len(x)==count//2 for x in chosen.values()):break
    if count%2 or sum(map(len,chosen.values()))!=count:raise ValueError('Need enough roots and an even count')
    return sorted(chosen[True]+chosen[False])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--train-roots',type=int,default=1024)
    ap.add_argument('--val-roots',type=int,default=256);ap.add_argument('--seed',type=int,default=913205)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    audit=json.loads((DATA/'b2_audit.json').read_text());ends,entries=root_map(audit)
    positions=np.load(DATA/'positions.npy',mmap_mode='r');splits=dict(np.load(DATA/'splits.npz'))
    if len(positions)!=ends[-1]:raise ValueError('Raw/processed mapping mismatch')
    rng=np.random.default_rng(args.seed)
    chosen={s:balanced_ids(positions,splits[s],n,rng) for s,n in [('train',args.train_roots),('val',args.val_roots)]}
    hashes={str(p):file_hash(p) for p in [DATA/'positions.npy',DATA/'splits.npz',DATA/'b2_audit.json',
        DATA/'split_game_ids.json',VALUE,TEACHER,ROOT/'native/monster_native.pyd',Path(__file__)]}
    atomic_json(args.out/'manifest.json',dict(hashes=hashes,roots=chosen,seed=args.seed,
        teacher='gen47 raw frontier + exact two-completed-turn minimax; CPU-rule repetition/cap',
        cpu_nodes=50000,leaves_per_root=2,depth=2,tree_limit=250000,complete=False))
    started=time.monotonic();roots=groups=rows=skipped=0;node_sum=frontier_sum=0
    with worker_lease(), (args.out/'groups.jsonl').open('x',encoding='utf-8') as log, \
         (args.out/'roots.jsonl').open('x',encoding='utf-8') as root_log:
        nn=NNEvaluator(str(TEACHER));nn.model.eval();cheap=native.CheapValue(str(VALUE))
        for split,indices in chosen.items():
            last=None
            for index in indices:
                slot=bisect.bisect_right(ends,index);actual,path,offset=entries[slot]
                if actual!=split:raise ValueError('Root split mapping mismatch')
                source=ROOT/'iterations/b2_001/raw'/path
                if source!=last:
                    expected_hash=audit['sources'].get(str(Path(path)),audit['sources'].get(path))
                    if file_hash(source)!=expected_hash:raise ValueError('Raw source changed')
                    records=[json.loads(s) for s in source.read_text().splitlines() if s.strip()];last=source
                record=records[index-offset];g,tracker=restore_state(record['state'])
                expected=fen_to_tensor(record['fen'],g.is_white_turn,g.white_half_pending,24,g.turn_count)
                if not np.array_equal(expected,positions[index]):raise ValueError('Root input differs')
                meta=dict(root=index,split=split,source=path,source_row=index-offset,state=record['state'])
                roots+=1
                if g.is_terminal() or tracker.fired_at is not None:
                    meta['skip']='terminal';root_log.write(json.dumps(meta)+'\n');continue
                r=native.alphabeta_search(g.board.fen(en_passant='fen'),pending=g.white_half_pending,
                    turn_count=g.turn_count,prior_positions=prior_keys(tracker,g),evaluator=cheap,
                    node_limit=50000,max_depth=8,seconds=60,collect_leaves=2,collect_leaf_paths=True,
                    leaf_seed=args.seed+index)
                if r.interrupted and r.nodes<50000:raise TimeoutError('CPU sampler hit clock')
                meta.update(cpu_nodes=r.nodes,cpu_depth=r.completed_depth,leaf_paths=r.leaf_paths,leaves=r.leaf_samples)
                root_log.write(json.dumps(meta)+'\n');root_log.flush()
                for sample,path_moves in enumerate([[]]+r.leaf_paths):
                    if sample:
                        leaf_game,leaf_tracker=restore_state(record['state'])
                        for uci in path_moves:
                            action=next(m for m in leaf_game.get_search_actions() if m.uci()==uci)
                            leaf_game.apply_search_action(action);leaf_tracker.record(leaf_game)
                        fen,pending,count,_=r.leaf_samples[sample-1]
                        if (leaf_game.board.fen(en_passant='fen'),leaf_game.white_half_pending,leaf_game.turn_count)!=(fen,pending,count):
                            raise ValueError('CPU leaf lineage mismatch')
                    else:leaf_game,leaf_tracker=g,tracker
                    if leaf_game.is_terminal() or leaf_tracker.fired_at is not None:raise ValueError('Terminal sampled leaf')
                    t=time.monotonic()
                    try:group=targets(leaf_game,leaf_tracker,nn,cheap,rng)
                    except ValueError as exc:
                        if 'label tree node limit exceeded' not in str(exc):raise
                        skipped+=1;log.write(json.dumps(dict(root=index,split=split,sample=sample,skip=str(exc)))+'\n');continue
                    group.update(root=index,split=split,sample=sample,seconds=time.monotonic()-t)
                    log.write(json.dumps(group)+'\n');log.flush()
                    groups+=1;rows+=len(group['records']);node_sum+=group['nodes'];frontier_sum+=group['frontier']
                if roots%8==0:
                    progress=dict(roots=roots,groups=groups,rows=rows,skipped=skipped,seconds=time.monotonic()-started)
                    atomic_json(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
        peak=torch.cuda.max_memory_allocated()
        if peak>12*1024**3:raise ValueError('VRAM limit exceeded')
        if groups==0 or skipped/max(1,groups+skipped)>.1:raise ValueError('Too many capped/empty label trees')
        if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError('Pinned generation input changed')
        report=dict(complete=True,roots=roots,groups=groups,rows=rows,skipped=skipped,nodes=node_sum,
                    frontier=frontier_sum,seconds=time.monotonic()-started,cuda_peak_bytes=peak,
                    groups_hash=file_hash(args.out/'groups.jsonl'),roots_hash=file_hash(args.out/'roots.jsonl'))
        atomic_json(args.out/'complete.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
