"""Representative depth/time profile and exact White continuation census."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'native')]
import monster_native as native
from monster_chess import MonsterChessGame
from match import load_book
from profile_cpu_search import ARMS,VALUE,record,states
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease
from cpu_search_engine import CpuSearchEngine


def identity(game):
    return (' '.join(game.board.fen(en_passant='fen').split()[:4]),
            game.white_half_pending,game.turn_count)

def pair_census(game):
    if not game.is_white_turn or game.white_half_pending:return None
    finals=[];firsts=game.get_search_actions()
    for first in firsts:
        after=game.clone();after.apply_search_action(first)
        if after.is_terminal():finals.append(identity(after));continue
        for second in after.get_search_actions():
            final=after.clone();final.apply_search_action(second);finals.append(identity(final))
    return dict(first_moves=len(firsts),continuations=len(finals),unique_final_states=len(set(finals)),
                redundant=len(finals)-len(set(finals)))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    book,_=load_book('benchmarks/b2_challenger_confirmation_20260910/confirmation_book.json')
    inputs=[[r['fen'],bool(r['half']),r['turn_count']] for r in book[288:300]]+states()[::3]
    initial=MonsterChessGame()
    for _ in range(3):
        inputs.append([initial.board.fen(en_passant='fen'),initial.white_half_pending,initial.turn_count])
        initial.apply_search_action(initial.get_search_actions()[0])
    report=dict(runtime=file_hash(ROOT/'native/monster_native.pyd'),value=file_hash(VALUE),
                states=inputs,fixed_depth=[],timed=[],gpu_ordering=[],pair_census=[],complete=False)
    with worker_lease():
        net=native.CheapValue(str(VALUE))
        for index,(f,h,c) in enumerate(inputs):
            common=dict(pending=h,turn_count=c,evaluator=net)
            probe=native.alphabeta_search(f,seconds=.15,**common)
            depth=min(8,max(3,probe.completed_depth+1))
            trials={a:[] for a in ARMS}
            for repeat in range(3):
                arms=list(ARMS);shift=(index+repeat)%len(arms);arms=arms[shift:]+arms[:shift]
                for arm in arms:
                    t=time.perf_counter()
                    r=native.alphabeta_search(f,seconds=30,max_depth=depth,**common,**ARMS[arm])
                    trials[arm].append(record(r,time.perf_counter()-t))
            if any(r['interrupted'] for rows in trials.values() for r in rows):raise ValueError('Interrupted fixed-depth trial')
            if len({r['value'] for rows in trials.values() for r in rows})!=1:raise ValueError('Fixed-depth value mismatch')
            report['fixed_depth'].append(dict(state=index,depth=depth,arms=trials))
            g=MonsterChessGame(f);g.white_half_pending=h;g.turn_count=c
            census=pair_census(g)
            if census:report['pair_census'].append(dict(state=index,**census))
            print('profile state',index,'depth',depth,flush=True)
        totals={a:sum(float(np.median([r['seconds'] for r in row['arms'][a]]))
                      for row in report['fixed_depth']) for a in ARMS}
        nominee=min(totals,key=totals.get)
        if totals['baseline']/totals[nominee]<1.05:nominee='baseline'
        report.update(seconds_by_arm=totals,nominee=nominee,options=ARMS[nominee],
                      speedup=totals['baseline']/totals[nominee])
        for i in [0,2,4,6,8,10,len(inputs)-3,len(inputs)-2,len(inputs)-1]:
            f,h,c=inputs[i]
            for seconds in (.3,2.):
                for arm in dict.fromkeys(['baseline',nominee]):
                    t=time.perf_counter()
                    r=native.alphabeta_search(f,pending=h,turn_count=c,evaluator=net,
                        seconds=seconds,**ARMS[arm])
                    report['timed'].append(dict(state=i,arm=arm,budget=seconds,**record(r,time.perf_counter()-t)))
        from evaluation import NNEvaluator
        nn=NNEvaluator('models/candidates/bootstrap_main_gen_0047/arena_selected.pt')
        engine=CpuSearchEngine(VALUE,ARMS[nominee],nn)
        nn.batch_policies([MonsterChessGame()])
        for i,(f,h,c) in enumerate(inputs[:6]):
            g=MonsterChessGame(f);g.white_half_pending=h;g.turn_count=c
            depth=report['fixed_depth'][i]['depth']
            action,detail=engine.choose(g,[],30,max_depth=depth)
            baseline=report['fixed_depth'][i]['arms']['baseline'][0]
            if detail['interrupted'] or detail['value']!=baseline['value']:raise ValueError('GPU hint changed exact value')
            report['gpu_ordering'].append(dict(state=i,action=action,**detail))
        peak=nn.torch.cuda.max_memory_allocated() if nn.device.type=='cuda' else 0
        if peak>12*1024**3:raise ValueError('VRAM ceiling exceeded')
        report.update(cuda_peak_bytes=peak,complete=True)
        atomic_json(args.out,report)
        print(json.dumps({k:v for k,v in report.items() if k in ('nominee','options','seconds_by_arm','speedup','complete')},indent=2))

if __name__=='__main__':main()
