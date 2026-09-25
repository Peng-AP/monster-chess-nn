"""Default parity, numerical drift and full-search speed before scaling games."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'native'),str(ROOT/'src'),str(ROOT/'tools')]
import monster_native as native
from monster_chess import MonsterChessGame
from profile_cpu_search import VALUE,ARMS,record
from profile_search_costs import replay_roots,SOURCE
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease

TOLERANCE=2e-5

def numerical_states(count=10000):
    rng=random.Random(9612);g=MonsterChessGame();states=[]
    for _ in range(count):
        if g.is_terminal() or not g.get_search_actions():g=MonsterChessGame()
        states.append((g.board.fen(en_passant='fen'),g.white_half_pending,g.turn_count))
        g.apply_search_action(rng.choice(g.get_search_actions()))
    # Same boards with every phase/count, rights loss, raw EP changes, promotions,
    # terminal king absence, and distant jumps (not just legal adjacent positions).
    for fen in ('r3k2r/8/8/3pP3/8/8/8/4K3 w kq d6 0 1',
                'r3k2r/8/3P4/8/8/8/8/4K3 w kq - 0 1',
                'r4rk1/8/8/8/8/8/8/4K3 b - - 0 1',
                'k7/3P4/8/8/8/8/8/4K3 w - - 0 1',
                'k2Q4/8/8/8/8/8/8/4K3 w - - 0 1',
                '8/8/8/8/8/8/8/4K3 w - - 0 1'):
        for side,pending in (('w',False),('w',True),('b',False)):
            for turn in (0,1,55,148,149,150,151):
                fields=fen.split();fields[1]=side;states.append((' '.join(fields),pending,turn))
    states+=rng.sample(states,min(1000,len(states)))
    return states

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    base=ROOT/'benchmarks/search_cpu_scaling_20260912'
    snapshot=json.loads((base/'baseline.json').read_text())
    costs=json.loads((base/'costs.json').read_text())
    if not costs['complete'] or costs['source_hash']!=file_hash(SOURCE):raise ValueError('Missing cost profile')
    report=dict(complete=False,runtime=file_hash(ROOT/'native/monster_native.pyd'),value=file_hash(VALUE),
        tolerance=TOLERANCE,profile_source_hash=file_hash(SOURCE),fixed_depth=[],timed=[])
    arms={'baseline':{},'optimized':ARMS['large_tt'],
          'incremental':{**ARMS['large_tt'],'incremental_eval':True},'incremental_plain':{'incremental_eval':True}}
    with worker_lease():
        net=native.CheapValue(str(VALUE))
        if snapshot['value']!=report['value']:raise ValueError('Baseline model changed')
        fields=('action','value','completed_depth','nodes','cutoffs','interrupted','eval_cache_hits','tt_hits','tt_cutoffs')
        for (f,h,c),old in zip(snapshot['states'],snapshot['fixed_node']):
            r=native.alphabeta_search(f,pending=h,turn_count=c,evaluator=net,seconds=30,node_limit=30000,max_depth=6)
            if any(getattr(r,k)!=old[k] for k in fields):raise ValueError('Default search changed')
        report['default_parity']=True
        states=numerical_states();direct,_,_=net.evaluate_sequence(states,False)
        incremental,updates,refreshes=net.evaluate_sequence(states,True)
        error=np.abs(np.array(direct)-incremental)
        report['numerical']=dict(states=len(states),max_error=float(error.max()),mean_error=float(error.mean()),
                                 updates=updates,refreshes=refreshes)
        if error.max()>TOLERANCE or updates==0:raise ValueError('Incremental evaluator drift/parity failure')
        roots=replay_roots();report['roots']=roots
        for i,root in enumerate(roots):
            common={k:root[k] for k in ('fen','pending','turn_count','prior_positions')}
            probe=native.alphabeta_search(**common,evaluator=net,seconds=.15,node_limit=100000000)
            depth=min(8,max(3,probe.completed_depth+1));trials={a:[] for a in arms}
            for repeat in range(3):
                order=list(arms);shift=(i+repeat)%len(order);order=order[shift:]+order[:shift]
                for arm in order:
                    started=time.perf_counter()
                    r=native.alphabeta_search(**common,evaluator=net,seconds=60,max_depth=depth,
                        node_limit=100000000,**arms[arm])
                    if r.interrupted:raise ValueError('Fixed-depth speed trial interrupted')
                    trials[arm].append(dict(**record(r,time.perf_counter()-started),
                        incremental_updates=r.incremental_updates,refreshes=r.eval_refreshes))
            value=trials['baseline'][0]['value']
            maximum=max(abs(r['value']-value) for rows in trials.values() for r in rows)
            if maximum>TOLERANCE:raise ValueError('Fixed-depth value drift')
            report['fixed_depth'].append(dict(state=i,depth=depth,arms=trials,max_value_error=maximum,
                action_changes={a:trials[a][0]['action']!=trials['baseline'][0]['action'] for a in arms}))
            atomic_json(args.out.with_suffix('.partial.json'),report)
            print('validation root',i,'depth',depth,flush=True)
        totals={a:sum(float(np.median([r['seconds'] for r in row['arms'][a]])) for row in report['fixed_depth']) for a in arms}
        nominee=min(('incremental','incremental_plain'),key=totals.get)
        incremental_speedup=min(totals['baseline'],totals['optimized'])/totals[nominee]
        nominated=incremental_speedup>=1.05
        for i in range(0,len(roots),3):
            common={k:roots[i][k] for k in ('fen','pending','turn_count','prior_positions')}
            for arm in ('optimized',nominee):
                for budget in (2.,8.):
                    t=time.perf_counter()
                    r=native.alphabeta_search(**common,evaluator=net,seconds=budget,node_limit=100000000,
                                            **arms[arm])
                    report['timed'].append(dict(state=i,arm=arm,budget=budget,**record(r,time.perf_counter()-t)))
                    if r.interrupted and r.nodes>=100000000:raise ValueError('100M node budget insufficient')
        selected={a:arms[a] for a in ('baseline','optimized')}
        if nominated:selected[nominee]=arms[nominee]
        report.update(complete=True,arms=selected,seconds_by_arm=totals,incremental_nominee=nominee,
            incremental_speedup=incremental_speedup,incremental_nominated=nominated,
            note='Fixed-depth speed and tolerance parity qualify play-testing, not promotion')
        atomic_json(args.out,report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('roots','fixed_depth','timed')},indent=2))

if __name__=='__main__':main()
