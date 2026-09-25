"""Disjoint native search costs on replayed game roots with original history."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'native'),str(ROOT/'src'),str(ROOT/'tools')]
import monster_native as native
from monster_chess import MonsterChessGame
from repetition import RepetitionTracker
from search_first_match import prior_keys
from profile_cpu_search import VALUE,ARMS,record
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease

SOURCE=ROOT/'benchmarks/search_cpu_gpu_20260912/campaign/confirmation_baseline/games.jsonl'

def replay_roots():
    groups={i:[] for i in range(3)}
    for game_index,line in enumerate(SOURCE.read_text().splitlines()[:12]):
        row=json.loads(line);s=row['start'];g=MonsterChessGame(s['fen'])
        g.white_half_pending=bool(s['half']);g.turn_count=s['turn_count']
        tracker=RepetitionTracker(enabled=True,threshold=3);tracker.record(g)
        target=int(len(row['decisions'])*(.2,.5,.8)[game_index%3])
        chosen=set()
        for ply,d in enumerate(row['decisions']):
            phase=2 if not g.is_white_turn else int(g.white_half_pending)
            if ply>=target and phase not in chosen and len(groups[phase])<6:
                groups[phase].append(dict(fen=g.board.fen(en_passant='fen'),pending=g.white_half_pending,
                    turn_count=g.turn_count,prior_positions=prior_keys(tracker,g),game=game_index,ply=ply))
                chosen.add(phase)
            action=next((m for m in g.get_search_actions() if m.uci()==d['action']),None)
            if action is None:raise ValueError('Replay illegal move')
            g.apply_search_action(action);tracker.record(g,ply+1)
        if g.fen()!=row['final_fen']:raise ValueError('Replay final state mismatch')
    if any(len(rows)!=6 for rows in groups.values()):raise ValueError('Insufficient phase coverage')
    return [r for rows in groups.values() for r in rows]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    roots=replay_roots();report=dict(complete=False,source=str(SOURCE),source_hash=file_hash(SOURCE),
        runtime=file_hash(ROOT/'native/monster_native.pyd'),value=file_hash(VALUE),roots=roots,trials=[])
    with worker_lease():
        net=native.CheapValue(str(VALUE))
        for i,root in enumerate(roots):
            state={k:root[k] for k in ('fen','pending','turn_count','prior_positions')}
            for arm in ('baseline','large_tt'):
                trials={}
                for enabled in ([False,True] if i%2==0 else [True,False]):
                    t=time.perf_counter()
                    r=native.alphabeta_search(**state,evaluator=net,seconds=60,max_depth=12,
                        node_limit=250000,profile_search=enabled,**ARMS[arm])
                    if r.interrupted and r.nodes<250000:raise ValueError('Fixed-node trial hit clock')
                    trials[str(enabled)]=dict(**record(r,time.perf_counter()-t),
                        costs=dict(zip(r.profile_labels,r.profile_seconds)),
                        calls=dict(zip(r.profile_labels,r.profile_calls)))
                fields=('action','value','completed_depth','nodes','cutoffs','eval_cache_hits','tt_hits')
                if any(trials['False'][k]!=trials['True'][k] for k in fields):raise ValueError('Instrumentation changed search')
                report['trials'].append(dict(state=i,arm=arm,trials=trials))
            print('cost profile root',i,flush=True)
            atomic_json(args.out.with_suffix('.partial.json'),report)
    costs={};total=plain=0
    for row in report['trials']:
        enabled=row['trials']['True'];total+=enabled['seconds'];plain+=row['trials']['False']['seconds']
        for k,v in enabled['costs'].items():costs[k]=costs.get(k,0)+v
    report.update(complete=True,cost_seconds=costs,cost_fraction={k:v/total for k,v in costs.items()},
        timed_seconds=total,untimed_seconds=plain,instrumentation_slowdown=total/plain,
        residual_fraction=1-sum(costs.values())/total)
    atomic_json(args.out,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('roots','trials')},indent=2))

if __name__=='__main__':main()
