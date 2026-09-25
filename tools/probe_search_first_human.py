"""Known human-line diagnostic: complete root turns, not a strength verdict.

Reconstruct actual settled history using the existing audited human-log helper.
Keep these positions out of training. Both engines start with fresh search trees.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for folder in ('src','native','tools'):
    sys.path.insert(0,str(ROOT/folder))
import monster_native as native
from evaluation import NNEvaluator
from native_mcts import NativeMCTS
from probe_human_line import reconstruct,force,state_record
from search_first_match import prior_keys
from match_evidence import atomic_json,file_hash,read_rows
from worker_lease import worker_lease


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--value',default='models/candidates/search_first_distilled_w512_001/epoch_007.bin')
    ap.add_argument('--reference',default='models/candidates/bootstrap_main_gen_0047/arena_selected.pt')
    ap.add_argument('--human',default='data/raw/human_games/black_2026_07/game_00031.jsonl')
    ap.add_argument('--cases',nargs='+',default=['7','14','14:c4b3','14:c4d3','14:f2f3','15','16'])
    ap.add_argument('--seconds',type=float,nargs='+',default=[.3,2.])
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if min(args.seconds)<=0:raise ValueError('positive clocks required')
    rows=read_rows(args.human)
    report=dict(complete=False,probes=[],manifest=dict(
        hashes={str(p):file_hash(p) for p in [args.value,args.reference,args.human,
            ROOT/'native/monster_native.pyd',Path(__file__),ROOT/'tools/probe_human_line.py',
            ROOT/'tools/search_first_match.py']},
        cases=args.cases,seconds=args.seconds,
        note='known diagnostic, no ground-truth assignment; fresh trees; soft half-move clocks'))
    with worker_lease():
        net=native.CheapValue(args.value)
        engine=NativeMCTS(10_000_000,NNEvaluator(args.reference),root_noise=False,
                          allow_early_stop=False,reuse_across_moves=True,seed=9173)
        engine.get_best_action(reconstruct(rows,0)[0],temperature=0,seconds=.05)
        for case in args.cases:
            index,_,moves=case.partition(':')
            for seconds in args.seconds:
                for label in ('alphabeta','gen47'):
                    g,tracker,history=reconstruct(rows,int(index))
                    force(g,[m for m in moves.split(',') if m],tracker)
                    engine._reuse_tree=engine._reuse_key=None
                    engine._decisions=0
                    root_color=g.is_white_turn
                    probe=dict(case=case,seconds=seconds,engine=label,root=state_record(g),
                        reconstructed_prefix=history,decisions=[])
                    while g.is_white_turn==root_color and not g.is_terminal() and tracker.fired_at is None:
                        before=state_record(g)
                        started=time.monotonic()
                        if label=='alphabeta':
                            r=native.alphabeta_search(g.board.fen(en_passant='fen'),
                                pending=g.white_half_pending,turn_count=g.turn_count,seconds=seconds,
                                evaluator=net,prior_positions=prior_keys(tracker,g))
                            action=next((a for a in g.get_search_actions() if a.uci()==r.action),None)
                            details=dict(value_white=r.value,depth=r.completed_depth,nodes=r.nodes)
                        else:
                            action,_,value=engine.get_best_action(g,temperature=0,seconds=seconds)
                            details=dict(value_white=float(value)*(1 if g.is_white_turn else -1),
                                         timing=engine.last_search_timing)
                        if action is None:raise ValueError('No action in live diagnostic')
                        probe['decisions'].append(dict(before=before,action=action.uci(),
                            elapsed=time.monotonic()-started,**details))
                        g.apply_search_action(action)
                        tracker.record(g)
                    probe['after']=state_record(g)
                    report['probes'].append(probe)
                    atomic_json(args.out,report)
                    print(case,seconds,label,[d['action'] for d in probe['decisions']],flush=True)
    report['complete']=True
    atomic_json(args.out,report)


if __name__=='__main__':main()
