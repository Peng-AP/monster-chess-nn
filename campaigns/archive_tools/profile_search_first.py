"""Fixed-state correctness/performance and incumbent clock-overhead breakdown."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for folder in ('src','native'):
    sys.path.insert(0,str(ROOT/folder))
import chess
import monster_native as native
from monster_chess import MonsterChessGame
from evaluation import NNEvaluator
from native_mcts import NativeMCTS
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--value',default='models/candidates/search_first_distilled_001/epoch_022.bin')
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    log=ROOT/'benchmarks/search_first_20260911/exact_frontier_300ms/games.jsonl'
    games=[json.loads(x) for x in log.read_text().splitlines()]
    states=[]
    for row in games[:4]:
        g=MonsterChessGame(row['start']['fen'])
        g.white_half_pending=row['start']['half'];g.turn_count=row['start']['turn_count']
        wanted={0,len(row['decisions'])//2,max(0,len(row['decisions'])-3)}
        for i,d in enumerate(row['decisions']):
            if i in wanted:states.append(g.clone())
            g.apply_search_action(chess.Move.from_uci(d['action']))
    report=dict(source=file_hash(log),runtime=file_hash(ROOT/'native/monster_native.pyd'),
                value=file_hash(args.value),fixed_depth=[],timed_puct=[])
    with worker_lease():
        evaluator=native.CheapValue(args.value)
        for index,g in enumerate(states):
            results={}
            for optimized in (False,True):
                start=time.perf_counter()
                r=native.alphabeta_search(g.board.fen(en_passant='fen'),pending=g.white_half_pending,
                    turn_count=g.turn_count,max_depth=3,seconds=30,evaluator=evaluator,optimizations=optimized)
                results[str(optimized)]=dict(value=r.value,action=r.action,nodes=r.nodes,
                    depth=r.completed_depth,seconds=time.perf_counter()-start,interrupted=r.interrupted,
                    eval_hits=r.eval_cache_hits,tt_hits=r.tt_hits)
            if any(v['interrupted'] for v in results.values()) or results['False']['value']!=results['True']['value']:
                raise AssertionError(f'Fixed-depth mismatch at state{index}: {results}')
            report['fixed_depth'].append(dict(fen=g.fen(),pending=g.white_half_pending,
                                             turn_count=g.turn_count,results=results))
        print('fixed-depth optimized/uncached parity passed',flush=True)
        nn=NNEvaluator('models/candidates/bootstrap_main_gen_0047/arena_selected.pt')
        for reuse in (True,False):
            engine=NativeMCTS(10_000_000,nn,root_noise=False,allow_early_stop=False,
                              reuse_across_moves=reuse,seed=9173)
            engine.get_best_action(MonsterChessGame(),temperature=0,seconds=.05)
            for index,g in enumerate(states[:3]):
                for seconds in (.3,2.):
                    start=time.perf_counter()
                    action,_,value=engine.get_best_action(g,temperature=0,seconds=seconds)
                    record=dict(state=index,reuse_across_moves=reuse,budget=seconds,
                        elapsed=time.perf_counter()-start,action=action.uci(),value=value,
                        timing=engine.last_search_timing)
                    report['timed_puct'].append(record)
                    print(json.dumps(record),flush=True)
    report['complete']=True
    atomic_json(args.out,report)


if __name__=='__main__':main()
