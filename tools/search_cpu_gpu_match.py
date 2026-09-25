"""Timed CPU/GPU cooperation match; reusable adapters, explicit backend receipts."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'native')]
from cpu_search_engine import CpuSearchEngine
from monster_chess import MonsterChessGame
from native_mcts import NativeMCTS
from evaluation import NNEvaluator
from repetition import RepetitionTracker
from search_first_match import prior_keys
from match import load_book
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease

VALUE='models/candidates/search_leaf_leaf_001/epoch_003.bin'
REFERENCE='models/candidates/bootstrap_main_gen_0047/arena_selected.pt'

class Player:
    def __init__(self,mode,value,options,nn,seed):
        self.mode=mode
        self.cpu=CpuSearchEngine(value,{} if mode=='baseline' else options,
                                  nn if mode=='guided' else None) if mode!='puct' else None
        self.puct=NativeMCTS(10000000,nn,batch_size=16,root_noise=False,
            allow_early_stop=False,reuse_across_moves=True,seed=seed) if mode in ('puct','split') else None
    def warmup(self):
        if self.puct:self.puct.get_best_action(MonsterChessGame(),temperature=0,seconds=.05)
        if self.mode=='guided':self.cpu.policy_evaluator.batch_policies([MonsterChessGame()])
    def reset(self):
        if self.puct:
            self.puct._reuse_tree=self.puct._reuse_key=None;self.puct._decisions=0
    def choose(self,game,prior,seconds,node_limit=10000000):
        if self.mode=='puct' or (self.mode=='split' and game.is_white_turn):
            action,_,value=self.puct.get_best_action(game,temperature=0,seconds=seconds)
            return None if action is None else action.uci(),dict(alphabeta=False,value=value,
                timing=self.puct.last_search_timing,backend='gpu_puct')
        action,detail=self.cpu.choose(game,prior,seconds,node_limit=node_limit)
        return action,{**detail,'backend':'gpu_guided_cpu' if self.mode=='guided' else 'cpu'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['baseline','cpu','guided','split','puct'],default='cpu')
    ap.add_argument('--opponent',choices=['baseline','puct'],default='puct')
    ap.add_argument('--options',type=Path,required=True,help='Successful workload profile receipt')
    ap.add_argument('--value',default=VALUE);ap.add_argument('--reference',default=REFERENCE)
    ap.add_argument('--book',default='benchmarks/b2_challenger_confirmation_20260910/confirmation_book.json')
    ap.add_argument('--pairs',type=int,default=16);ap.add_argument('--offset',type=int,default=288)
    ap.add_argument('--seconds',type=float,default=.3);ap.add_argument('--seed',type=int,default=9373)
    ap.add_argument('--opponent-seconds',type=float,help='Defaults to candidate clock')
    ap.add_argument('--node-limit',type=int,default=10000000,help='CPU candidate node ceiling')
    ap.add_argument('--opponent-node-limit',type=int,default=10000000,help='CPU opponent node ceiling')
    ap.add_argument('--selfplay',action='store_true')
    args=ap.parse_args()
    if args.opponent_seconds is None:args.opponent_seconds=args.seconds
    if (args.pairs<1 or args.offset<0 or any(not math.isfinite(t) or t<=0 for t in
        (args.seconds,args.opponent_seconds)) or min(args.node_limit,args.opponent_node_limit)<1):
        raise ValueError('Positive finite match limits required')
    if args.selfplay and args.opponent_seconds!=args.seconds:raise ValueError('Selfplay requires one clock')
    profile=json.loads(args.options.read_text())
    if not profile['complete'] or profile['runtime']!=file_hash(ROOT/'native/monster_native.pyd'):
        raise ValueError('Incomplete/stale required profile')
    if profile['value']!=file_hash(args.value):raise ValueError('Profile model differs')
    entries,_=load_book(args.book);entries=entries[args.offset:args.offset+args.pairs]
    if len(entries)!=args.pairs:raise ValueError('Insufficient openings')
    args.out.mkdir(parents=True,exist_ok=False)
    sources=[ROOT/'native/monster_native.pyd',ROOT/'src/cpu_search_engine.py',ROOT/'src/native_mcts.py',
             ROOT/'src/evaluation.py',Path(__file__),Path(args.value),Path(args.reference),Path(args.book),args.options]
    sources+=list((ROOT/'native/src').glob('*.rs'))
    atomic_json(args.out/'manifest.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        hashes={**{str(p):file_hash(p) for p in sources},
                'native/monster_native.pyd':file_hash(ROOT/'native/monster_native.pyd')},cpu_options=profile['options'],
        budget_unit='per half-move, including recurring GPU policy overhead',incumbent_finisher=False,
        cap='capture-only draw',repetition='threefold settled positions',
        note='Experimental search modes; GPU clock checks between complete batches; actual times logged'))
    rows=[]
    with worker_lease():
        needs_gpu=args.mode in ('guided','split','puct') or (not args.selfplay and args.opponent=='puct')
        nn=NNEvaluator(args.reference) if needs_gpu else None
        a=Player(args.mode,args.value,profile['options'],nn,args.seed)
        b=a if args.selfplay else Player(args.opponent,args.value,{},nn,args.seed)
        start=time.perf_counter();a.warmup()
        if b is not a:b.warmup()
        atomic_json(args.out/'warmup.json',dict(seconds=time.perf_counter()-start,
            device=str(nn.device) if nn else 'cpu',cpu_search_threads=1))
        with (args.out/'games.jsonl').open('x',encoding='utf-8') as log:
            for pair,entry in enumerate(entries):
                for a_white in ([True] if args.selfplay else [True,False]):
                    random.seed(args.seed+pair);np.random.seed(args.seed+pair)
                    game=MonsterChessGame(entry['fen']);game.white_half_pending=bool(entry['half'])
                    game.turn_count=int(entry['turn_count'])
                    initial=dict(fen=game.fen(),half=game.white_half_pending,turn_count=game.turn_count)
                    tracker=RepetitionTracker();tracker.record(game);a.reset()
                    if b is not a:b.reset()
                    decisions=[]
                    while not game.is_terminal() and tracker.fired_at is None:
                        is_a=args.selfplay or game.is_white_turn==a_white
                        player=a if is_a else b
                        budget=args.seconds if is_a else args.opponent_seconds
                        node_limit=args.node_limit if is_a else args.opponent_node_limit
                        started=time.perf_counter()
                        prior=prior_keys(tracker,game)
                        action,detail=player.choose(game,prior,max(.000001,budget-(time.perf_counter()-started)),node_limit)
                        legal=game.get_search_actions()
                        move=next((m for m in legal if m.uci()==action),None)
                        elapsed=time.perf_counter()-started
                        if move is None:raise ValueError('No legal action in live game')
                        decisions.append(dict(action=action,white=game.is_white_turn,half=game.white_half_pending,
                            candidate=is_a,seconds=elapsed,budget_seconds=budget,
                            overrun_seconds=max(0,elapsed-budget),**detail))
                        game.apply_search_action(move);tracker.record(game,len(decisions))
                    value=game.get_result() if game.is_terminal() else 0
                    value=int(value) if abs(value)>=1 else 0
                    row=dict(pair=pair,a_white=a_white,result_white=value,result_a=value if a_white else -value,
                        start=initial,final_fen=game.fen(),decisions=decisions,repetition=tracker.fired_at is not None)
                    rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    progress=dict(games=len(rows),white_wins=sum(r['result_white']>0 for r in rows),
                        black_wins=sum(r['result_white']<0 for r in rows),draws=sum(r['result_white']==0 for r in rows),
                        score=sum((r['result_a']+1)/2 for r in rows)/len(rows))
                    for color,name in [(True,'as_white'),(False,'as_black')]:
                        subset=[r for r in rows if r['a_white']==color]
                        progress[name]=sum((r['result_a']+1)/2 for r in subset)/len(subset) if subset else None
                    atomic_json(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
        peak=nn.torch.cuda.max_memory_allocated() if nn and nn.device.type=='cuda' else 0
        if peak>12*1024**3:raise ValueError('VRAM ceiling exceeded')
        atomic_json(args.out/'hardware.json',dict(cuda_peak_bytes=peak,device=str(nn.device) if nn else 'cpu'))
        atomic_json(args.out/'complete.json',progress)

if __name__=='__main__':main()
