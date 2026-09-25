"""Reusable experimental CPU search with optional GPU root-policy ordering.

One per-move clock pays for encoding, policy inference and native search. Value
and proof semantics remain those of alpha-beta, even when GPU ordering is used.
"""
from pathlib import Path
import sys
import time
import math

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'native'))
import monster_native as native
from encoding import move_to_policy_index


class CpuSearchEngine:
    def __init__(self,value_path,options=None,policy_evaluator=None):
        self.value=native.CheapValue(str(value_path))
        self.options=dict(options or {})
        allowed={'pvs','fresh_tt','tt_capacity','incremental_eval'}
        if set(self.options)-allowed:raise ValueError('Unknown CPU search options')
        self.policy_evaluator=policy_evaluator

    def choose(self,game,prior_positions,seconds,max_depth=12,node_limit=10000000):
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite clock required')
        started=time.perf_counter();hint=None;policy_seconds=0.
        legal=None
        if self.policy_evaluator is not None:
            legal=game.get_search_actions()
            policies=self.policy_evaluator.batch_policies([game])
            policy=policies[0]
            hint=sorted(legal,key=lambda m:(-float(policy[move_to_policy_index(m,len(policy)>4096)]),m.uci()))
            hint=[m.uci() for m in hint]
            policy_seconds=time.perf_counter()-started
        fen=game.board.fen(en_passant='fen')
        remaining=seconds-(time.perf_counter()-started)
        if remaining<=0:
            legal=legal if legal is not None else game.get_search_actions()
            action=hint[0] if hint else (legal[0].uci() if legal else None)
            return action,dict(alphabeta=True,value=None,depth=0,nodes=0,interrupted=True,
                core_seconds=0.,policy_seconds=policy_seconds,clock_exhausted_before_search=True)
        result=native.alphabeta_search(fen,pending=game.white_half_pending,turn_count=game.turn_count,
            prior_positions=prior_positions,evaluator=self.value,seconds=remaining,
            max_depth=max_depth,node_limit=node_limit,root_order=hint,**self.options)
        detail=dict(alphabeta=True,value=result.value,depth=result.completed_depth,nodes=result.nodes,
            interrupted=result.interrupted,core_seconds=result.elapsed_seconds,policy_seconds=policy_seconds,
            eval_cache_hits=result.eval_cache_hits,tt_hits=result.tt_hits,tt_cutoffs=result.tt_cutoffs,
            extension_nodes=result.extension_nodes,max_ply_reached=result.max_ply_reached,
            pvs_scouts=result.pvs_scouts,pvs_researches=result.pvs_researches,
            tt_rejected=result.tt_rejected,phase_nodes=result.phase_nodes,
            phase_generated=result.phase_generated,clock_exhausted_before_search=False)
        detail.update(node_limit=node_limit,node_limit_reached=result.interrupted and result.nodes>=node_limit)
        detail.update(incremental_updates=result.incremental_updates,eval_refreshes=result.eval_refreshes)
        detail.update(depth_limit=max_depth,depth_limit_reached=result.completed_depth==max_depth
                      and result.action is not None and result.value is not None and abs(result.value)<1)
        return result.action,detail
