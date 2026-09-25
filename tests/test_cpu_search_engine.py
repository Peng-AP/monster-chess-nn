from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'native')]
from cpu_search_engine import CpuSearchEngine
from monster_chess import MonsterChessGame

@pytest.fixture
def value():
    path=ROOT/'models/candidates/search_leaf_leaf_001/epoch_003.bin'
    if not path.exists():pytest.skip('Experimental checkpoint absent')
    return path

class ReversePolicy:
    def batch_policies(self,games):return [np.arange(4096,dtype=np.float32) for _ in games]

def test_policy_changes_order_but_preserves_fixed_depth_value(value):
    g=MonsterChessGame()
    a,plain=CpuSearchEngine(value).choose(g,[],30,max_depth=3)
    b,guided=CpuSearchEngine(value,{'pvs':True,'fresh_tt':True},ReversePolicy()).choose(g,[],30,max_depth=3)
    assert plain['value']==guided['value']
    assert not plain['interrupted'] and not guided['interrupted']
    assert guided['policy_seconds']>0
    assert b in {m.uci() for m in g.get_search_actions()}

def test_spent_policy_clock_has_legal_fallback_without_search(value,monkeypatch):
    import cpu_search_engine
    clock=iter([0.,.2,.3])
    monkeypatch.setattr(cpu_search_engine.time,'perf_counter',lambda:next(clock))
    game=MonsterChessGame()
    action,detail=CpuSearchEngine(value,policy_evaluator=ReversePolicy()).choose(game,[],.1)
    assert detail['clock_exhausted_before_search']
    assert detail['depth']==0 and detail['nodes']==0 and detail['value'] is None
    assert action in {m.uci() for m in game.get_search_actions()}
