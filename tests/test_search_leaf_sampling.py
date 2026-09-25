"""Opt-in leaf instrumentation must not change fixed-node search decisions."""
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'native'))
import monster_native as native


def evaluator():
    path=ROOT/'models/candidates/search_relative_absolute_001/epoch_007.bin'
    if not path.exists():pytest.skip('Local experiment checkpoint unavailable')
    return native.CheapValue(str(path))


@pytest.mark.parametrize('pending,side',[(False,'w'),(True,'w'),(False,'b')])
def test_samples_do_not_change_fixed_node_search(pending,side):
    fen=f'k7/8/8/8/8/8/3P4/4K3 {side} - - 0 1'
    args=dict(pending=pending,evaluator=evaluator(),node_limit=10000,seconds=30,max_depth=8)
    baseline=native.alphabeta_search(fen,**args)
    sampled=native.alphabeta_search(fen,collect_leaves=17,leaf_seed=52,**args)
    again=native.alphabeta_search(fen,collect_leaves=17,leaf_seed=52,**args)
    for field in ('action','value','nodes','completed_depth','cutoffs','tt_hits','eval_cache_hits'):
        assert getattr(sampled,field)==getattr(baseline,field)
    assert baseline.leaf_samples==[]
    assert 0<len(sampled.leaf_samples)<=17
    assert sampled.leaf_samples==again.leaf_samples
    for fen,half,count,value in sampled.leaf_samples:
        assert not half
        assert value==args['evaluator'].evaluate(fen,half,count)


def test_terminal_roots_are_not_neural_samples():
    result=native.alphabeta_search('k7/8/8/8/8/8/3P4/4K3 w - - 0 1',
        turn_count=150,evaluator=evaluator(),collect_leaves=17)
    assert result.value==0
    assert result.leaf_samples==[]
    assert result.leaf_evaluations==0


def test_sample_limit_rejected():
    with pytest.raises(ValueError):
        native.alphabeta_search('k7/8/8/8/8/8/3P4/4K3 w - - 0 1',collect_leaves=4097)
