"""Exact-search optimizations must preserve values through all Monster phases."""
from pathlib import Path
import random
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'native')]
import monster_native as native
from monster_chess import MonsterChessGame

OPTIONS=[{'pvs':True},{'fresh_tt':True},{'pvs':True,'fresh_tt':True,'tt_capacity':128}]

@pytest.mark.parametrize('options',OPTIONS)
def test_fixed_depth_values_in_random_walk_all_phases(options):
    rng=random.Random(9482);game=MonsterChessGame()
    path=ROOT/'models/candidates/search_leaf_leaf_001/epoch_003.bin'
    evaluator=native.CheapValue(str(path)) if path.exists() else None
    for i in range(45):
        if game.is_terminal():game=MonsterChessGame()
        fen=game.board.fen(en_passant='fen')
        kwargs=dict(pending=game.white_half_pending,turn_count=game.turn_count,
                    max_depth=3,seconds=15,evaluator=evaluator)
        if i%3==0 or i<3:
            a=native.alphabeta_search(fen,**kwargs)
            b=native.alphabeta_search(fen,**kwargs,**options)
            assert not a.interrupted and not b.interrupted
            assert (a.value,a.completed_depth)==(b.value,b.completed_depth)
            assert sum(b.phase_nodes)==b.nodes
            assert b.action in native.Game(fen,game.white_half_pending,game.turn_count).search_actions()
        actions=game.get_search_actions()
        if not actions:game=MonsterChessGame()
        else:game.apply_search_action(rng.choice(actions))

@pytest.mark.parametrize('fen,pending,count',[
    ('r3k2r/8/8/3pP3/8/8/8/4K3 w kq d6 0 1',True,55),
    ('k7/3P4/8/8/8/8/8/4K3 w - - 0 1',False,12),
    ('k7/8/8/8/8/8/3P4/4K3 w - - 0 1',False,149),
    ('k7/8/8/8/8/8/3P4/4K3 b - - 0 1',False,150)])
def test_edge_states_and_reverse_root_order(fen,pending,count):
    legal=native.Game(fen,pending,count).search_actions()
    kwargs=dict(pending=pending,turn_count=count,max_depth=3,seconds=15)
    a=native.alphabeta_search(fen,**kwargs)
    b=native.alphabeta_search(fen,**kwargs,pvs=True,fresh_tt=True,root_order=list(reversed(legal)))
    assert not a.interrupted and not b.interrupted
    assert a.value==b.value

def test_repetition_and_invalid_hints():
    fen='k7/8/8/8/8/8/3P4/4K3 w - - 0 1'
    key=' '.join(fen.split()[:4])
    a=native.alphabeta_search(fen,pvs=True,fresh_tt=True,prior_positions=[key,key])
    assert a.value==0 and a.action is None
    for options in ({'root_order':['a1a8']},{'root_order':['d2d3','d2d3']},{'tt_capacity':1048577}):
        with pytest.raises(ValueError):native.alphabeta_search(fen,**options)

def test_interrupted_research_has_only_last_completed_depth():
    fen=MonsterChessGame().board.fen(en_passant='fen')
    a=native.alphabeta_search(fen,pvs=True,node_limit=1)
    assert a.interrupted and a.completed_depth==0 and a.value is None
