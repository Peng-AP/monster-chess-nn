from pathlib import Path
import json
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src'),str(ROOT/'native')]
import monster_native as native
from validate_cpu_scaling import numerical_states,TOLERANCE
from profile_cpu_search import VALUE
from cpu_search_engine import CpuSearchEngine
from monster_chess import MonsterChessGame
from search_cpu_scaling_campaign import extension_justified
from analyze_cpu_gpu import paired_difference

@pytest.fixture
def net():
    if not VALUE.exists():pytest.skip('Experimental model absent')
    return native.CheapValue(str(VALUE))

def test_incremental_sequence_matches_direct_with_refresh_and_arbitrary_jumps(net):
    states=numerical_states(1000)
    a,_,_=net.evaluate_sequence(states,False);b,updates,refreshes=net.evaluate_sequence(states,True)
    assert max(abs(x-y) for x,y in zip(a,b))<TOLERANCE
    assert updates>32 and refreshes>1
    assert net.evaluate_sequence([],True)==([],0,0)

def test_incremental_relative_falls_back_to_direct(tmp_path):
    import torch
    from train_search_value import model,export
    torch.manual_seed(9612);path=tmp_path/'relative.bin';export(model(32,16,6240).eval(),path)
    net=native.CheapValue(str(path));states=numerical_states(80)
    a,_,_=net.evaluate_sequence(states,False);b,updates,refreshes=net.evaluate_sequence(states,True)
    assert a==b and updates==0 and refreshes==len(states)

def test_opt_in_timers_do_not_change_search(net):
    for f,h,c in numerical_states(10)[:3]:
        kwargs=dict(pending=h,turn_count=c,evaluator=net,node_limit=15000,seconds=30)
        a=native.alphabeta_search(f,**kwargs);b=native.alphabeta_search(f,**kwargs,profile_search=True)
        for field in ('action','value','completed_depth','nodes','cutoffs','tt_hits'):
            assert getattr(a,field)==getattr(b,field)
        assert sum(a.profile_calls)==0 and sum(b.profile_calls)>0
        assert 0<sum(b.profile_seconds)<=b.elapsed_seconds

def test_node_limit_exposed_and_passed_to_cpu(net):
    engine=CpuSearchEngine(VALUE,{'incremental_eval':True})
    _,detail=engine.choose(MonsterChessGame(),[],2,node_limit=1)
    assert detail['nodes']==1 and detail['node_limit_reached'] and detail['node_limit']==1

def test_player_passes_assigned_cpu_limit():
    from search_cpu_gpu_match import Player
    from unittest.mock import Mock
    player=object.__new__(Player);player.mode='cpu';player.cpu=Mock()
    player.cpu.choose.return_value=('e1e2',{})
    player.choose('game',['prior'],8.,100000000)
    player.cpu.choose.assert_called_once_with('game',['prior'],8.,node_limit=100000000)

def test_extension_requires_overall_gain_without_observed_black_regression():
    assert extension_justified({'delta':.1,'black_delta':0})
    assert not extension_justified({'delta':.099,'black_delta':.5})
    assert not extension_justified({'delta':.2,'black_delta':-.01})

def test_paired_difference_can_pool_distinct_extensions_but_rejects_duplicates(tmp_path):
    folders=[]
    for arm in ('a','b'):
        selected=[]
        for index in range(2):
            folder=tmp_path/(arm+str(index));folder.mkdir();selected.append(folder)
            rows=[dict(start=dict(fen=str(index),half=False,turn_count=0),a_white=color,
                       result_a=1 if arm=='b' else 0) for color in (True,False)]
            (folder/'games.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        folders.append(selected)
    result=paired_difference(*folders)
    assert result['starts']==2 and result['delta']==.5
    with pytest.raises(ValueError,match='Duplicate'):paired_difference(folders[0]*2,folders[1]*2)
