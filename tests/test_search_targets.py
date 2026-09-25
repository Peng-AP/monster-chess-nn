from pathlib import Path
import math
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'native')]
import monster_native as native
from monster_chess import MonsterChessGame
from encoding import fen_to_tensor

@pytest.fixture
def constant(tmp_path):
    import torch
    from train_search_value import model,export
    net=model(8,4)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
        net[4].bias.fill_(math.atanh(.2))
    path=tmp_path/'constant.bin';export(net,path)
    return native.CheapValue(str(path))

@pytest.mark.parametrize('fen,pending',[
    ('k7/8/8/8/8/8/3P4/4K3 w - - 0 1',False),
    ('k7/8/8/8/8/8/3P4/4K3 w - - 0 1',True),
    ('k7/8/8/8/8/8/3P4/4K3 b - - 0 1',False),
    ('8/8/8/8/4k3/8/4K3/8 w - - 0 1',False),
    ('k7/8/8/8/8/8/8/r3K3 b - - 0 1',False)])
def test_teacher_backup_matches_fixed_turn_cpu_minimax(constant,fen,pending):
    tree=native.LabelTree(fen,pending,depth=2)
    expected=native.alphabeta_search(fen,pending=pending,evaluator=constant,max_depth=2,seconds=30)
    assert not expected.interrupted
    assert abs(tree.solve([.2]*tree.frontier_count())[0]-expected.value)<1e-6
    states=tree.record_states()
    assert len(states)==tree.record_count()
    assert all(c==1 and not h for _,_,h,c,_,_ in states[1:])

def test_capture_cap_and_history_precedence():
    fen='k7/8/8/8/8/8/3P4/4K3 w - - 0 1';key=' '.join(fen.split()[:4])
    cap=native.LabelTree(fen,turn_count=150)
    assert cap.solve([])==[0.] and cap.cap_hits==1
    capture=native.LabelTree('8/8/8/8/8/8/3P4/4K3 w - - 0 1',turn_count=150)
    assert capture.solve([])==[1.] and capture.cap_hits==0
    repeated=native.LabelTree(fen,prior_positions=[key,key])
    assert repeated.solve([])==[0.] and repeated.repetition_hits==1
    pending=native.LabelTree(fen,pending=True,prior_positions=[key,key])
    assert pending.node_count()>1

def test_label_input_preserves_raw_ep_and_turn_budget():
    fen='r3k2r/8/8/3pP3/8/8/8/4K3 w kq d6 0 1'
    tree=native.LabelTree(fen,True,55,depth=1)
    data,signs=tree.input_batch(0,512,24,True)
    actual=np.frombuffer(data,dtype='<f4').reshape(-1,8,8,24)
    for a,(_,f,h,c,_,_),sign in zip(actual,tree.record_states(),signs):
        np.testing.assert_array_equal(a,fen_to_tensor(f,f.split()[1]=='w',h,24,c))
        assert sign==(1 if f.split()[1]=='w' else -1)
    assert tree.record_states()[0][1].split()[3]=='d6'

def test_label_limits_and_invalid_values_fail_loudly():
    fen=MonsterChessGame().fen()
    with pytest.raises(ValueError,match='no partial'):native.LabelTree(fen,node_limit=1)
    with pytest.raises(ValueError):native.LabelTree(fen,prior_positions=[fen])
    t=native.LabelTree(fen,depth=1)
    for values in ([],[float('nan')]*t.frontier_count(),[2.]*t.frontier_count()):
        with pytest.raises(ValueError):t.solve(values)
    with pytest.raises(ValueError):t.input_batch(batch=4097)

@pytest.mark.parametrize('pvs',[False,True])
def test_leaf_lineage_is_exact_and_does_not_change_search(pvs):
    value=ROOT/'models/candidates/search_leaf_leaf_001/epoch_003.bin'
    if not value.exists():pytest.skip('Experimental model absent')
    g=MonsterChessGame();kwargs=dict(evaluator=native.CheapValue(str(value)),node_limit=20000,
        seconds=30,collect_leaves=12,leaf_seed=9123,pvs=pvs)
    a=native.alphabeta_search(g.fen(),**kwargs)
    b=native.alphabeta_search(g.fen(),**kwargs,collect_leaf_paths=True)
    for field in ('value','action','nodes','cutoffs','leaf_samples','completed_depth'):
        assert getattr(a,field)==getattr(b,field)
    assert a.leaf_paths==[] and len(b.leaf_paths)==len(b.leaf_samples)>0
    for path,(fen,pending,count,value) in zip(b.leaf_paths,b.leaf_samples):
        child=g.clone()
        for uci in path:
            action=next(m for m in child.get_search_actions() if m.uci()==uci)
            child.apply_search_action(action)
        assert child.board.fen(en_passant='fen')==fen
        assert (child.white_half_pending,child.turn_count)==(pending,count)

def test_rank_sign_and_target_margin():
    import torch
    from train_search_backed import ranking_loss
    a=torch.tensor([.5,-.5]);b=torch.tensor([0.,0.]);sign=torch.tensor([1.,-1.]);margin=torch.tensor([.25,.25])
    assert ranking_loss(a,b,sign,margin)==0
    assert ranking_loss(b,a,sign,margin)>0

def test_pair_construction_ignores_ties_and_conflicting_dedup_targets():
    from prepare_search_backed import make_pairs
    index={b'a':0,b'b':1,b'c':2}
    groups=[(1,[(b'a',.8),(b'b',.4),(b'c',.79)])]
    pairs=make_pairs(groups,index,np.array([.8,.4,.79]))
    np.testing.assert_array_equal(pairs,np.array([[0,1,1,.25]],np.float32))
    assert len(make_pairs(groups,index,np.array([.1,.4,.79])))==0
