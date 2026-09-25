"""Run after MCSV002 native integration; preserve all MCSV001 contracts."""
import random
from pathlib import Path
import struct
import sys
import tempfile
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'native')]
import monster_native as native
from monster_chess import MonsterChessGame
from encoding import fen_to_tensor
from train_search_value import model,export
from search_value_features import dense_features


def test_relative_native_features_and_export_match_torch():
    torch.manual_seed(71)
    net=model(32,16,6240).eval()
    cases=[
        ('r3k2r/8/8/3pP3/8/8/8/4K3 w kq d6 0 1',True,55),
        ('8/3Q4/8/8/4k3/8/8/K7 b - - 0 1',False,149),
        ('8/8/8/8/8/8/3P4/K7 w - - 0 1',False,150),
    ]
    rng=random.Random(9157)
    g=MonsterChessGame()
    for i in range(90):
        if g.is_terminal() or not g.get_search_actions():g=MonsterChessGame()
        if i%3==0:
            cases.append((g.board.fen(en_passant='fen'),g.white_half_pending,g.turn_count))
        g.apply_search_action(rng.choice(g.get_search_actions()))
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'relative.bin'
        export(net,path)
        evaluator=native.CheapValue(str(path))
        assert evaluator.input_count==6240
        for fen,pending,count in cases:
            expected=dense_features(fen_to_tensor(fen,fen.split()[1]=='w',pending,24,count)[None])[0]
            actual=np.zeros(6240,dtype=np.float32)
            for i,v in native.CheapValue.features(fen,pending,count,relative=True):
                actual[i]=v
            np.testing.assert_array_equal(actual,expected)
            with torch.no_grad():reference=net(torch.from_numpy(expected)).item()
            assert abs(reference-evaluator.evaluate(fen,pending,count))<1e-6


def test_feature_version_dimension_mismatch_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'bad.bin'
        path.write_bytes(struct.pack('<8sIII',b'MCSV002\0',840,32,16))
        with pytest.raises(ValueError):native.CheapValue(str(path))


def test_sparse_reconstruction_matches_dense_forward_and_gradients():
    from search_value_features import sparse_features,MAX_ACTIVE
    p=fen_to_tensor(MonsterChessGame().fen(),True,False,24,0)[None]
    for relative,inputs in [(False,840),(True,6240)]:
        ids,weights=sparse_features(p,relative)
        reconstructed=torch.zeros((1,inputs+MAX_ACTIVE))
        reconstructed.scatter_(1,torch.from_numpy(ids).long(),torch.from_numpy(weights))
        actual=reconstructed[:,:inputs]
        expected=torch.from_numpy(dense_features(p,relative))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        net=model(16,8,inputs)
        net(actual).sum().backward()
        gradients=[x.grad.clone() for x in net.parameters()]
        net.zero_grad()
        net(expected).sum().backward()
        for a,b in zip(gradients,net.parameters()):torch.testing.assert_close(a,b.grad,rtol=0,atol=0)
