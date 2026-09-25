"""Geometry is a representation, not a hand-coded tactical preference."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from encoding import fen_to_tensor
from search_value_features import sparse_features,dense_features,MAX_ACTIVE
from train_search_value import features


def tensor(fen,pending=False,count=0):
    return fen_to_tensor(fen,fen.split()[1]=='w',pending,24,count)[None]


def test_absolute_features_unchanged_and_scatter_indices_unique():
    p=tensor('r3k2r/8/8/3pP3/8/8/8/4K3 w kq d6 0 1',True,55)
    for relative in (False,True):
        actual=dense_features(p,relative)
        np.testing.assert_array_equal(actual[:,:840],features(p))
        indices,weights=sparse_features(p,relative)
        assert len(np.unique(indices[0]))==MAX_ACTIVE
        assert np.count_nonzero(weights[0])>0


def test_relative_offsets_are_translation_invariant_but_absolute_board_is_not():
    a=dense_features(tensor('8/8/8/8/4k3/8/3P4/2K5 w - - 0 1'))
    b=dense_features(tensor('8/8/8/8/5k2/8/4P3/3K4 w - - 0 1'))
    assert not np.array_equal(a[:,:768],b[:,:768])
    np.testing.assert_array_equal(a[:,840:],b[:,840:])


def test_missing_king_has_no_center_features():
    p=tensor('8/8/8/8/8/8/3P4/2K5 w - - 0 1')
    actual=dense_features(p)
    assert actual[:,840:3540].sum()==2
    assert actual[:,3540:].sum()==0


def test_piece_offset_orientation_and_king_identity():
    p=tensor('8/8/8/8/4k3/8/3P4/2K5 w - - 0 1')
    a=dense_features(p)[0]
    # White pawn d2 from White king c1: (+1,+1); from Black king e4: (-1,-2).
    assert a[840+8*15+8]==1
    assert a[840+2700+5*15+6]==1


def test_multiple_kings_rejected():
    with pytest.raises(ValueError):
        dense_features(tensor('8/8/8/8/4k3/8/3P4/2KK4 w - - 0 1'))
