import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from b2_nine_hour import model_panel
from b2_validation_analysis import intervals


def test_all_fixed_epoch_replications_present():
    panel=model_panel()
    assert len(panel)==9
    assert len(set(panel.values()))==9
    for seed in ('original','replica'):
        for arm in ('control','state_cnn'):
            for epoch in (8,15):
                assert panel[f'{seed}_{arm}_e{epoch}'].endswith(f'selected_epoch_{epoch:03d}.pt')


def test_paired_identical_models_zero_uncertainty():
    values=[[1,0],[.5,1],[0,.5]]
    result=intervals(values,values)
    assert all(v==dict(difference=0,lower95=0,upper95=0) for v in result.values())


def test_paired_color_specific_effect():
    result=intervals([[1,0]]*4,[[0,1]]*4)
    assert result['white']['difference']==1
    assert result['black']['difference']==-1
    assert result['overall']['difference']==0
