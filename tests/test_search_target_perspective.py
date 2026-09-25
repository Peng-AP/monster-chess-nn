"""Recorded MCTS targets and outcomes do not share a value perspective."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_search_targets import white_targets


def test_black_search_values_are_flipped_but_white_halves_are_not():
    actual=white_targets(np.array([.7,.4,-.3]),np.array([1.,1.,-1.]))
    np.testing.assert_allclose(actual,[.7,.4,.3])


@pytest.mark.parametrize('values,side',[
    ([.2],[0]),([np.nan],[1]),([1.2],[1]),([.2,.3],[1]),
])
def test_bad_target_or_turn_plane_is_rejected(values,side):
    with pytest.raises(ValueError):
        white_targets(np.asarray(values),np.asarray(side))
