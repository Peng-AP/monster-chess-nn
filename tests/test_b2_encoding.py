import numpy as np
import pytest
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'native'))
from encoding import fen_to_tensor, mirror_tensor
from monster_chess import MonsterChessGame


def test_state_requires_clock():
    game = MonsterChessGame()
    with pytest.raises(ValueError, match='turn_count'):
        fen_to_tensor(game.fen(), input_channels=24)
    x = fen_to_tensor(game.fen(), input_channels=24, turn_count=0)
    assert x.shape == (8, 8, 24)
    assert np.all(x[:, :, 23] == 1)
    assert np.all(x[:, 0, 17] == -1)
    assert np.all(x[:, 7, 17] == 1)
    with pytest.raises(ValueError, match='mirror'):
        mirror_tensor(x)


@pytest.mark.parametrize('turn_count', [0, 73, 149, 150])
@pytest.mark.parametrize('half', [False, True])
def test_native_state_encoding(turn_count, half):
    import monster_native as mn
    fen = 'r3k2r/8/8/3pP3/8/8/8/4K3 w kq d6 0 1'
    expected = fen_to_tensor(fen, input_channels=24, turn_count=turn_count,
                             half_pending=half)
    actual = np.asarray(mn.encode_fen(fen, True, half, 24, turn_count), dtype=np.float32)
    np.testing.assert_array_equal(actual.reshape(8, 8, 24), expected)
    assert expected[5, 3, 22] == 1
    assert np.all(expected[:, :, 20:22] == 1)
