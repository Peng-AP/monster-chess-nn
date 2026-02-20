"""Tests for data processor tensor encoding and move indexing."""
import chess
import numpy as np

from config import STARTING_FEN, TURN_LAYER, PAWN_ADVANCEMENT_LAYER, POLICY_SIZE
from data_processor import fen_to_tensor, mirror_tensor, policy_dict_to_target
from encoding import move_to_index, mirror_move_index


def test_fen_to_tensor_shape():
    tensor = fen_to_tensor(STARTING_FEN)
    assert tensor.shape == (8, 8, 15)
    assert tensor.dtype == np.float32


def test_turn_layer_white():
    tensor = fen_to_tensor(STARTING_FEN, is_white_turn=True)
    assert tensor[0, 0, TURN_LAYER] == 1.0


def test_turn_layer_black():
    tensor = fen_to_tensor(STARTING_FEN, is_white_turn=False)
    assert tensor[0, 0, TURN_LAYER] == -1.0


def test_starting_fen_piece_placement():
    """Check that known pieces are in the right places for the starting FEN."""
    tensor = fen_to_tensor(STARTING_FEN)
    # White king on e1 (rank 0, file 4, layer 5)
    assert tensor[0, 4, 5] == 1.0
    # White pawns on c2, d2, e2, f2 (rank 1, files 2-5, layer 0)
    for f in [2, 3, 4, 5]:
        assert tensor[1, f, 0] == 1.0
    # Black king on e8 (rank 7, file 4, layer 11)
    assert tensor[7, 4, 11] == 1.0
    # Black pawns on rank 7 (rank 6, all files, layer 6)
    for f in range(8):
        assert tensor[6, f, 6] == 1.0


def test_pawn_advancement_gradient():
    """White pawns should have advancement gradient on layer 14."""
    tensor = fen_to_tensor(STARTING_FEN)
    # Pawns on rank 2 (index 1): advancement = (1-1)/6 = 0.0
    for f in [2, 3, 4, 5]:
        assert tensor[1, f, PAWN_ADVANCEMENT_LAYER] == 0.0

    # A pawn on rank 4 (index 3) should have advancement = (3-1)/6
    fen_adv = "4k3/8/8/8/3P4/8/8/4K3 w - - 0 1"
    tensor_adv = fen_to_tensor(fen_adv)
    assert abs(tensor_adv[3, 3, PAWN_ADVANCEMENT_LAYER] - 2.0 / 6.0) < 1e-6


def test_move_to_index_basic():
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(move)
    expected = move.from_square * 64 + move.to_square
    assert idx == expected


def test_move_to_index_range():
    """All indices should be in [0, 4095]."""
    for from_sq in range(64):
        for to_sq in range(64):
            move = chess.Move(from_sq, to_sq)
            idx = move_to_index(move)
            assert 0 <= idx < POLICY_SIZE


def test_mirror_move_index_involution():
    """Mirroring twice should return the original index."""
    for idx in [0, 100, 2048, 4095]:
        assert mirror_move_index(mirror_move_index(idx)) == idx


def test_mirror_tensor_involution():
    """Mirroring a tensor twice should return the original."""
    tensor = fen_to_tensor(STARTING_FEN)
    double_mirrored = mirror_tensor(mirror_tensor(tensor))
    np.testing.assert_array_equal(tensor, double_mirrored)


def test_policy_dict_to_target_normalization():
    """policy_dict_to_target should produce a vector summing to ~1."""
    policy_dict = {"e2e4": 0.5, "d2d4": 0.3, "c2c4": 0.2}
    target = policy_dict_to_target(policy_dict, is_white=False)
    assert target.shape == (POLICY_SIZE,)
    assert abs(target.sum() - 1.0) < 1e-5


def test_policy_dict_to_target_empty():
    """Empty/None policy should return all zeros."""
    target = policy_dict_to_target(None, is_white=False)
    assert target.sum() == 0.0
