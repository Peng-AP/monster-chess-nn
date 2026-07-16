"""Contracts for the v18-recovery foundations: channel conversion parity,
decisive checkpoint scoring, and explicit-channel processing."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from encoding import fen_to_tensor
from model_diff import convert_channels
from train import _decisive_score
import data_processor

FENS = [
    # monster start: White king + 4 pawns vs full Black army
    ("rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3", "w", False),
    # advanced pawns both colors (nonzero progress planes both sides)
    ("rnbqkb1r/ppp1pppp/8/4P3/3p4/8/2PP4/4K3", "b", False),
    # White second half-move pending
    ("rnbqkbnr/pppppppp/8/8/4P3/8/2PP1P2/4K3", "w", True),
]


def _batch(channels):
    return np.stack([
        fen_to_tensor(f"{board} {turn} - - 0 1", is_white_turn=(turn == "w"),
                      half_pending=half, input_channels=channels)
        for board, turn, half in FENS
    ])


def test_convert_17_to_15_matches_direct_encoding():
    np.testing.assert_array_equal(convert_channels(_batch(17), 15), _batch(15))


def test_convert_15_to_17_matches_direct_encoding():
    np.testing.assert_array_equal(convert_channels(_batch(15), 17), _batch(17))


def test_convert_same_width_is_identity():
    x = _batch(15)
    assert convert_channels(x, 15) is x


def test_decisive_score_punishes_color_collapse():
    balanced = {
        "policy_top1_white": 0.30, "policy_top1_black": 0.30,
        "sign_acc_white": 0.85, "sign_acc_black": 0.85,
    }
    collapsed = {
        # better on three metrics, collapsed on Black policy
        "policy_top1_white": 0.60, "policy_top1_black": 0.10,
        "sign_acc_white": 0.95, "sign_acc_black": 0.90,
    }
    assert _decisive_score(balanced) > _decisive_score(collapsed)


def test_decisive_score_none_when_side_missing():
    assert _decisive_score({
        "policy_top1_white": 0.3, "policy_top1_black": None,
        "sign_acc_white": 0.8, "sign_acc_black": 0.8,
    }) is None


@pytest.mark.parametrize("channels,expected", [(15, 15), (17, 17), (None, 17)])
def test_processor_emits_requested_channel_width(channels, expected):
    board, turn, half = FENS[1]
    records = [{
        "fen": f"{board} {turn} - - 0 1",
        "current_player": "black",
        "half": half,
        "mcts_value": -0.4,
        "game_result": -1,
        "policy": {"d4d3": 1.0},
    }]
    X, _, _, _, _ = data_processor._convert_games_to_arrays(
        [{"records": records}], augment=False, input_channels=channels)
    assert X.shape == (1, 8, 8, expected)
