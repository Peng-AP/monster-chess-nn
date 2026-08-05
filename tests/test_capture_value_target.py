"""Raw terminal provenance survives processing for capture-aligned WDL."""
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_processor import _convert_games_to_arrays  # noqa: E402
from train import get_targets  # noqa: E402


def _record(result):
    return {
        "fen": "7k/8/8/8/8/8/8/K7 w - - 0 1",
        "current_player": "white", "half": 0,
        "mcts_value": 0.0, "policy": {"a1a2": 1.0},
        "game_result": result,
    }


def test_capture_result_distinguishes_caps_from_true_king_captures():
    games = [{"records": [_record(value)]}
             for value in (-1.0, -0.5, 0.5, 1.0)]
    out = _convert_games_to_arrays(
        games, augment=False, include_capture_results=True)
    np.testing.assert_array_equal(
        out[-1], np.array([-1.0, 0.0, 0.0, 1.0], np.float32))


def test_capture_target_uses_preserved_values_without_remapping():
    capture = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    assert get_targets(np.zeros_like(capture), capture, "capture_result") is capture
