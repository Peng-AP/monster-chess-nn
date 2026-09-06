"""Held-out probe records states/half-moves without modifying human data."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from match_evidence import read_rows
from probe_human_line import reconstruct, force, decision


def test_reconstruct_restores_turn_count_and_admits_ambiguous_white_path():
    source = ROOT / "data/raw/human_games/black_2026_07/game_00031.jsonl"
    if not source.exists():
        pytest.skip("local held-out human game is not in a clean clone")
    rows = read_rows(source)
    game, repetition, history = reconstruct(rows, 14)
    assert game.fen() == rows[14]["fen"]
    assert game.turn_count == 14
    assert not game.white_half_pending
    assert any(len(h["possible_paths"]) > 1 for h in history)
    force(game, ["c4b3"], repetition)
    assert game.white_half_pending
    assert game.turn_count == 14
    assert "f2f3" not in [a.uci() for a in game.get_search_actions()]
    force(game, ["b3a2"], repetition)
    assert game.fen() == rows[15]["fen"]
    assert game.turn_count == 15


def test_reject_missing_history_and_impossible_transition():
    with pytest.raises(ValueError, match="initial"):
        reconstruct([{"fen": "8/8/8/8/8/8/4P3/4K2k w - - 0 1"}], 0)
    from monster_chess import MonsterChessGame
    rows = [{"fen": MonsterChessGame().fen()}] * 2
    with pytest.raises(ValueError, match="no legal transition"):
        reconstruct(rows, 1)


def test_decision_records_each_half_value_separately():
    from monster_chess import MonsterChessGame
    game = MonsterChessGame()
    class Engine:
        def get_best_action(self, state, temperature):
            uci, value = ("e2e4", .1) if not state.white_half_pending else ("e4e5", -.2)
            action = next(a for a in state.get_search_actions() if a.uci() == uci)
            return action, {uci: 1.0}, value
    first = decision(game, Engine(), 0)
    second = decision(game, Engine(), 0)
    assert first["root_value_side_to_move"] == .1
    assert second["root_value_side_to_move"] == -.2
    assert not first["before"]["half"] and second["before"]["half"]
    assert first["after"]["turn_count"] == 0
    assert second["after"]["turn_count"] == 1
