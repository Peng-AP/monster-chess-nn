"""Contracts for the oscillation selection penalty and the White first-half
guard (owner games 2026-07-17: standstill shuffle; queen check answered by
standing still because every m2 after the chosen m1 hung the king)."""
import os
import sys
from types import SimpleNamespace

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from monster_chess import MonsterChessGame
from mcts import (
    OSCILLATION_VISIT_PENALTY,
    _is_reversal,
    _m1_dooms_king,
    _oscillation_adjusted_visits,
    _own_previous_moves,
    _selected_child_value,
    _white_first_half_override,
)


def _mv(uci):
    return chess.Move.from_uci(uci)


def _children(*pairs):
    return [(SimpleNamespace(action=_mv(u)), u, v) for u, v in pairs]


# ----------------------------------------------------------------------
# Oscillation selection penalty
# ----------------------------------------------------------------------

def _black_root_after_shuffle():
    """Black rook just played h8->h7; White answered; Black to move again."""
    game = MonsterChessGame("1k5r/8/8/8/8/8/2P1K3/8 b - - 0 1")
    game.apply_action(_mv("h8h7"))
    game.apply_action((_mv("c2c3"), _mv("e2e3")))
    assert not game.is_white_turn
    return game


def test_own_previous_moves_black_root():
    game = _black_root_after_shuffle()
    assert _own_previous_moves(game) == [_mv("h8h7")]


def test_is_reversal():
    assert _is_reversal(_mv("h7h8"), [_mv("h8h7")])
    assert not _is_reversal(_mv("h7h6"), [_mv("h8h7")])


def test_penalty_discounts_only_the_reversal():
    game = _black_root_after_shuffle()
    info = _children(("h7h8", 100), ("h7h6", 95))
    adj = _oscillation_adjusted_visits(game, info)
    assert adj == [100 * (1.0 - OSCILLATION_VISIT_PENALTY), 95]
    # near-tie now breaks toward progress...
    assert adj[1] > adj[0]


def test_penalty_does_not_flip_a_clear_preference():
    game = _black_root_after_shuffle()
    info = _children(("h7h8", 100), ("h7h6", 40))
    adj = _oscillation_adjusted_visits(game, info)
    assert adj[0] > adj[1]  # 90 vs 40: the searched-best reversal still plays


def test_penalty_white_second_half_reversing_m1():
    """m2 undoing this turn's m1 (net null pair) is discounted."""
    game = MonsterChessGame("1k5r/8/8/8/8/8/2P1K3/8 w - - 0 1")
    game.apply_search_action(_mv("e2d2"))
    assert game.white_half_pending
    assert _mv("e2d2") in _own_previous_moves(game)
    info = _children(("d2e2", 50), ("c2c3", 50))
    adj = _oscillation_adjusted_visits(game, info)
    assert adj[0] < adj[1]


def test_penalty_without_history_is_noop():
    game = MonsterChessGame("1k5r/8/8/8/8/8/2P1K3/8 b - - 0 1")
    info = _children(("h8h7", 100), ("h8g8", 40))
    assert _oscillation_adjusted_visits(game, info) == [100, 40]


def test_penalty_noop_on_white_first_half():
    game = MonsterChessGame("1k5r/8/8/8/8/8/2P1K3/8 w - - 0 1")
    info = _children(("e2d2", 100), ("c2c3", 40))
    assert _oscillation_adjusted_visits(game, info) == [100, 40]


# ----------------------------------------------------------------------
# White first-half guard
# ----------------------------------------------------------------------

# Ka1 is checked by Qd4. After m1 Ka2 every m2 hangs the king
# (a1/b2 queen diagonal, a3 pawn b4, b1/b3 knight d2);
# after m1 Kb1 the completion Kc1 is safe (knight on d2 cannot reach c1).
DOOM_FEN = "6k1/8/8/8/1p1q4/8/3n4/K7 w - - 0 1"


def test_m1_dooms_king_detects_the_pocket():
    game = MonsterChessGame(DOOM_FEN)
    assert _m1_dooms_king(game, _mv("a1a2"))
    assert not _m1_dooms_king(game, _mv("a1b1"))


def test_first_half_override_swaps_doomed_m1():
    game = MonsterChessGame(DOOM_FEN)
    info = _children(("a1a2", 100), ("a1b1", 60))
    assert _white_first_half_override(game, _mv("a1a2"), info) == _mv("a1b1")


def test_first_half_override_keeps_safe_m1():
    game = MonsterChessGame(DOOM_FEN)
    info = _children(("a1b1", 100), ("a1a2", 60))
    assert _white_first_half_override(game, _mv("a1b1"), info) == _mv("a1b1")


def test_first_half_override_noop_on_second_half():
    game = MonsterChessGame(DOOM_FEN)
    game.apply_search_action(_mv("a1a2"))
    info = _children(("a2a1", 100),)
    assert _white_first_half_override(game, _mv("a2a1"), info) == _mv("a2a1")


# ----------------------------------------------------------------------
# Reported search value = selected child's Q, not the root average
# ----------------------------------------------------------------------

def _q_children(*triples):
    """(uci, visits, q) -> children_info with q_value exposed."""
    return [
        (SimpleNamespace(action=_mv(u), visit_count=v, q_value=q), u, v)
        for u, v, q in triples
    ]


def test_reported_value_is_selected_childs_q():
    # proven mate on the selected child; root average diluted by exploration
    info = _q_children(("h7h8", 300, 1.0), ("h7h6", 100, -0.2))
    assert _selected_child_value(info, _mv("h7h8"), fallback=0.7) == 1.0
    assert _selected_child_value(info, _mv("h7h6"), fallback=0.7) == -0.2


def test_reported_value_falls_back_when_unvisited():
    info = _q_children(("h7h8", 0, 0.0),)
    assert _selected_child_value(info, _mv("h7h8"), fallback=0.42) == 0.42
    assert _selected_child_value(info, _mv("a1a2"), fallback=0.42) == 0.42
