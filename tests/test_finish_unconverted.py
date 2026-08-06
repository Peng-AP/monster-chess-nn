"""Contracts for resuming unconverted (-0.5) games.

The first version of this tool replayed each game forward from its opening
record and matched FENs to recover the final position. That silently returned
None for every real game, because generation retains records side-selectively,
so consecutive records are not consecutive plies. A whole run reported
"unreplayable" instead of measuring anything. These pin the resume path.
"""
import os
import sys

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from finish_unconverted import resume_state  # noqa: E402

# A real mid-turn record: White has played one half-move and is still to move,
# so the FEN's turn field reads 'w' while half is 1.
MID_TURN = {"fen": "3k4/8/8/8/r4P2/q5q1/2K5/8 w - - 3 6", "half": 1}
BLACK_TO_MOVE = {"fen": "8/7k/8/8/8/7q/r7/2K3q1 b - - 119 48", "half": 0}


def test_resume_uses_final_record_not_a_forward_replay():
    # Sparse, non-consecutive records: a forward replay cannot connect these.
    game = resume_state([BLACK_TO_MOVE, MID_TURN])
    assert game.fen() == MID_TURN["fen"]


def test_mid_turn_record_restores_the_pending_half_move():
    game = resume_state([MID_TURN])
    assert game.white_half_pending is True
    assert game.is_white_turn is True


def test_black_to_move_record_has_no_pending_half():
    game = resume_state([BLACK_TO_MOVE])
    assert game.white_half_pending is False
    assert game.is_white_turn is False


def test_resumed_game_is_not_born_terminal():
    # These games stopped at MAX_GAME_TURNS. Carrying that count over would make
    # every resumed game terminal on arrival and convert nothing.
    game = resume_state([BLACK_TO_MOVE])
    assert game.turn_count == 0
    assert not game.is_terminal()
    assert game.get_search_actions()


def test_resumed_position_still_has_both_kings():
    game = resume_state([BLACK_TO_MOVE])
    assert game.board.king(chess.WHITE) is not None
    assert game.board.king(chess.BLACK) is not None
