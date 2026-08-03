"""The oracle must play a proven forced capture rather than reason about fences.

E0(b), 2026-08-03: on positions drawn from the set `mate_algo_applicable`
actually admits — not the narrower one `verify_scripted_mate` generates — the
fence heuristic converted 1 of 8 positions that held a forced capture in <= 3
Black moves, shuffling past the other 7 for more than 40 moves. The preflight
took the same 8 to 8 of 8, each in the proven minimum of 3.

These tests pin the preflight's precedence, its off switch, and the one
behaviour that must never regress: a budget hit falls through to the
heuristic instead of raising or claiming the position is drawn.
"""
import sys
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forced_capture import forced_capture_depth  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from scripted_mate import ScriptedMate  # noqa: E402


def play_black(fen, bot, max_moves=12):
    """Run `bot` as Black against a White defence that maximises survival."""
    state = MonsterChessGame(fen)
    state.turn_count, state._terminal, state._result = 0, False, None
    for ply in range(1, max_moves + 1):
        move = bot.select_move(state)
        if move is None:
            return None
        state.apply_action(move)
        if state.board.king(chess.WHITE) is None:
            return ply
        best, best_score = None, -1
        for action in state.get_legal_actions():
            child = state.clone()
            child.apply_action(action)
            if child.board.king(chess.BLACK) is None:
                best = action
                break
            child.turn_count, child._terminal, child._result = 0, False, None
            d = forced_capture_depth(child, max_black_moves=1, node_budget=20000)
            score = 99 if d is None else d
            if score > best_score:
                best, best_score = action, score
        if best is None:
            return None
        state.apply_action(best)
        state.turn_count, state._terminal, state._result = 0, False, None
    return None


class TestPreflightTakesPrecedence(unittest.TestCase):
    def test_an_immediate_capture_is_still_played(self):
        g = MonsterChessGame("r3k3/8/8/8/8/8/8/K7 b - - 0 1")
        move = ScriptedMate().select_move(g)
        child = g.clone()
        child.apply_action(move)
        self.assertIsNone(child.board.king(chess.WHITE))

    def test_a_forced_line_is_played_in_the_proven_number_of_moves(self):
        # Two rooks and a queen against a bare king, with NO immediate
        # capture available -- select_move answers those before the preflight
        # is consulted, so a mate-in-1 fixture cannot test this at all.
        fen = "4k3/8/8/8/8/2q5/1r5r/K7 b - - 0 1"
        g = MonsterChessGame(fen)
        proven = forced_capture_depth(g, max_black_moves=3)
        if proven is None:
            self.skipTest("fixture holds no forced capture within 3")
        plies = play_black(fen, ScriptedMate(forced_depth=3))
        self.assertIsNotNone(plies, "preflight failed to convert a forced win")
        self.assertLessEqual(plies, proven)

    def test_the_preflight_reports_when_it_fires(self):
        g = MonsterChessGame("4k3/8/8/8/8/2q5/1r5r/K7 b - - 0 1")
        bot = ScriptedMate(forced_depth=3)
        bot.select_move(g)
        self.assertEqual(bot.forced_hits, 1)


class TestDefaultIsOptIn(unittest.TestCase):
    def test_the_preflight_is_off_unless_asked_for(self):
        # Owner's call 2026-08-03. His standing veto covers changes to the
        # oracle, so the shipped default must leave generation byte-identical
        # to pre-preflight behaviour; callers opt in explicitly. A future
        # change flipping this default silently would breach that, so it is
        # pinned rather than left to convention.
        self.assertEqual(ScriptedMate().forced_depth, 0)

    def test_opting_in_actually_engages_the_search(self):
        g = MonsterChessGame("4k3/8/8/8/8/2q5/1r5r/K7 b - - 0 1")
        off, on = ScriptedMate(), ScriptedMate(forced_depth=3)
        off.select_move(g)
        on.select_move(g)
        self.assertEqual(off.forced_hits, 0)
        self.assertGreater(on.forced_hits, 0)


class TestOffSwitch(unittest.TestCase):
    def test_forced_depth_zero_restores_the_old_behaviour(self):
        # The comparison that produced the 1-of-8 measurement must stay
        # reproducible, so the heuristic has to remain reachable on its own.
        g = MonsterChessGame("4k3/8/8/8/8/2q5/1r5r/K7 b - - 0 1")
        bot = ScriptedMate(forced_depth=0)
        move = bot.select_move(g)
        self.assertIsNotNone(move)
        self.assertEqual(bot.forced_hits, 0)


class TestBudgetFallsThrough(unittest.TestCase):
    def test_a_budget_hit_returns_a_move_rather_than_raising(self):
        g = MonsterChessGame("4k3/8/8/8/8/2q5/1r5r/K7 b - - 0 1")
        bot = ScriptedMate(forced_depth=3, forced_budget=1)
        move = bot.select_move(g)
        self.assertIsNotNone(move, "exhaustion must fall through to the heuristic")
        self.assertIn(move, list(g.get_search_actions()))


if __name__ == "__main__":
    unittest.main()
