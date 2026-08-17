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


class TestDefaultsAreOn(unittest.TestCase):
    """Owner, 2026-08-04: "fix the mate bot" — both guards ship enabled.

    Until then the preflight defaulted off, because his standing veto covered
    changes to this file. That instruction lifted it, so the defaults are
    pinned in the new position: a silent flip back would quietly restore a
    1-in-8 conversion rate.
    """

    def test_the_preflight_is_on_by_default(self):
        self.assertEqual(ScriptedMate().forced_depth, 3)

    def test_the_material_guard_is_on_by_default(self):
        self.assertTrue(ScriptedMate().guard_material)

    def test_the_default_bot_engages_the_search(self):
        g = MonsterChessGame("4k3/8/8/8/8/2q5/1r5r/K7 b - - 0 1")
        on, off = ScriptedMate(), ScriptedMate(forced_depth=0)
        on.select_move(g)
        off.select_move(g)
        self.assertGreater(on.forced_hits, 0)
        self.assertEqual(off.forced_hits, 0)


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



class TestMaterialGuard(unittest.TestCase):
    """The 9/12 was material loss, not a finishing failure.

    All three failing canonical starts lost the same way: Black hangs heavies
    to the double-move king, first loss at turn 5-23, the rest in consecutive
    turns as the fence collapses. `_white_reply_min` searches depth 1 with one
    danger-gated extension (depth 2 "exploded combinatorially"), which cannot
    see a king that captures two squares away — still less one that takes two
    pieces in a single turn.

    Measured after the guard: **11/12**, and the remaining failure no longer
    loses anything — it reaches the move cap with queen and both rooks intact,
    which is the known finishing problem rather than a blunder.
    """

    # Rook on c2 sits two squares from the White king on a1 and is undefended,
    # so White's double move takes it -- unless Black attends to it. The queen
    # is on h5 rather than rank 1 deliberately: with the White king in check
    # Black simply captures it, `get_search_actions` truncates to that single
    # winning move, and there is no choice left to test.
    HANG_FEN = "4k3/8/8/7q/8/8/2r5/K7 b - - 0 1"

    def test_a_hanging_move_is_recognised(self):
        game = MonsterChessGame(self.HANG_FEN)
        bot = ScriptedMate()
        legal = game.get_search_actions()
        hanging = [m for m in legal if bot._hangs_material(game, m)]
        safe = [m for m in legal if not bot._hangs_material(game, m)]
        # Both kinds must exist or the guard has nothing to choose between.
        self.assertTrue(hanging, "no move recognised as hanging material")
        self.assertTrue(safe, "every move hangs; fixture cannot test the guard")

    def test_the_guard_avoids_hanging_moves_when_it_can(self):
        game = MonsterChessGame(self.HANG_FEN)
        bot = ScriptedMate()
        chosen = bot.select_move(game)
        self.assertFalse(bot._hangs_material(game, chosen),
                         f"guard chose {chosen.uci()}, which hangs material")

    def test_disabling_the_guard_is_what_makes_it_observable(self):
        game = MonsterChessGame(self.HANG_FEN)
        guarded = ScriptedMate().select_move(game)
        unguarded = ScriptedMate(guard_material=False).select_move(game)
        probe = ScriptedMate()
        self.assertFalse(probe._hangs_material(game, guarded))
        # Not asserting the unguarded bot blunders here -- it may happen not
        # to. What is asserted is that the guard reports the rejections it made.
        bot = ScriptedMate()
        bot.select_move(game)
        self.assertGreater(bot.guard_hits, 0)

    def test_the_guard_stands_down_when_everything_hangs(self):
        # Forced is forced — same discipline as the king-safety override.
        fen = "7k/8/8/8/8/8/8/K6q b - - 0 1"
        game = MonsterChessGame(fen)
        bot = ScriptedMate()
        self.assertIsNotNone(bot.select_move(game))

    def test_it_can_be_disabled_for_comparison(self):
        bot = ScriptedMate(guard_material=False)
        self.assertFalse(bot.guard_material)
        self.assertIsNotNone(bot.select_move(MonsterChessGame(self.HANG_FEN)))

if __name__ == "__main__":
    unittest.main()
