"""Threefold repetition as a driver-level rule.

Why it is not in `is_terminal`: repetition is path-dependent, so putting it in
the search reintroduces the graph-history problem and fights a transposition
table. Why it is off by default: it is a RULES CHANGE. It makes every earlier
number incomparable, the same break the 2026-08-03 captures-only correction
caused, and it lowers Black -- a capped ending scores -0.5 today and 0.0 under
this rule.

Measured 2026-08-16: in the capped games the network drives, the last 100
records hold as few as 29 distinct positions. In the ones ScriptedMate drives
they hold 94-98, because the oracle has explicit anti-repetition drift -- so
this rule barely touches oracle games.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monster_chess import MonsterChessGame  # noqa: E402
from repetition import (RepetitionTracker, position_key,  # noqa: E402
                        repetition_enabled)

ENDGAME = "6k1/8/8/8/8/8/8/r3K3 b - - 0 1"


class TestOptIn(unittest.TestCase):
    def test_disabled_without_the_env_flag(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(repetition_enabled())

    def test_enabled_with_the_flag(self):
        with mock.patch.dict(os.environ, {"MONSTER_REPETITION": "1"}):
            self.assertTrue(repetition_enabled())

    def test_a_disabled_tracker_never_fires(self):
        tracker = RepetitionTracker(enabled=False)
        game = MonsterChessGame(fen=ENDGAME)
        for _ in range(10):
            self.assertFalse(tracker.record(game))


class TestThreefold(unittest.TestCase):
    def test_fires_on_the_third_occurrence_not_the_second(self):
        tracker = RepetitionTracker(enabled=True)
        game = MonsterChessGame(fen=ENDGAME)
        self.assertFalse(tracker.record(game))   # 1st
        self.assertFalse(tracker.record(game))   # 2nd
        self.assertTrue(tracker.record(game))    # 3rd

    def test_distinct_positions_do_not_accumulate_together(self):
        tracker = RepetitionTracker(enabled=True)
        a = MonsterChessGame(fen=ENDGAME)
        b = MonsterChessGame(fen="6k1/8/8/8/8/8/8/r4K2 b - - 0 1")
        for _ in range(2):
            self.assertFalse(tracker.record(a))
            self.assertFalse(tracker.record(b))
        self.assertTrue(tracker.record(a))

    def test_a_repetition_draw_carries_no_lean(self):
        """The +-0.5 cap relabel is a proxy for an UNFINISHED game.

        A position repeated three times is drawn by rule, so it must not
        inherit the lean that flatters whoever was ahead when the clock ran out.
        """
        self.assertEqual(RepetitionTracker(enabled=True).draw_result, 0.0)

    def test_records_the_ply_it_fired_on(self):
        tracker = RepetitionTracker(enabled=True)
        game = MonsterChessGame(fen=ENDGAME)
        tracker.record(game, 10)
        tracker.record(game, 20)
        tracker.record(game, 30)
        self.assertEqual(tracker.fired_at, 30)


class TestPositionIdentity(unittest.TestCase):
    def test_mid_turn_states_are_not_counted(self):
        """Between White's two half-moves neither side can claim anything.

        Counting the mid-turn state would fire on ordinary play, because every
        White turn passes through one.
        """
        tracker = RepetitionTracker(enabled=True)
        game = MonsterChessGame()
        game.apply_search_action(game.get_search_actions()[0])
        self.assertTrue(game.white_half_pending)
        for _ in range(5):
            self.assertFalse(tracker.record(game))

    def test_side_to_move_is_part_of_the_identity(self):
        white = MonsterChessGame(fen="6k1/8/8/8/8/8/8/r3K3 w - - 0 1")
        black = MonsterChessGame(fen=ENDGAME)
        self.assertNotEqual(position_key(white), position_key(black))

    def test_move_counters_are_excluded(self):
        """They always differ; including them would mean nothing repeats."""
        early = MonsterChessGame(fen="6k1/8/8/8/8/8/8/r3K3 b - - 0 1")
        late = MonsterChessGame(fen="6k1/8/8/8/8/8/8/r3K3 b - - 99 60")
        self.assertEqual(position_key(early), position_key(late))

    def test_en_passant_rights_separate_positions(self):
        """Only a *capturable* ep square counts, which is the FEN convention.

        python-chess emits the square only when some pawn can actually take it,
        and standard repetition uses the same rule -- so the Black d4 pawn has
        to exist for e3 to be part of the identity at all.
        """
        without = MonsterChessGame(
            fen="rnbqkbnr/pppppppp/8/8/3pP3/8/2PP1P2/4K3 b - - 0 1")
        with_ep = MonsterChessGame(
            fen="rnbqkbnr/pppppppp/8/8/3pP3/8/2PP1P2/4K3 b - e3 0 1")
        self.assertNotEqual(position_key(without), position_key(with_ep))


if __name__ == "__main__":
    unittest.main()
