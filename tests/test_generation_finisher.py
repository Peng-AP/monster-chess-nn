"""Contract tests for the exact finisher inside data generation.

Why it exists: law 1a says the model plays won positions as lost because the
CORPUS says they are -- Black converts 88% with White on a bare king but only
36% from king+3-pawns. REPORT.md section 30 showed those capped games are
unconverted wins, with a forced king capture inside four Black moves in 6 of 24
sampled games. Running the exact search during generation ends such a game with
a REAL king capture, so the label becomes a true Black win instead of a -0.5
move-limit relabel.

What must hold: it is OFF unless MONSTER_FINISHER is set, so no historical
generation changes; it runs only where the scripted oracle abstains, leaving
that verified class untouched; it never fires for White or mid-half; and a
search that finds nothing falls through to the network rather than being read
as "no win".
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_generation as dg  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from scripted_mate import mate_algo_applicable  # noqa: E402

# Bare White king, two Black rooks: won, but only TWO heavies, so the scripted
# oracle abstains (it requires three). Exactly the gap the finisher fills.
# Note the rooks are far from the king, so this is a ladder mate rather than a
# capture forced inside the horizon -- it exercises the gate, not a hit.
BARE_KING_TWO_ROOKS = "r5kr/8/8/8/8/8/8/4K3 b - - 0 1"
# Rook on e8, White king on e1, open file: the capture is available right now.
IMMEDIATE_CAPTURE = "4r1k1/8/8/8/8/8/8/4K3 b - - 0 1"
# Same, plus a White pawn: still inside the material ceiling.
KING_AND_PAWN = "r5kr/8/8/8/8/8/4P3/4K3 b - - 0 1"
# Black has nothing to capture with: no forced win exists.
LONE_BLACK_KING = "6k1/8/8/8/8/8/4P3/4K3 b - - 0 1"


class TestFinisherIsOptIn(unittest.TestCase):
    def test_enabled_by_default(self):
        """On since 2026-08-16: the native solver made it ~18.8x cheaper, and
        dropping the oracle handed it the bare-king class."""
        with mock.patch.dict(os.environ, {}, clear=True):
            enabled, _depth, _nodes, _mat = dg._finisher_settings()
        self.assertTrue(enabled)

    def test_disabled_by_the_escape_hatch(self):
        with mock.patch.dict(os.environ, {"MONSTER_NO_FINISHER": "1"},
                             clear=True):
            enabled, _depth, _nodes, _mat = dg._finisher_settings()
        self.assertFalse(enabled)

    def test_enabled_defaults_to_the_measured_depth_four(self):
        # REPORT.md section 30.1: depth 4 completed all 72 probe positions with
        # zero budget exhaustion; depth 5 cost ~2 orders of magnitude more.
        with mock.patch.dict(os.environ, {}, clear=True):
            enabled, depth, nodes, _mat = dg._finisher_settings()
        self.assertTrue(enabled)
        self.assertEqual(depth, 4)
        self.assertGreater(nodes, 0)

    def test_depth_and_budget_are_tunable(self):
        env = {"MONSTER_FINISHER_DEPTH": "3",
               "MONSTER_FINISHER_NODES": "1234"}
        with mock.patch.dict(os.environ, env, clear=True):
            _enabled, depth, nodes, _mat = dg._finisher_settings()
        self.assertEqual(depth, 3)
        self.assertEqual(nodes, 1234)

    def test_garbage_overrides_fall_back_to_defaults(self):
        env = {"MONSTER_FINISHER_DEPTH": "deep",
               "MONSTER_FINISHER_NODES": ""}
        with mock.patch.dict(os.environ, env, clear=True):
            _enabled, depth, nodes, _mat = dg._finisher_settings()
        self.assertEqual(depth, dg.FINISHER_DEFAULT_DEPTH)
        self.assertEqual(nodes, dg.FINISHER_DEFAULT_NODES)


class TestFinisherGate(unittest.TestCase):
    def test_never_fires_on_white_to_move(self):
        self.assertFalse(_finisher_applicable(MonsterChessGame()))

    def test_never_fires_mid_white_half(self):
        game = MonsterChessGame()
        game.apply_search_action(game.get_search_actions()[0])
        self.assertTrue(game.white_half_pending)
        self.assertFalse(_finisher_applicable(game))

    def test_does_not_fire_while_white_still_has_pawns(self):
        """An exact search in the opening costs time where it cannot help."""
        game = MonsterChessGame()
        game.apply_search_action(game.get_search_actions()[0])
        game.apply_search_action(game.get_search_actions()[0])
        self.assertFalse(game.is_white_turn)
        self.assertFalse(_finisher_applicable(game))

    def test_fires_against_a_bare_king(self):
        self.assertTrue(_finisher_applicable(
            MonsterChessGame(BARE_KING_TWO_ROOKS)))

    def test_fires_against_king_and_one_pawn(self):
        self.assertTrue(_finisher_applicable(MonsterChessGame(KING_AND_PAWN)))


class TestFinisherRespectsTheOracle(unittest.TestCase):
    """The oracle's class is verified; the finisher must not disturb it."""

    def test_the_two_rook_case_is_outside_the_oracle(self):
        game = MonsterChessGame(BARE_KING_TWO_ROOKS)
        self.assertFalse(mate_algo_applicable(game),
                         "oracle needs 3+ heavies; this is the finisher's gap")
        self.assertTrue(_finisher_applicable(game))

    def test_generation_consults_the_finisher_only_where_oracle_abstains(self):
        source = (ROOT / "src" / "data_generation.py").read_text(
            encoding="utf-8")
        # The finisher is gated on the oracle NOT taking the position, and the
        # oracle branch and the finisher branch key off the same decision.
        self.assertIn("oracle_here = scripted_mate_on "
                      "and _mate_algo_applicable(game)", source)
        self.assertIn("and not oracle_here", source)
        self.assertIn("_finisher_applicable(game, finisher_material_max)",
                      source)


class TestFinisherMoves(unittest.TestCase):
    def test_returns_a_legal_playable_move_when_a_win_is_forced(self):
        import chess
        game = MonsterChessGame(IMMEDIATE_CAPTURE)
        move = dg._finisher_move(game, depth=4, node_budget=2_000_000)
        self.assertIsNotNone(move)
        self.assertIn(move, game.get_legal_actions())
        game.apply_search_action(move)     # must not raise
        self.assertIsNone(game.board.king(chess.WHITE),
                          "a finisher hit must actually end the game")

    def test_no_forced_win_returns_none_so_play_falls_through(self):
        """'No answer' is never 'no win' -- the network still gets the move."""
        game = MonsterChessGame(LONE_BLACK_KING)
        move = dg._finisher_move(game, depth=4, node_budget=200_000)
        self.assertIsNone(move)

    def test_a_starved_budget_returns_none_rather_than_asserting(self):
        game = MonsterChessGame(BARE_KING_TWO_ROOKS)
        move = dg._finisher_move(game, depth=4, node_budget=1)
        self.assertIsNone(move)


class TestExhaustionStaysDistinguishable(unittest.TestCase):
    """A starved budget must not look like a proven absence of a win.

    Play falls through to the network either way, but measurement cannot:
    the 2026-08-16 pilot's whole conclusion turned on showing that its capped
    games were searched to COMPLETION (zero exhaustion) rather than truncated.
    """

    def test_starved_budget_reports_exhausted(self):
        game = MonsterChessGame(BARE_KING_TWO_ROOKS)
        move, exhausted = dg._finisher_probe(game, depth=4, node_budget=1)
        self.assertIsNone(move)
        self.assertTrue(exhausted)

    def test_completed_search_with_no_win_reports_not_exhausted(self):
        game = MonsterChessGame(LONE_BLACK_KING)
        move, exhausted = dg._finisher_probe(game, depth=4,
                                             node_budget=2_000_000)
        self.assertIsNone(move)
        self.assertFalse(exhausted,
                         "a completed search proves absence within the horizon")

    def test_a_hit_is_never_reported_as_exhausted(self):
        game = MonsterChessGame(IMMEDIATE_CAPTURE)
        move, exhausted = dg._finisher_probe(game, depth=4,
                                             node_budget=2_000_000)
        self.assertIsNotNone(move)
        self.assertFalse(exhausted)


def _finisher_applicable(game):
    from benchmark import FINISHER_WHITE_MATERIAL_MAX
    return dg._finisher_applicable(game, FINISHER_WHITE_MATERIAL_MAX)


if __name__ == "__main__":
    unittest.main()


class TestScriptedOracleIsDropped(unittest.TestCase):
    """The oracle is off by default (owner, 2026-08-16).

    Measured over 24 generated games: it drove Black in 12 and converted 2,
    drew 5, and LOST 5 against a bare White king -- 17% against its authored
    deck's 92% -- stamping every move at policy 1.0 including the losses. Of
    387 policy-1.0 records in that batch, 241 came from games that never
    converted. A 120-game A/B without it ran 2.1x faster with Black 0.317 ->
    0.375.
    """

    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(dg._scripted_mate_enabled())

    def test_opt_in_restores_it_for_reproductions(self):
        with mock.patch.dict(os.environ, {"MONSTER_SCRIPTED_MATE": "1"},
                             clear=True):
            self.assertTrue(dg._scripted_mate_enabled())
