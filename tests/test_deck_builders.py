"""The deck builders' selection rules.

Decks decide what every downstream measurement is *about*. If the
promotion-defense filter admitted positions with no capture available, M2's
"capture rate" would be measuring nothing; if the cliff filter admitted
positions outside the pawn phase, D3 would generate self-play from the wrong
phase entirely. Neither failure announces itself -- both produce a full deck
and a clean-looking number.

Process note SS12: distrust any rate that has not been tested against the hard
case. These are the hard cases.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import make_cliff_deck as cliff  # noqa: E402
import make_promotion_defense_deck as pdd  # noqa: E402


class TestPromotionDefenseSelection(unittest.TestCase):
    """Black to move, a White pawn one square from queening, capture available."""

    # The real position from HANDOFF SS4.4: White pawn on b7, Bxb7 available.
    ANCHOR = "r1bqkb1r/pPpp1ppp/5n2/8/4PP2/4K3/8/8 b kq - 0 5"

    def test_the_anchor_position_qualifies(self):
        got = pdd.qualifies(self.ANCHOR)
        self.assertIsNotNone(got)
        _game, targets, caps = got
        import chess
        self.assertEqual({chess.square_name(s) for s in targets}, {"b7"})
        self.assertIn("c8b7", {m.uci() for m in caps})

    def test_white_to_move_is_rejected(self):
        # Same structure, White to move: not a Black defense problem.
        fen = self.ANCHOR.replace(" b ", " w ")
        self.assertIsNone(pdd.qualifies(fen))

    def test_pawn_not_on_the_seventh_is_rejected(self):
        # Pawn on b6 instead of b7 -- one move further from queening.
        fen = "r1bqkb1r/p1pp1ppp/1P3n2/8/4PP2/4K3/8/8 b kq - 0 5"
        self.assertIsNone(pdd.qualifies(fen))

    def test_promoting_pawn_with_no_capture_available_is_rejected(self):
        # White pawn on a7, and no Black piece attacks a7.
        fen = "4kb1r/P1pp1ppp/5n2/8/4PP2/4K3/8/8 b k - 0 5"
        got = pdd.qualifies(fen)
        if got is not None:
            _g, _t, caps = got
            self.fail(f"expected rejection, got captures {[m.uci() for m in caps]}")

    def test_seventh_rank_constant_is_the_queening_rank(self):
        import chess
        self.assertEqual(chess.square_rank(chess.B7), pdd.SEVENTH)


class TestCliffSelection(unittest.TestCase):
    def test_white_pawn_counting_ignores_black_pawns(self):
        # Black pawns are lowercase; only White's count toward the phase.
        self.assertEqual(cliff.white_pawns(
            "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"), 4)
        self.assertEqual(cliff.white_pawns(
            "rnbqkbnr/pppppppp/8/8/8/8/8/4K3 b kq - 0 1"), 0)

    def test_black_to_move_reads_the_side_field(self):
        self.assertTrue(cliff.black_to_move("8/8/8/8/8/8/8/4K3 b - - 0 1"))
        self.assertFalse(cliff.black_to_move("8/8/8/8/8/8/8/4K3 w - - 0 1"))

    def test_dedup_ignores_clocks_but_not_position(self):
        a = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 b kq - 0 1"
        b = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 b kq - 9 44"   # same, later
        c = "rnbqkbnr/pppppppp/8/8/8/8/2PPP3/4K3 b kq - 0 1"     # a pawn fewer
        self.assertEqual(cliff.dedup_key(a), cliff.dedup_key(b))
        self.assertNotEqual(cliff.dedup_key(a), cliff.dedup_key(c))

    def test_dedup_separates_the_same_placement_by_side_to_move(self):
        w = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"
        b = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 b kq - 0 1"
        self.assertNotEqual(cliff.dedup_key(w), cliff.dedup_key(b))


class TestCliffHarvestFilters(unittest.TestCase):
    def setUp(self):
        import json
        import shutil
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.json = json

    def write(self, name, fens, result):
        recs = [{"fen": f, "policy": {}, "game_result": result} for f in fens]
        path = self.tmp / name
        with open(path, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(self.json.dumps(r) + "\n")

    def harvest(self, **kw):
        opts = dict(min_white_pawns=3, require_black_win=True, offset_from_end=0)
        opts.update(kw)
        return list(cliff.harvest(str(self.tmp), **opts))

    def test_white_won_games_are_excluded_when_required(self):
        pawn_fen = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 b kq - 0 1"
        self.write("white_won.jsonl", [pawn_fen], result=1)
        self.assertEqual(self.harvest(), [])
        self.assertEqual(len(self.harvest(require_black_win=False)), 1)

    def test_positions_below_the_pawn_threshold_are_excluded(self):
        two_pawns = "rnbqkbnr/pppppppp/8/8/8/8/3PP3/4K3 b kq - 0 1"
        four_pawns = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 b kq - 0 1"
        self.write("g.jsonl", [two_pawns, four_pawns], result=-1)
        got = self.harvest()
        self.assertEqual([e["white_pawns"] for e in got], [4])

    def test_white_to_move_positions_are_excluded(self):
        fen_w = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"
        fen_b = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 b kq - 0 1"
        self.write("g.jsonl", [fen_w, fen_b], result=-1)
        self.assertEqual(len(self.harvest()), 1)

    def test_offset_from_end_drops_the_final_plies(self):
        # A position two moves from a king capture teaches nothing about
        # converting, so the tail of each game is dropped.
        fen = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 b kq - 0 1"
        self.write("g.jsonl", [fen] * 10, result=-1)
        self.assertEqual(len(self.harvest(offset_from_end=0)), 10)
        self.assertEqual(len(self.harvest(offset_from_end=6)), 4)


if __name__ == "__main__":
    unittest.main()
