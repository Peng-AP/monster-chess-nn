"""Pins for `draw_anatomy`, including the mistake it used to make.

The tool originally sorted drawn games by White's remaining pawns and labelled
the pawnless ones "structural, unwinnable". That was an assumption written into
the output and then read back as a finding. 150 games refuted it: White scores
0.6141 from pawnless positions against 0.6293 with pawns, and 34 of its 62 wins
ended with zero pawns -- the double-moving king really does hunt, exactly as
`evaluation.py` always said.

Outcomes separate on TIME, not material. So the material parse is still pinned
(it is reported as description), and the framing is pinned too, so the
discredited verdict cannot quietly return.
"""
import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

_spec = importlib.util.spec_from_file_location(
    "draw_anatomy", os.path.join(ROOT, "tools", "draw_anatomy.py"))
draw_anatomy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(draw_anatomy)


class MaterialParse(unittest.TestCase):
    def test_start_position_is_four_white_pawns_and_a_full_black_army(self):
        white_pawns, black_pieces = draw_anatomy._material(
            "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1")
        self.assertEqual(white_pawns, 4)
        self.assertEqual(black_pieces, 16)

    def test_a_bare_white_king_counts_zero_pawns(self):
        """The case the whole split turns on.

        White wins only by capturing the black king; with no pawns it is a lone
        double-moving king against an army, and that draw is structural rather
        than a missed conversion.
        """
        white_pawns, black_pieces = draw_anatomy._material(
            "rnbqkbnr/pppppppp/8/8/8/8/8/4K3 b kq - 0 40")
        self.assertEqual(white_pawns, 0)
        self.assertEqual(black_pieces, 16)

    def test_case_matters_black_pawns_are_not_white_pawns(self):
        """`P` and `p` are different sides.

        A case-insensitive count would report every position as a conversion
        failure, because Black almost always still has pawns.
        """
        white_pawns, _black = draw_anatomy._material(
            "4k3/pppppppp/8/8/8/8/8/4K3 w - - 0 1")
        self.assertEqual(white_pawns, 0)

    def test_black_count_includes_the_king_and_every_piece_type(self):
        _white, black_pieces = draw_anatomy._material(
            "4k3/8/8/8/8/8/2PPPP2/4K3 w - - 0 1")
        self.assertEqual(black_pieces, 1)

    def test_only_the_placement_field_is_read(self):
        """A FEN's later fields carry letters too.

        Castling rights spell `KQkq`; counting across the whole string would
        add phantom material that changes the verdict.
        """
        white_pawns, black_pieces = draw_anatomy._material(
            "4k3/8/8/8/8/8/2PPPP2/4K3 w KQkq - 0 1")
        self.assertEqual(white_pawns, 4)
        self.assertEqual(black_pieces, 1)


class ProbeContract(unittest.TestCase):
    def test_every_terminal_cause_is_named(self):
        """Silence must not be a possible outcome.

        A game that ends any way other than repetition/terminal/cap still has
        to be labelled, or draws quietly vanish from the denominator.
        """
        source = open(os.path.join(ROOT, "tools", "draw_anatomy.py"),
                      encoding="utf-8").read()
        for reason in ('"repetition"', '"terminal"', '"ply_cap"', '"no_move"'):
            self.assertIn(reason, source)

    def test_material_is_never_reported_as_a_verdict(self):
        """The discredited labels must not come back.

        "structural, unwinnable" was wrong: pawnless White scores 0.6141, and
        more than half its wins end with no pawns at all. Material is cheap to
        record and worth printing, but it does not predict the result and the
        tool must not imply that it does.
        """
        source = open(os.path.join(ROOT, "tools", "draw_anatomy.py"),
                      encoding="utf-8").read()
        analysis = source.split('"""', 2)[2]      # skip the module docstring
        for discredited in ("structural, unwinnable", "conversion failures"):
            self.assertNotIn(discredited, analysis)
        self.assertIn("material does not predict the result", analysis)

    def test_outcomes_are_split_by_game_length(self):
        """Time is the axis that separates win from draw.

        Wins average 30 plies and 90% land by ply 50; draws average 82 with the
        same material on both sides. A report that omits length cannot show it.
        """
        source = open(os.path.join(ROOT, "tools", "draw_anatomy.py"),
                      encoding="utf-8").read()
        self.assertIn("outcome by game length", source)
        self.assertIn("wins landing by ply", source)
        self.assertIn('"by_outcome": buckets', source)
        self.assertIn('"wins_landing_by_ply": horizon', source)

    def test_the_refutation_is_printed_on_every_run(self):
        """Both scores, side by side, so the mistake stays visible."""
        source = open(os.path.join(ROOT, "tools", "draw_anatomy.py"),
                      encoding="utf-8").read()
        self.assertIn('"score_when_pawnless"', source)
        self.assertIn('"score_when_pawns_remain"', source)

    def test_the_book_start_state_is_restored(self):
        """A FEN alone cannot say which half of White's turn is pending.

        `MonsterChessGame(fen)` also restarts turn_count at 0, which would hand
        every game extra turns before the cap and lower the draw rate -- the
        exact statistic this tool exists to report.
        """
        source = open(os.path.join(ROOT, "tools", "draw_anatomy.py"),
                      encoding="utf-8").read()
        self.assertIn("game.white_half_pending = bool(half)", source)
        self.assertIn("game.turn_count = int(turn_count)", source)


if __name__ == "__main__":
    unittest.main()
