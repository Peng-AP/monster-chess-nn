"""The two things in the reviewer that would silently mislead if wrong.

A move-quality reviewer is only useful if a negative number means "this move
hurt the player who made it" and the move named is the move actually played.
Both are easy to get backwards and neither fails loudly.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

_spec = importlib.util.spec_from_file_location(
    "review_game", os.path.join(ROOT, "tools", "review_game.py"))
review_game = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review_game)


START = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


class DescribeMove(unittest.TestCase):
    def test_names_a_single_black_move_in_san(self):
        before = "rnbqkbnr/pppppppp/8/8/4P3/8/2PPKP2/8 b kq - 1 1"
        after = "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/2PPKP2/8 w kq - 0 2"
        self.assertEqual(review_game.describe_move(before, after), "e6")

    def test_names_a_capture_by_its_destination_piece(self):
        """A from/to pair cannot distinguish a capture from a quiet move.

        Matching on the resulting board catches it: the piece standing on the
        destination afterwards is what identifies the move.
        """
        before = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/2PPKP2/8 b kq - 0 2"
        after = "rnbqkbnr/ppp1pppp/8/8/4p3/8/2PPKP2/8 w kq - 0 3"
        self.assertEqual(review_game.describe_move(before, after), "dxe4")

    def test_a_white_double_move_falls_back_to_a_square_diff(self):
        """White moves twice per turn, so no single legal move explains it.

        The reviewer must say SOMETHING rather than return None and drop the
        row -- a silently skipped move is worse than an imprecise label.
        """
        after = "rnbqkbnr/pppppppp/8/8/3PP3/8/2P2P2/4K3 b kq - 0 1"
        described = review_game.describe_move(START, after)
        self.assertIsNotNone(described)
        self.assertTrue(described.startswith("("))
        for square in ("d2", "d4", "e4"):
            self.assertIn(square, described)

    def test_returns_none_when_nothing_changed(self):
        self.assertIsNone(review_game.describe_move(START, START))


class LoadRecords(unittest.TestCase):
    def test_skips_blank_lines(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                             delete=False, encoding="utf-8")
        with handle:
            handle.write(json.dumps({"fen": START}) + "\n\n")
            handle.write(json.dumps({"fen": START}) + "\n")
        try:
            self.assertEqual(len(review_game.load_records(handle.name)), 2)
        finally:
            os.unlink(handle.name)


class SwingConvention(unittest.TestCase):
    """The sign must mean the same thing for both colours.

    `search` returns a side-to-move value, so the value after the move belongs
    to the OPPONENT and has to be negated before it is compared. Without that
    negation every good move reads as a blunder and vice versa, and the output
    still looks entirely plausible.
    """

    def test_the_reviewer_negates_the_post_move_value(self):
        source = open(os.path.join(ROOT, "tools", "review_game.py"),
                      encoding="utf-8").read()
        self.assertIn("_reply, after = search(after_fen)", source)
        self.assertIn("after = -after", source)
        self.assertIn('"swing": round(float(after - before), 4)', source)


if __name__ == "__main__":
    unittest.main()
