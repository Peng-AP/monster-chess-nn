"""Imported White turns must emit half-move records, not an (m1,m2) pair.

`policy_dict_to_target` marginalizes a "m1,m2" policy key down to m1, so a
pair record teaches White's first move and silently drops the second — in a
variant defined by the second move. These pin the two-record emission and, more
importantly, that the second record's FEN is the mid-turn state the encoder's
half_pending flag describes.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "tools"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import chess
from config import STARTING_FEN
from encoding import move_to_index, policy_dict_to_target
from import_playstrategy import convert_from_bundle

# 1. d4,Kd2 d5  2. e4,exd5 Nf6 — White's second move is a capture both times,
# so a marginalized pair record would lose real information.
BUNDLE = {
    "res": 1,
    "uci": ["d2d4", "e1d2", "d7d5", "e2e4", "e4d5", "g8f6"],
    "turn": [0, 0, 1, 2, 2, 3],
}


class ImportEmitsHalfMoveRecordsTests(unittest.TestCase):
    def setUp(self):
        self.recs, err = convert_from_bundle(BUNDLE, winner_only=False)
        self.assertIsNone(err)

    def test_white_turn_becomes_two_records(self):
        whites = [r for r in self.recs if r["current_player"] == "white"]
        self.assertEqual(len(whites), 4)                  # 2 turns x 2 halves
        self.assertEqual([r["half"] for r in whites], [0, 1, 0, 1])

    def test_no_policy_key_is_a_pair(self):
        for rec in self.recs:
            for key in rec["policy"]:
                self.assertNotIn(",", key, f"pair key survived: {key!r}")

    def test_second_half_teaches_the_second_move(self):
        whites = [r for r in self.recs if r["current_player"] == "white"]
        self.assertEqual(list(whites[0]["policy"]), ["d2d4"])
        self.assertEqual(list(whites[1]["policy"]), ["e1d2"])
        self.assertEqual(list(whites[2]["policy"]), ["e2e4"])
        self.assertEqual(list(whites[3]["policy"]), ["e4d5"])

    def test_second_half_fen_is_the_mid_turn_state(self):
        whites = [r for r in self.recs if r["current_player"] == "white"]
        # after d2d4, White still to move, king still home
        board = chess.Board(whites[1]["fen"])
        self.assertEqual(board.turn, chess.WHITE)
        self.assertEqual(board.piece_at(chess.D4),
                         chess.Piece(chess.PAWN, chess.WHITE))
        self.assertEqual(board.piece_at(chess.E1),
                         chess.Piece(chess.KING, chess.WHITE))
        self.assertEqual(whites[0]["fen"], STARTING_FEN)

    def test_black_records_unchanged(self):
        blacks = [r for r in self.recs if r["current_player"] == "black"]
        self.assertEqual([list(r["policy"]) for r in blacks],
                         [["d7d5"], ["g8f6"]])
        self.assertNotIn("half", blacks[0])

    def test_target_vector_now_carries_the_second_move(self):
        # The regression this prevents: with a pair record, m2's index got no
        # mass at all.
        whites = [r for r in self.recs if r["current_player"] == "white"]
        target = policy_dict_to_target(whites[1]["policy"], True)
        self.assertAlmostEqual(
            float(target[move_to_index(chess.Move.from_uci("e1d2"))]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
