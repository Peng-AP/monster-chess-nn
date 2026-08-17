"""The native encoder must match `encoding.fen_to_tensor` byte-for-byte (E2).

Byte-equality rather than tolerance: the tensor *is* the network's view of a
position, so one differing element is a different input. Measured 2026-08-03
over 98,272 positions in each layout with zero inequalities.

Skipped when the crate is not built.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "native"))

try:
    import monster_native as mn
except ImportError:
    mn = None

import config  # noqa: E402
from encoding import (fen_to_tensor, move_to_index,
                      promotion_move_to_index)  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
import chess  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


@unittest.skipIf(mn is None, "native crate not built")
class TestEncodingParity(unittest.TestCase):
    def assert_same(self, fen, is_white=True, pending=False, channels=17):
        want = fen_to_tensor(fen, is_white, pending, input_channels=channels)
        got = np.asarray(mn.encode_fen(fen, is_white, pending, channels),
                         dtype=np.float32).reshape(8, 8, channels)
        self.assertTrue(np.array_equal(want, got),
                        f"{fen} ch={channels} white={is_white} pending={pending}")

    def test_start_position_both_layouts(self):
        for channels in (15, 17):
            self.assert_same(START_FEN, channels=channels)

    def test_turn_and_half_move_channels(self):
        # Channel 12 is +-1 and channel 13 marks White's SECOND half -- the
        # network routes the two halves differently on the strength of it.
        for is_white in (True, False):
            for pending in (True, False):
                self.assert_same(START_FEN, is_white, pending)

    def test_pawn_progress_channels_for_both_colours(self):
        self.assert_same("4k3/1p6/8/8/8/8/1P6/4K3 w - - 0 1")

    def test_promotion_rank_pawns(self):
        self.assert_same("4k3/1P6/8/8/8/8/1p6/4K3 w - - 0 1")

    def test_legacy_layout_keeps_white_only_channel_14(self):
        fen = "4k3/1p6/8/8/8/8/1P6/4K3 w - - 0 1"
        legacy = np.asarray(mn.encode_fen(fen, True, False, 15),
                            dtype=np.float32).reshape(8, 8, 15)
        want = fen_to_tensor(fen, True, False, input_channels=15)
        self.assertTrue(np.array_equal(legacy, want))
        # and it must differ from the 17ch layout's channel 14 (rank coord)
        current = np.asarray(mn.encode_fen(fen, True, False, 17),
                             dtype=np.float32).reshape(8, 8, 17)
        self.assertFalse(np.array_equal(legacy[:, :, 14], current[:, :, 14]))

    def test_unsupported_channel_count_is_rejected(self):
        with self.assertRaises(ValueError):
            mn.encode_fen(START_FEN, True, False, 16)

    def test_random_walk_both_layouts(self):
        import random
        rng = random.Random(20260803)
        checked = 0
        for channels in (15, 17):
            for _ in range(12):
                game = MonsterChessGame(START_FEN)
                for _ in range(40):
                    if game.is_terminal():
                        break
                    self.assert_same(game.fen(), game.is_white_turn,
                                     game.white_half_pending, channels)
                    checked += 1
                    actions = game.get_search_actions()
                    if not actions:
                        break
                    game.apply_search_action(rng.choice(actions))
        self.assertGreater(checked, 400)


@unittest.skipIf(mn is None, "native crate not built")
class TestPolicyIndex(unittest.TestCase):
    def test_matches_python_move_to_index(self):
        for uci in ("e2e4", "a1h8", "h7h8q", "b8c6", "a2a1"):
            self.assertEqual(mn.move_to_index(uci),
                             move_to_index(chess.Move.from_uci(uci)))

    def test_index_range(self):
        self.assertEqual(mn.move_to_index("a1a1"), 0)
        self.assertEqual(mn.move_to_index("h8h8"), config.POLICY_SIZE - 1)

    def test_matches_distinct_promotion_indices(self):
        for uci in ("a7a8q", "c7d8n", "h2g1r", "e2e1b"):
            self.assertEqual(
                mn.promotion_move_to_index(uci),
                promotion_move_to_index(chess.Move.from_uci(uci)),
            )


if __name__ == "__main__":
    unittest.main()
