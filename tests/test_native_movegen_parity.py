"""The native movegen must reproduce python-chess exactly (DIRECTIVE E1).

Skipped when the crate has not been built, so the suite still runs on a box
without a Rust toolchain. When it is built these are binding: the whole value
of the native core is that it plays the same game, and the quirks below are the
ones a from-scratch bitboard generator gets *wrong* in the most plausible way.

Full differential lives in `tools/movegen_parity.py`; 248,272 positions, zero
mismatches as of 2026-08-03. This file keeps the specific traps in CI.
"""
import sys
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "native"))

try:
    import monster_native as mn
except ImportError:  # crate not built on this machine
    mn = None

from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def reference(fen):
    return {m.uci() for m in chess.Board(fen).pseudo_legal_moves}


@unittest.skipIf(mn is None, "native crate not built")
class TestMovegenMatchesPythonChess(unittest.TestCase):
    def assert_parity(self, fen, note=""):
        self.assertEqual(set(mn.pseudo_legal_uci(fen)), reference(fen), note or fen)

    def test_monster_start_position(self):
        self.assert_parity(START_FEN)

    def test_castling_is_refused_while_in_check(self):
        # The trap: this is "pseudo-legal" generation, so a natural bitboard
        # port emits castling here. python-chess does not, and the Python
        # engine inherits that.
        self.assert_parity("4r3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "castling while in check")

    def test_castling_refused_only_on_the_attacked_side(self):
        self.assert_parity("5r2/8/8/8/8/8/8/R3K2R w KQ - 0 1", "f1 attacked")

    def test_castling_available_when_clear(self):
        self.assert_parity("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "clear castling")

    def test_en_passant_only_when_the_fen_carries_it(self):
        self.assert_parity("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "ep declared")
        self.assert_parity("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 2", "ep absent")

    def test_king_capture_is_an_ordinary_move(self):
        self.assert_parity("k7/8/8/8/8/8/8/R3K3 w - - 0 1", "rook may take the king")

    def test_promotions_enumerate_all_four_pieces(self):
        self.assert_parity("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "promotion")
        self.assert_parity("n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "capture-promotion")

    def test_black_to_move_positions(self):
        self.assert_parity("rnbqkbnr/pppppppp/8/8/3PP3/8/2P2P2/4K3 b kq - 0 1")


@unittest.skipIf(mn is None, "native crate not built")
class TestFenRoundTrip(unittest.TestCase):
    def test_round_trip_is_byte_identical(self):
        for fen in (
            START_FEN,
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 5 27",
            "8/K7/8/8/8/8/8/8 b - - 0 15",
        ):
            self.assertEqual(mn.fen_roundtrip(fen), fen)


@unittest.skipIf(mn is None, "native crate not built")
class TestRandomWalkParity(unittest.TestCase):
    def test_a_short_random_walk_stays_in_parity(self):
        import random
        rng = random.Random(20260803)
        checked = 0
        for _ in range(20):
            game = MonsterChessGame(START_FEN)
            for _ in range(40):
                if game.is_terminal():
                    break
                self.assert_parity_fen(game.fen())
                checked += 1
                actions = game.get_search_actions()
                if not actions:
                    break
                game.apply_search_action(rng.choice(actions))
        self.assertGreater(checked, 200)

    def assert_parity_fen(self, fen):
        self.assertEqual(set(mn.pseudo_legal_uci(fen)), reference(fen), fen)



@unittest.skipIf(mn is None, "native crate not built")
class TestBugsFoundByTheDifferential(unittest.TestCase):
    """Regression pins for the two defects the E1 differential caught.

    Both were mine, both were invisible to ordinary chess intuition, and both
    would have corrupted generated data rather than crashing.
    """

    def test_ep_capturer_must_stand_on_the_capturing_rank(self):
        # After White's c2c4 the ep square is c3, and Monster Chess forces
        # board.turn back to WHITE between halves -- so a naive generator
        # offers White capturing its OWN ep square with the d2 pawn.
        # python-chess constrains capturers to rank 5 (White) / rank 4 (Black).
        fen = "rnbqkbnr/pppppppp/8/8/2P5/8/3PPP2/4K3 w kq c3 0 1"
        self.assertNotIn("d2c3", set(mn.pseudo_legal_uci(fen)))
        self.assertEqual(set(mn.pseudo_legal_uci(fen)), reference(fen))

    def test_castling_rights_die_when_the_king_is_captured(self):
        # python-chess voids a side's rights unless its king is home, so a
        # White piece landing on e8 drops "kq" from the FEN. A port that only
        # cleared rook squares kept them.
        fen = "r1bqkbnr/pp1Ppppp/8/2p1n3/2P5/3P1P2/8/3K4 w kq - 0 5"
        got = mn.push_uci(fen, "d7e8q")
        board = chess.Board(fen)
        board.push(chess.Move.from_uci("d7e8q"))
        self.assertEqual(got, board.fen())
        self.assertIn(" - ", got)


@unittest.skipIf(mn is None, "native crate not built")
class TestMonsterActionApis(unittest.TestCase):
    def setUp(self):
        self.fen = START_FEN
        self.game = MonsterChessGame(START_FEN)

    def test_white_pairs_match_the_python_engine(self):
        py = {f"{m1.uci()},{m2.uci()}"
              for m1, m2 in self.game._get_white_actions(truncate_wins=False)}
        self.assertEqual(set(mn.white_actions_uci(self.fen, False)), py)

    def test_white_first_half_has_no_safety_filter(self):
        py = {m.uci() for m in self.game._white_single_moves()}
        self.assertEqual(set(mn.white_single_moves_uci(self.fen)), py)

    def test_black_moves_match_the_python_engine(self):
        fen = "rnbqkbnr/pppppppp/8/8/3PP3/8/2P2P2/4K3 b kq - 0 1"
        game = MonsterChessGame(fen)
        py = {m.uci() for m in game._get_black_actions(truncate_wins=False)}
        self.assertEqual(set(mn.black_actions_uci(fen, False)), py)

    def test_a_king_capture_truncates_to_one_action(self):
        # Black rook a1 sees the White king on e1 along the rank.
        fen = "k7/8/8/8/8/8/8/r3K3 b - - 0 1"
        truncated = mn.black_actions_uci(fen, True)
        self.assertEqual(len(truncated), 1)
        self.assertEqual(truncated[0], "a1e1")
        # and the complete oracle lists that winner first, then everything else
        complete = mn.black_actions_uci(fen, False)
        self.assertEqual(complete[0], "a1e1")
        self.assertGreater(len(complete), 1)

if __name__ == "__main__":
    unittest.main()
