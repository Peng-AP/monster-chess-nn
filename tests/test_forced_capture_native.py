"""The native forced-capture solver must agree with the Python one exactly.

A faster solver that disagrees is worse than no solver: this search is the only
component that returns *proofs*, and every conclusion drawn from it (REPORT.md
sections 30 and 34) assumes "no win found" means searched-out, not truncated.

Measured 2026-08-16: 145x over the Python solver at depth 4 with zero
disagreements on 24 real capped-game positions, which moved depth 5 from
~23 CPU-minutes per position to 1.78s and made depth 6 reachable.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "native"))

import forced_capture as fc  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

try:
    import monster_native as mn
except Exception:                                    # pragma: no cover
    mn = None

# Black rook on e8, White king on e1, open file: the capture is available now.
IMMEDIATE = "4r1k1/8/8/8/8/8/8/4K3 b - - 0 1"
# Won, but a ladder mate -- no capture forced inside a short horizon.
TWO_ROOKS = "r5kr/8/8/8/8/8/8/4K3 b - - 0 1"
# Black has nothing to capture with.
LONE_KING = "6k1/8/8/8/8/8/4P3/4K3 b - - 0 1"


@unittest.skipIf(mn is None, "native module not built")
class TestNativeSolverMatchesPython(unittest.TestCase):
    def _both(self, fen, depth=4, budget=6_000_000):
        game = MonsterChessGame(fen=fen)
        py = fc.try_forced_capture_move(
            game, max_black_moves=depth, node_budget=budget)
        rs = mn.forced_capture_move(fen, depth, budget)
        return py, rs

    def test_immediate_capture_agrees(self):
        py, rs = self._both(IMMEDIATE)
        self.assertIsNotNone(py[0])
        self.assertIsNotNone(rs[0])
        self.assertEqual(py[1], rs[1])          # same depth-to-win
        self.assertFalse(py[2] or rs[2])

    def test_no_forced_win_agrees(self):
        py, rs = self._both(LONE_KING)
        self.assertIsNone(py[0])
        self.assertIsNone(rs[0])
        self.assertFalse(py[2] or rs[2],
                         "a completed search proves absence within the horizon")

    def test_ladder_mate_is_not_a_short_forced_capture(self):
        py, rs = self._both(TWO_ROOKS, depth=3)
        self.assertIsNone(py[0])
        self.assertIsNone(rs[0])

    def test_returned_move_is_legal_and_ends_the_game(self):
        import chess
        rs = mn.forced_capture_move(IMMEDIATE, 4, 6_000_000)
        game = MonsterChessGame(fen=IMMEDIATE)
        move = chess.Move.from_uci(rs[0])
        self.assertIn(move, game.get_legal_actions())
        game.apply_search_action(move)
        self.assertIsNone(game.board.king(chess.WHITE))


@unittest.skipIf(mn is None, "native module not built")
class TestNativeSolverContracts(unittest.TestCase):
    def test_starved_budget_reports_exhausted_not_no_win(self):
        """'No answer' must never read as 'no win'."""
        move, depth, exhausted = mn.forced_capture_move(TWO_ROOKS, 4, 1)
        self.assertIsNone(move)
        self.assertIsNone(depth)
        self.assertTrue(exhausted)

    def test_completed_search_is_not_flagged_exhausted(self):
        move, _depth, exhausted = mn.forced_capture_move(LONE_KING, 4, 6_000_000)
        self.assertIsNone(move)
        self.assertFalse(exhausted)

    def test_white_to_move_is_rejected(self):
        with self.assertRaises(ValueError):
            mn.forced_capture_move(
                "4r1k1/8/8/8/8/8/8/4K3 w - - 0 1", 4, 1000)

    def test_dedup_fires_on_a_bare_king(self):
        """White's two half-moves transpose heavily once it is a bare king."""
        _d, nodes, _hits, dedup, _ex = mn.forced_capture_stats(
            TWO_ROOKS, 4, 6_000_000)
        self.assertGreater(nodes, 0)
        self.assertGreater(dedup, 0,
                           "transposing White pairs should be collapsed")


class TestPythonSolverOptimisationsAreExact(unittest.TestCase):
    """Memoisation must not change an answer, and must not cache a cut search."""

    def test_memo_does_not_change_results(self):
        for fen in (IMMEDIATE, TWO_ROOKS, LONE_KING):
            game = MonsterChessGame(fen=fen)
            a = fc.try_forced_capture_move(game, max_black_moves=3,
                                           node_budget=6_000_000)
            b = fc.try_forced_capture_move(game, max_black_moves=3,
                                           node_budget=6_000_000)
            self.assertEqual(
                (a[0].uci() if a[0] else None, a[1], a[2]),
                (b[0].uci() if b[0] else None, b[1], b[2]))

    def test_budget_cut_is_not_cached_as_a_result(self):
        game = MonsterChessGame(fen=TWO_ROOKS)
        starved = fc.try_forced_capture_move(game, max_black_moves=4,
                                             node_budget=1)
        self.assertTrue(starved[2])
        full = fc.try_forced_capture_move(game, max_black_moves=4,
                                          node_budget=6_000_000)
        self.assertFalse(full[2],
                         "a fresh full search must not inherit a cut result")


if __name__ == "__main__":
    unittest.main()
