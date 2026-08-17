"""Selection layer: the owner's overrides, ported verbatim (DIRECTIVE §0.3).

These are product decisions, not implementation detail:

* **King safety** (owner 2026-07-12, engine-wide) — never hand the opponent an
  immediate king capture when a searched alternative survives. Search normally
  avoids this via the eval clamps, but at value saturation every move reads
  ~-0.98 and selection degenerates to noise.
* **White first half** (owner 2026-07-17) — an m1 must keep at least one
  king-safe completion when a searched alternative does. The m2-level override
  cannot repair an m1 blunder: once the first half walks into a pocket, every
  second half hangs and it sees "all moves lose, forced".
* **Oscillation** (owner 2026-07-17) — *penalise* exact reversals, never forbid
  them; sometimes going back is best. The discount only decides ties, and the
  raw visit distribution is still what training sees.

Fixtures were found by scanning real games for positions where the property
actually varies across moves — a position where every move hangs, or none does,
cannot test an override that only fires on the difference.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "native"))

try:
    import monster_native as mn
except ImportError:
    mn = None

from monster_chess import MonsterChessGame  # noqa: E402
from mcts import _hangs_king, _m1_dooms_king  # noqa: E402

# Black to move; f7g6 and f7e6 hand White an immediate king capture, the rest
# do not.
HANG_FEN = "rn1q1bnr/2pppk1p/2b5/ppP5/3P1P2/1K4p1/8/8 b - - 2 10"
HANGING, SAFE = "f7g6", "g8h6"

# White's first half; h5h4 and c5c6 leave every second half hanging.
DOOM_FEN = "rn2kbr1/p3p1p1/P6q/2Pp1p1K/4nP2/8/8/8 w q - 1 14"
DOOMED, FINE = "h5h4", "h5h6"


def tree_with_children(fen, actions, visits, pending=False):
    """A root expanded to exactly `actions`, with visit counts imposed."""
    tree = mn.Tree(fen, pending, 0)
    for action, count in zip(actions, visits):
        idx = tree.add_child(0, action, 1.0 / len(actions))
        tree.set_stats(idx, count, 0.0)
    return tree


@unittest.skipIf(mn is None, "native crate not built")
class TestFixturesStillBehave(unittest.TestCase):
    """If the fixtures stop exercising the difference the tests are vacuous."""

    def test_hang_fixture(self):
        game = MonsterChessGame(HANG_FEN)
        self.assertTrue(_hangs_king(game, next(
            m for m in game.get_search_actions() if m.uci() == HANGING)))
        self.assertFalse(_hangs_king(game, next(
            m for m in game.get_search_actions() if m.uci() == SAFE)))

    def test_doom_fixture(self):
        game = MonsterChessGame(DOOM_FEN)
        self.assertTrue(_m1_dooms_king(game, next(
            m for m in game.get_search_actions() if m.uci() == DOOMED)))
        self.assertFalse(_m1_dooms_king(game, next(
            m for m in game.get_search_actions() if m.uci() == FINE)))


@unittest.skipIf(mn is None, "native crate not built")
class TestKingSafetyOverride(unittest.TestCase):
    def test_a_hanging_top_choice_is_replaced(self):
        # The hanging move has the most visits, so selection would play it.
        tree = tree_with_children(HANG_FEN, [HANGING, SAFE], [100, 10])
        action, _probs, _v = tree.best_action(temperature=0.0)
        self.assertEqual(action, SAFE)

    def test_a_safe_top_choice_is_left_alone(self):
        tree = tree_with_children(HANG_FEN, [HANGING, SAFE], [10, 100])
        action, _probs, _v = tree.best_action(temperature=0.0)
        self.assertEqual(action, SAFE)

    def test_the_raw_training_target_is_not_rewritten(self):
        # The override changes the move played, never the policy target.
        tree = tree_with_children(HANG_FEN, [HANGING, SAFE], [100, 10])
        action, probs, _v = tree.best_action(temperature=0.0)
        self.assertEqual(action, SAFE)
        self.assertAlmostEqual(dict(probs)[HANGING], 100 / 110)

    def test_when_every_move_hangs_it_stands_down(self):
        tree = tree_with_children(HANG_FEN, [HANGING], [50])
        action, _probs, _v = tree.best_action(temperature=0.0)
        self.assertEqual(action, HANGING)  # genuinely forced


@unittest.skipIf(mn is None, "native crate not built")
class TestWhiteFirstHalfOverride(unittest.TestCase):
    def test_a_doomed_first_half_is_replaced(self):
        tree = tree_with_children(DOOM_FEN, [DOOMED, FINE], [100, 10])
        action, _probs, _v = tree.best_action(temperature=0.0)
        self.assertEqual(action, FINE)

    def test_a_sound_first_half_is_left_alone(self):
        tree = tree_with_children(DOOM_FEN, [DOOMED, FINE], [10, 100])
        action, _probs, _v = tree.best_action(temperature=0.0)
        self.assertEqual(action, FINE)


@unittest.skipIf(mn is None, "native crate not built")
class TestSelectedChildValue(unittest.TestCase):
    def test_value_is_the_selected_childs_q_not_the_root_average(self):
        # The owner's complaint: a proven mate reading ~+0.7 because the root
        # average includes simulations spent refuting losing siblings.
        tree = mn.Tree(HANG_FEN)
        good = tree.add_child(0, SAFE, 0.5)
        other = tree.add_child(0, "g8f6", 0.5)
        tree.set_stats(good, 10, 10.0)    # q = 1.0
        tree.set_stats(other, 30, -30.0)  # q = -1.0, and more visits
        tree.set_stats(0, 40, -20.0)      # root average = -0.5
        _action, _probs, value = tree.best_action(temperature=0.0)
        # 'other' wins on visits but hangs nothing, so it is selected; its Q is
        # what must be reported, not the root's -0.5.
        self.assertAlmostEqual(value, -1.0)

    def test_unvisited_selection_falls_back_to_the_root(self):
        tree = mn.Tree(HANG_FEN)
        tree.add_child(0, SAFE, 1.0)
        tree.set_stats(0, 4, 2.0)  # root q = 0.5
        _action, _probs, value = tree.best_action(temperature=0.0)
        self.assertAlmostEqual(value, 0.5)


@unittest.skipIf(mn is None, "native crate not built")
class TestOscillationPenalty(unittest.TestCase):
    def test_history_offsets_match_the_python_engine(self):
        # The penalty reads offsets -1/-3/-4 (White second half) or -3 (Black).
        # A state rebuilt without history loses the penalty silently, so the
        # offsets themselves are pinned.
        from mcts import _own_previous_moves
        game = MonsterChessGame("rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1")
        applied = []
        for _ in range(6):
            move = game.get_search_actions()[0]
            applied.append(move.uci())
            game.apply_search_action(move)
        native = mn.Game(game.fen(), game.white_half_pending, game.turn_count, applied)
        self.assertEqual(native.own_previous_uci(),
                         [m.uci() for m in _own_previous_moves(game)])

    def test_a_reversal_is_discounted_when_history_carries_it(self):
        # Black to move with a rook that just came h8->h5; playing h5->h8
        # reverses it and must be discounted against an equal-visit sibling.
        # Black's own previous move sits at offset -3, because the push cycle
        # is Black 1 push then White 2 -- a single history entry is White's,
        # not Black's, and would never be reachable.
        fen = "4k3/8/8/7r/8/8/8/4K3 b - - 0 1"
        tied = mn.Tree(fen, False, 4, ["h8h5", "e1f1", "f1e1"])
        rev = tied.add_child(0, "h5h8", 0.5)
        alt = tied.add_child(0, "h5h6", 0.5)
        tied.set_stats(rev, 100, 0.0)
        tied.set_stats(alt, 95, 0.0)
        action, _probs, _v = tied.best_action(temperature=0.0)
        # 100 * 0.9 = 90 < 95, so the near-tie flips to the non-reversal.
        self.assertEqual(action, "h5h6")

    def test_without_history_the_same_position_keeps_the_reversal(self):
        # Same tree, no history supplied: nothing to reverse, so 100 > 95 wins.
        fen = "4k3/8/8/7r/8/8/8/4K3 b - - 0 1"
        tree = mn.Tree(fen, False, 4)
        rev = tree.add_child(0, "h5h8", 0.5)
        alt = tree.add_child(0, "h5h6", 0.5)
        tree.set_stats(rev, 100, 0.0)
        tree.set_stats(alt, 95, 0.0)
        action, _probs, _v = tree.best_action(temperature=0.0)
        self.assertEqual(action, "h5h8")

    def test_penalty_only_decides_ties(self):
        # With a clear visit lead the reversal still wins: 100 * 0.9 = 90 > 50.
        tree = tree_with_children(HANG_FEN, [SAFE, "g8f6"], [100, 50])
        action, _probs, _v = tree.best_action(temperature=0.0)
        self.assertEqual(action, SAFE)


if __name__ == "__main__":
    unittest.main()
