"""The search knows the driver's repetition rule (owner, 2026-10-07).

The rule lives in `RepetitionTracker`; the tracker hands its counts to the
game, `NativeMCTS` hands them to the tree, and the tree scores a position's
Nth occurrence as a draw: the side that is ahead steers away from it, the side
that is behind may steer into it. CPU only: heuristic leaf values, no network.
"""
import os
import sys

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src")]

import native_mcts  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from repetition import RepetitionTracker, position_key  # noqa: E402

mn = native_mcts.mn

# Black: king e8, queen d8, rook a8. White: bare king h1. Black is far ahead.
BLACK_TO_MOVE = "r2qk3/8/8/8/8/8/8/7K b - - 0 40"
WHITE_TO_MOVE = "r2qk3/8/8/8/8/8/8/7K w - - 0 40"


def after(game, *ucis):
    g = game.clone()
    for uci in ucis:
        g.apply_search_action(chess.Move.from_uci(uci))
    return g


def tracked(fen, seen_twice):
    """A game at `fen` whose tracker has already seen `seen_twice` positions twice."""
    game = MonsterChessGame(fen)
    tracker = RepetitionTracker(threshold=3, enabled=True)
    tracker.record(game, 0)
    for line in seen_twice:
        tracker.counts[position_key(after(game, *line))] = 2
    return game, tracker


def search(game, sims=400):
    engine = native_mcts.NativeMCTS(num_simulations=sims, eval_fn=None, allow_early_stop=False)
    trees = []
    build = engine._tree_for
    engine._tree_for = lambda state: trees.append(build(state)) or trees[-1]
    move, _probs, _value = engine.get_best_action(game, temperature=0.0)
    return move, trees[-1]


def test_tracker_hands_its_counts_to_the_game():
    game, tracker = tracked(BLACK_TO_MOVE, [("a8a2",)])
    assert game.repetition_counts is tracker.counts and game.repetition_threshold == 3
    counts, threshold = native_mcts.NativeMCTS._repetition(game)
    assert threshold == 3 and all(len(key.split()) == 4 for key, _ in counts)


def test_the_side_far_ahead_does_not_repeat():
    # A third ...Ra2 would draw; every other Black move keeps the game going.
    game, _ = tracked(BLACK_TO_MOVE, [("a8a2",)])
    move, tree = search(game)
    assert tree.root_repetition_draws() == ["a8a2"]
    assert move.uci() != "a8a2"


def test_the_side_far_behind_takes_the_repetition():
    # White's bare king: returning to this placement with Black to move is the
    # third occurrence, i.e. a draw, which beats every losing alternative.
    game, tracker = tracked(WHITE_TO_MOVE, [("h1g1", "g1h1")])
    first, _ = search(game)
    game.apply_search_action(first)
    second, tree = search(game)
    if first.uci() == "h1g1":
        assert "g1h1" in tree.root_repetition_draws()
    game.apply_search_action(second)
    assert tracker.record(game, 2)


def test_off_switch_and_untracked_games_search_as_before(monkeypatch):
    game, _ = tracked(BLACK_TO_MOVE, [("a8a2",)])
    monkeypatch.setenv(native_mcts.REPETITION_SEARCH_OFF_ENV, "1")
    assert native_mcts.NativeMCTS._repetition(game) == (None, 0)
    _move, tree = search(game, sims=50)
    assert tree.repetition_threshold() == 0 and tree.root_repetition_draws() == []
    monkeypatch.delenv(native_mcts.REPETITION_SEARCH_OFF_ENV)
    assert native_mcts.NativeMCTS._repetition(MonsterChessGame(BLACK_TO_MOVE)) == (None, 0)
