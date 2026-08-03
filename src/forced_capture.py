"""Forced-king-capture solver — the E0(b) spike (DIRECTIVE 2026-08-03 §3).

The question this answers: in the dominant-but-unfinished endings that make up
the conversion pathology, **was there a forced king capture that MCTS walked
past?**  If yes, a finisher search is validated as the fix and the next
question is only whether it is affordable.  If no, those positions are not won
at reachable depth and the finisher drops down the exploit order.

This is an exact AND/OR search, not a heuristic:

- **Black node (OR)** — Black needs *one* move that wins.
- **White node (AND)** — *every* White double-move must still lose.  White
  capturing Black's king is an immediate refutation, and because
  `get_legal_actions` lists winning captures first, refutations surface on the
  first child in exactly the positions where they exist.

Depth is counted in **Black moves**, so `d=1` means "Black captures the king
right now" and `d=2` means "Black plays a move after which every White reply
loses the king immediately".

Two rules-layer details this must respect, both from `CONTEXT.md` §1:

1. **The turn cap is neutralised.**  `is_terminal()` fires at
   `turn_count >= MAX_GAME_TURNS` and relabels by heuristic sign, so a solver
   run on a near-cap position would see a "terminal draw" instead of searching.
   Every entry point works on a clone with the counter reset — the question is
   whether the position is *won*, not whether the clock is about to run out.
2. **King capture is unconditional and pseudo-legal.**  Black may capture the
   White king while leaving its own king en prise; the capture ends the game
   first.  So "Black can capture now" is exactly "some Black piece attacks the
   White king square", pins included — `board.attackers` is the right
   primitive and is far cheaper than generating moves.
"""
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monster_chess import MonsterChessGame  # noqa: E402


class NodeBudgetExceeded(Exception):
    """Raised when a search exceeds its node budget — reported, never swallowed.

    A budget hit is *not* "no forced win": conflating the two would silently
    understate the finisher's value, which is the exact number this spike
    exists to measure.
    """


def _fresh(state):
    """Clone with the move-limit clock reset (see module docstring, point 1)."""
    g = state.clone()
    g.turn_count = 0
    g._terminal = False
    g._result = None
    return g


def black_can_capture_now(state):
    """True if some Black piece attacks the White king square."""
    wk = state.board.king(chess.WHITE)
    if wk is None:
        return True  # already captured
    return bool(state.board.attackers(chess.BLACK, wk))


def white_can_capture_now(state):
    bk = state.board.king(chess.BLACK)
    if bk is None:
        return True
    return bool(state.board.attackers(chess.WHITE, bk))


class _Search:
    def __init__(self, node_budget):
        self.node_budget = node_budget
        self.nodes = 0

    def _tick(self):
        self.nodes += 1
        if self.node_budget is not None and self.nodes > self.node_budget:
            raise NodeBudgetExceeded()

    def black_wins_within(self, state, d):
        """Black to move: can Black force a king capture within `d` Black moves?"""
        if d <= 0:
            return False
        if black_can_capture_now(state):
            return True
        if d == 1:
            return False
        for move in state.get_legal_actions():
            self._tick()
            child = state.clone()
            child.apply_action(move)
            if child.board.king(chess.WHITE) is None:
                return True
            if self.white_all_lose(child, d - 1):
                return True
        return False

    def white_all_lose(self, state, d):
        """White to move: does every White turn still lose within `d` Black moves?"""
        actions = state.get_legal_actions()
        if not actions:
            # No pseudo-legal White move. The engine does not score this as a
            # terminal state (stalemate is deliberately not detected), so this
            # is not a forced capture — refuse to claim a win the engine's own
            # rules would not award.
            return False
        for action in actions:
            self._tick()
            child = state.clone()
            child.apply_action(action)
            if child.board.king(chess.BLACK) is None:
                return False  # White refutes by capturing first
            if not self.black_wins_within(child, d):
                return False
        return True


def forced_capture_depth(state, max_black_moves=2, node_budget=400_000):
    """Fewest Black moves that force a king capture, or None.

    Iterative deepening, so the returned depth is minimal. Raises
    `NodeBudgetExceeded` rather than returning None when the budget runs out.
    """
    root = _fresh(state)
    if root.is_white_turn:
        raise ValueError("forced_capture_depth expects a Black-to-move position")
    search = _Search(node_budget)
    for d in range(1, max_black_moves + 1):
        if search.black_wins_within(root, d):
            return d
    return None


def try_forced_capture_depth(state, max_black_moves=2, node_budget=400_000):
    """`forced_capture_depth`, but returns ("budget", nodes) instead of raising.

    Returns (depth_or_None, nodes_used, exhausted) where `exhausted` is True
    when the budget stopped the search — the caller must not read that as
    "no forced win".
    """
    root = _fresh(state)
    if root.is_white_turn:
        raise ValueError("expects a Black-to-move position")
    search = _Search(node_budget)
    try:
        for d in range(1, max_black_moves + 1):
            if search.black_wins_within(root, d):
                return d, search.nodes, False
    except NodeBudgetExceeded:
        return None, search.nodes, True
    return None, search.nodes, False
