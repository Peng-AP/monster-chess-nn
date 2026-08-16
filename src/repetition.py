"""Threefold-repetition draw, as a driver-level rule.

`monster_chess.is_terminal` fires only on king absence or the 150-turn cap
(CONTEXT.md section 1). Shuffling endings therefore run the full clock and are
relabelled +-0.5 by heuristic sign. Measured 2026-08-16 on V22 self-play: in
the capped games the network drives, the last 100 records contain as few as
**29 distinct positions**, so those games are spending most of their length
re-deriving the same handful of states.

**This is a rules change and is OFF by default.** Two consequences the caller
must accept before enabling it:

1. **It is not comparable to earlier numbers.** The same break the 2026-08-03
   captures-only correction caused, which invalidated the whole v19 ladder.
2. **It lowers Black.** A capped ending currently scores **-0.5** ("leaning
   Black"); a repetition draw scores **0.0**. It also hands White a saving
   resource, and White is unusually good at using it: `Ke1e2` then `Ke2e1` is a
   legal action pair returning the identical position, so White can pass a turn
   and Black cannot.

It lives here, in the driver, rather than in the search: repetition is
path-dependent, so putting it in the tree reintroduces the graph-history
problem and fights any transposition table.
"""
import os
from collections import Counter

REPETITION_ENV = "MONSTER_REPETITION"
REPETITION_N_ENV = "MONSTER_REPETITION_N"

# FOURfold, not the chess convention of three. Swept 2026-08-16 over 24 games:
#
#   threshold | Black wins | score  | mean records | wins destroyed
#      3      |     4      | 0.3542 |     77.0     |      1
#      4      |     5      | 0.3750 |     84.2     |      0
#      5      |     5      | 0.3750 |     84.8     |      0
#      6      |     5      | 0.3750 |     86.6     |      0
#
# Threefold truncated a real conversion -- game_00004, a king capture at ply
# 130 cut to a draw at ply 84 -- because Black repeats a position while
# maneuvering, which is ordinary technique against a king that can pass its
# turn. The fourth occurrence is the first that actually indicates shuffling.
# Five and six only add records back, so the entire risk sits in the step from
# 4 to 3. At 4 the speedup is 1.41x (2m59s vs 4m13s) for an identical score.
DEFAULT_THRESHOLD = 4


def repetition_enabled():
    """Off unless MONSTER_REPETITION is set, so no existing result moves."""
    return os.environ.get(REPETITION_ENV, "").strip().lower() in (
        "1", "true", "yes", "on")


def repetition_threshold():
    """How many occurrences end the game.

    Three is the chess convention, not a law of this game, and it is measurably
    too tight here: at threefold a real Black win was truncated (game_00004,
    a king capture at ply 130 cut to a draw at ply 84). Black converts against
    a double-moving king by MANEUVERING, and repeating a position while
    improving is ordinary technique -- so the threshold is tunable and the knee
    is an empirical question, not a rules question.
    """
    raw = os.environ.get(REPETITION_N_ENV, "").strip()
    if not raw:
        return DEFAULT_THRESHOLD
    try:
        return max(2, int(raw))
    except ValueError:
        return DEFAULT_THRESHOLD


def position_key(game):
    """What a repetition rule keys on.

    Piece placement, side to move, castling and en passant -- the same identity
    a FEN carries minus the clocks -- plus `white_half_pending`, because White
    is mid-turn there and the position is not the same one it will be after its
    second half. The move counters are excluded: they always differ, and
    including them would mean nothing ever repeats.
    """
    parts = game.fen().split()
    placement, side, castling, ep = parts[0], parts[1], parts[2], parts[3]
    return (placement, side, castling, ep,
            bool(getattr(game, "white_half_pending", False)))


class RepetitionTracker:
    """Counts settled positions and reports when one has occurred N times.

    Only *settled* positions count -- the mid-turn state between White's two
    half-moves is not a position either player can claim a repetition on, and
    counting it would fire on ordinary play.
    """

    def __init__(self, threshold=None, enabled=None):
        self.threshold = (repetition_threshold() if threshold is None
                          else int(threshold))
        self.enabled = repetition_enabled() if enabled is None else bool(enabled)
        self.counts = Counter()
        self.fired_at = None

    def record(self, game, ply=None):
        """Count this position. Returns True once the threshold is reached."""
        if not self.enabled:
            return False
        if getattr(game, "white_half_pending", False):
            return False
        self.counts[position_key(game)] += 1
        if self.counts[position_key(game)] >= self.threshold:
            if self.fired_at is None:
                self.fired_at = ply
            return True
        return False

    @property
    def draw_result(self):
        """A repetition is a real draw, not a lean.

        The +-0.5 cap relabel is a *proxy* for an unfinished game. A position
        repeated three times is drawn by rule, so it carries no lean for either
        side.
        """
        return 0.0
