"""Threefold-repetition draw, as a driver-level rule.

`monster_chess.is_terminal` fires only on king absence or the 150-turn cap
(CONTEXT.md section 1). Shuffling endings therefore run the full clock and are
relabelled +-0.5 by heuristic sign. Measured 2026-08-16 on V22 self-play: in
the capped games the network drives, the last 100 records contain as few as
**29 distinct positions**, so those games are spending most of their length
re-deriving the same handful of states.

**This is a rules change and it is ON by default** (owner, 2026-08-16).
Disable with `MONSTER_NO_REPETITION=1`. Two consequences it carries:

1. **It is not comparable to earlier numbers.** The same break the 2026-08-03
   captures-only correction caused, which invalidated the whole v19 ladder.
   Every gate, match and screen taken before this is on the other side of it.
2. **It changes the training label, not the match score.** A capped ending
   already scored as a draw under the captures-only rule, so match scoring
   moves only where a *decisive* result changes. What moves is the label:
   -0.5 ("leaning Black") becomes 0.0, removing partial value credit the old
   proxy gave Black. It also hands White a saving resource, and White is
   unusually good at using it -- `Ke1e2` then `Ke2e1` is a legal action pair
   returning the identical position, so White can pass a turn and Black cannot.

It lives here, in the driver, rather than in the search: repetition is
path-dependent, so putting it in the tree reintroduces the graph-history
problem and fights any transposition table.
"""
import os
from collections import Counter

REPETITION_OFF_ENV = "MONSTER_NO_REPETITION"
REPETITION_ENV = "MONSTER_REPETITION"      # retained; setting it also enables
REPETITION_N_ENV = "MONSTER_REPETITION_N"

# THREEfold (owner, 2026-08-16). The sweep below measured what it costs:
#
#   threshold | Black wins | score  | mean records | wall  | wins destroyed
#      3      |     4      | 0.3542 |     77.0     | 2m17s |      1
#      4      |     5      | 0.3750 |     84.2     | 2m59s |      0
#      5      |     5      | 0.3750 |     84.8     |   -   |      0
#      6      |     5      | 0.3750 |     86.6     |   -   |      0
#
# Threefold is 1.85x against fourfold's 1.41x, and it ends one game that would
# otherwise have been a Black king capture. That game is why 4 was the earlier
# default -- but inspection showed the "win" was not a conversion being cut
# short: Black held ELEVEN pieces against a bare king from record 40 and took
# until record 129 to finish, repeating positions on the way. Under threefold
# that reads as a draw, which is a fair verdict on the play.
DEFAULT_THRESHOLD = 3


def repetition_enabled():
    """ON by default (owner, 2026-08-16). Disable with MONSTER_NO_REPETITION=1.

    This is a RULES CHANGE and it breaks comparability with every number taken
    before it, exactly as the 2026-08-03 captures-only correction did. It is on
    because the owner directed it after seeing that cost.
    """
    if os.environ.get(REPETITION_OFF_ENV, "").strip().lower() in (
            "1", "true", "yes", "on"):
        return False
    return True


def repetition_threshold():
    """How many occurrences end the game.

    Tunable because the right number is empirical, not a rules question. The
    one game threefold ends that fourfold does not (game_00004) was examined
    rather than assumed: it is not a conversion cut short. Black held eleven
    pieces against a bare king from record 40 and needed until record 129 to
    capture, repeating positions on the way -- floundering, not maneuvering,
    since maneuvering by definition reaches NEW positions.
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
