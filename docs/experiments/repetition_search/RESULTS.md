# Repetition awareness in the search: results, October 7, 2026

Owner, October 7: "if it can help it it shouldn't repeat."

**The change** (commits `50edbf0` and `2fbc8d8`):

- The search tree scores a position's threefold occurrence as a draw.
- The counts come from the drivers' `RepetitionTracker`, through the game object.
- The search is told nothing the rule itself didn't already decide: repetition
  was already a draw by rule, and the search simply didn't know.

Effect on play:

- A side that is ahead now avoids repeating.
- A side that is behind may now seek the repetition.
- `MONSTER_NO_REPETITION_SEARCH=1` turns the awareness off;
  `match.py --no-repetition-search-a/-b` turns it off per side.

The engine is installed: `native/monster_native.pyd` SHA256 `ad494962…`. The
previous engine is `native/monster_native.pre_repetition_20261007.pyd`.

**This changes the runtime identity, so every gate measures its par afresh
from now on.**

Driver: `tools/repetition_ab_20261007.py`. Evidence:
`benchmarks/repetition_search_20261007/`.

## The owner's Watch game (gen53 vs gen53, 3,200 simulations, temperature 0)

Black to move, a queen, two rooks, two bishops and a knight against a bare king,
from four points of the game:

| Start | Solver on | Solver off | Solver to 5 moves |
|---|---|---|---|
| Before | 1/4 wins | 0/4 | 4/4 |
| **After** | **4/4** | **4/4** | 4/4 (shorter: 4–9 turns) |

gen53 now converts with no help from the solver, so the solver depth was left
unchanged.

## gen54's diagnostic draws, replayed on the same seeds (160 games each)

| Match | Before | After | gen54 as Black, before → after |
|---|---:|---:|---|
| gen54 vs v28 | 71.9% | **76.9%** | 58/22/0 → **80/0/0** |
| gen54 vs B2 | 92.2% | **98.4%** | 59/21/0 → **80/0/0** |

Before the change:

- All 43 of gen54's Black draws were repetitions with gen54 at least a rook
  ahead (21 distinct games).
- On the same seeds with the aware engine, every one became a king capture.

So, against these opponents, those positions were winnable. This is now
measured, not assumed.

The change helps the opponent too:

- Against v28, gen54's White fell from 57.5% to 53.8%.
- v28, now also unwilling to repeat while ahead, converted 18 games against
  gen54's bare-king White instead of 12.
- 22 games (14 distinct) ran to the 150-turn limit with v28 still unable to
  capture. These count as draws.

## Same network, awareness on vs off (400 games each)

| Network | Aware side's score | Aware as White W/D/L | Aware as Black W/D/L |
|---|---:|---|---|
| gen54 | 49.4% | 0/24/176 | 174/23/3 |
| v29 | 50.3% | 3/188/9 | 9/190/1 |

The change is neutral in level play: no cost and no gain where neither side is
clearly ahead. v29's self-play is 94% repetition draws, and those stay draws.

## Notes

- The forced-capture solver does not consider repetition. Its lines are at
  most 3 Black moves long, which leaves no room for a threefold repetition
  along the way.
- The Python reference engine (`src/mcts.py`) is unchanged. Gates, self-play
  and the site use the native engine.
- Training data generated from now on comes from the aware engine. Endgames
  that used to end in repetition draws will be played out to captures.
