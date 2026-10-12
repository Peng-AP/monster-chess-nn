# Where the project stands: game value, distance from perfect play, why progress stalled

October 12, 2026, written for the owner's question: "How close are we to perfect
play? I do believe that black is better by the structure of the game. Right now,
each model's progress seems to be just its own preferred opening..."

This uses only existing evidence plus two cheap measurements:

- a count of the opening positions;
- a 3,200-simulation search of all 58 White first turns by gen55 and v29
  (`benchmarks/game_value_20261012/first_turn_values_3200.json`).

## 1. The opening space is tiny, and the strong models use almost none of it

- White has **58 distinct positions** after its first turn, and the game has
  **1,156** after Black's reply. (Chess has 400 after one move each.)
- Across **19,480 games between the strong models** (gen51-deepvalue/v29,
  gen52 arms, gen53, gen54, gen55; gates, diagnostics and round robins,
  October 1–9), only **11 of the 58 first turns** were played:

| White first turn | Share of games |
|---|---:|
| d4 + e4 | **87.2%** |
| e4 + e5 | 6.8% |
| d4 + f4 | 3.6% |
| d4 + d5 | 1.7% |
| all other 7 | 0.7% |

- The 16 sampled opening half-moves give each 400-game gate leg only about
  45–65 distinct games.

## 2. Among strong models, White almost never wins

The same 19,480 games, by search depth:

| Simulations | Games | White wins | Draws | Black wins | White score |
|---:|---:|---:|---:|---:|---:|
| 3,200 | 9,080 | 4.6% | 50.1% | 45.3% | 29.7% |
| 12,800 | 3,120 | **1.4%** | 59.0% | 39.6% | 30.9% |

Gates make the same point from the other side: every recent candidate scored
**97–100% as Black** against v29, gen53 and gen54. Matches between strong models
are decided almost entirely by **how well White survives**.

## 3. The trend is toward Black as play improves

- **Early play (August):** our records show White dominating self-play. Free-play
  White scores fell from 0.87 (v24) to 0.79 (gen33) to 0.58 (gen38).
- **Now:** strong-vs-strong White scores about 30%.
- **The chain has been Black:** gen11→gen17 improved Black by +0.22 and White
  by +0.003 (project memory, measured under a block baseline).
- **Deeper search helps Black:** self-play White falls from 47% to 24% (v29) and
  from 39% to 17% (gen55) between 3,200 and 12,800. The repetition-aware engine
  turned 43 of gen54's Black draws into wins.
- **The networks agree:** gen55 and v29 searched every one of the 58 first
  turns. **None is good for White by both.**

| Network | Best | Median | Worst |
|---|---:|---:|---:|
| gen55 | +0.004 | −0.200 | −0.265 |
| v29 | +0.138 | −0.168 | −0.214 |

v29's best first turn is e4 + d4; gen55's is d4 + f4.

**Conclusion 1.** All the evidence says **Black is better by the structure of
the game.** The question is whether White can hold a draw.

- White winning is essentially off the table between competent players.
- The perfect-play value is either a Black win, or a draw that needs very
  precise White defence (the 40–59% draws). Our data cannot yet distinguish
  these.
- Nothing here is a proof.

## 4. How close is "perfect play"? Not close, and unevenly so

**Black's side** is near saturation against today's opponents: 97–100% against
the release, and every conversion test since repetition awareness is won.

**White's side** is far from it:

- **The evaluation has not converged.** The same pair of models gives results
  15–25 pp apart at 3,200 and 12,800. A perfect player's results would not
  depend on more search.
- **Self-play is blind to its own holes.**
  - gen55 rates d4 + f4 as its best first turn (+0.004). Against v29 that line
    lost **35 of 35**. In gen55's self-play it draws.
  - gen54's c4 + Ke2 line lost 60 of 141 against v29.
  - A model's Black cannot refute its own White's preferred line, so training
    never sees it lose there.
- **Coverage is minimal.** 47 of the 58 first turns are never played, and we do
  not know whether any of them holds better.
- **Technical wins went unclaimed until recently.** Forced captures were missed
  without the solver, and repeated positions were drawn without repetition
  awareness.

## 5. Why progress stalled (the owner's reading is correct)

A "generation" now mostly changes **which White line the model prefers** and
**whether its Black refutes the opponent's line**.

- Because matches are decided by White's survival in 1–3 opening lines, results
  are **non-transitive**:
  - gen53 rated +105 Elo against the group, but scored 52% head-to-head against
    v29;
  - gen55 beats gen54 70%, yet fails against v29 on White.
- Each generation repairs the losing line it was shown and finds another:
  gen54 → c4 + Ke2, gen55 → d4 + f4.
- The promotion gate therefore mostly asks one question: does the candidate's
  preferred White line survive v29's Black at 3,200?

## 6. What would move the project forward

1. **Measure the colours separately.** Black's score against the strongest
   White and White's score against the strongest Black, per first turn, instead
   of one aggregate from model-chosen openings.
   - This revises the standing rule that normal-start, model-chosen openings
     are the primary instrument. **The owner decides.**
2. **Map the opening tree directly; its small size is an asset.**
   - An "opening census": for each of the 58 first turns, and the main Black
     replies, play the strongest pair at depth in both directions.
   - It answers (a) whether any White setup holds against the best Black, and
     (b) which setups each model has never learned.
   - About an overnight run for the 58 first turns at 12,800.
3. **Train White on what it does not know.**
   - Generate White's training games across all the setups the census rates
     viable, against the strongest refuting Blacks (v29 and the current best).
   - Do not leave it to one preferred line that self-play never punishes.
   - Black needs little more beyond conversion technique.
4. **Treat the engine as a product of the game's value.** If Black is winning,
   the ideal engine:
   - as Black, converts quickly and surely (solver, repetition awareness,
     moves-left);
   - as White, defends for draws and fortresses and avoids the losing lines.

   The site's ratings should report colour-separated strength.
5. **Pause the generation-by-generation chase until the census is in.** Another
   generation without it will likely trade one hole for another again.
