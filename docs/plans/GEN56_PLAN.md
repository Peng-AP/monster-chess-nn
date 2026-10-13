# Gen56 plan (October 13, 2026; launched with option d)

Owner, October 12, on the census recommendations: "Yes, work on it".
Owner, October 13, on option d: "Yes, try it, and then start gen 56".

Evidence: `docs/experiments/game_value/OPENING_CENSUS.md`.

- No White first turn holds against the strong Blacks.
- Each of White's best first turns has one or two Black replies that hold White
  to 0–19%. Most other replies lose.
- Drawn structures, once reached, are real holds.
- Self-play never shows a model its own holes. gen55's White plays d4 + f4,
  which v29's Black refutes 10/10, while gen55's own Black does not.

## The idea

Train on the positions that decide games, not on whichever opening the teacher
happens to prefer, and measure each colour separately.

## Data

Teacher: **gen55** (the strongest Black; 70% against gen54).

| Source | Games | Change from gen55 |
|---|---:|---|
| gen55 self-play, **normal start** (+ forks, reanalysis) | 1,400 + 400 forks | Halved |
| gen55 self-play from **census book starts** | 1,400 | **New** |
| **Per-colour teachers**, book starts: White gen55 or gen54 vs Black gen53 or v29 (4 pairings × 350) | 1,400 | **New** (option d). Both sides' moves are policy targets. Replaces the v29 and gen54 self-play book games of the draft |
| Pool: gen55 vs v29, gen54, gen53, gen52 B, C, L, LR, R | 1,376 (86 per opponent per colour) | gen54 in, Arm A out |
| Deep-value continuations (gen55, reference v29), ramped | 768 × 2 | Unchanged |

**Book starts:** 70 lines from the normal start, with games shared by weight.

| Group | Positions | Weight | Share |
|---|---:|---:|---:|
| Census key first turns (Black to move): what White plays and what Black must answer | 14 | 2 each | 38% |
| Black's two best replies to each of White's best 6 first turns (White to move): White learns to defend the critical lines | 12 | 2 each | 32% |
| The other 44 first turns, for breadth | 44 | 0.5 each | 30% |

- The list is in `docs/plans/gen56_teacher_decision.json` and derived from the
  census reports.
- Book games are ordinary self-play from the position, with the line as its
  history: repetition-aware engine, same exploration, same labels.

**Unchanged:**

- ramped labels;
- the 1.9M tower, scratch training, seed 3173, replay of the last 8
  generations;
- held-out B2 and v27, never teachers or opponents.

## Selection: per-colour

1. Screen against gen55 and run deep probes, as before.
2. **Colour check per finalist:** 168 games against each of gen55 and v29 on
   the census key book (14 positions × 6 copies, 8 sampled half-moves).
3. **Rule, pre-declared:** among finalists that pass the deep guard, keep
   those whose White score (the mean against the two references) is at least
   the teacher's own on the same book, and whose Black score is no more than
   2 pp below the teacher's. Then take the best screen rank. If none
   qualifies, take the best White score and say so.

## Evaluation

All in one chain, as gen55:

- **The per-colour instrument** (`tools/colour_instrument.py`): White and Black
  scores against gen55 and v29 on the full census book, 560 games each. This
  is the **new primary measure**.
- Gate v4 against gen55 and against v29 (normal start), kept for continuity and
  for the current promotion rule.
- The B2/v27 held-out diagnostics, gen49, v28, self-play, the v27-position
  probe and the value audit.

## Promotion (owner decision needed)

`PROMOTION_RULE.md` requires gate v4 against the release. Its White floor
measures model-chosen openings, which the census shows are mostly
repertoire. Proposed revision:

1. **Per-colour non-regression against the release:** on the census book,
   against both references, the candidate's White score and Black score are
   each no worse than v29's own by more than 3 pp.
2. The held-out mean, unchanged.
3. The owner's sign-off, unchanged.

Gate v4 stays as a reported diagnostic.

## Cost

About 33 hours, as gen55: 1,400 fewer normal-start games, plus 1,400 book
games, plus about 2 hours of colour checks.

## What the per-colour baselines change (October 13)

Source: `docs/experiments/game_value/COLOUR_BASELINES.md`. The recipes and
the decision file are unchanged; these are options for the owner.

| Model | White mean | Black mean | Passes the proposed rule? |
|---|---:|---:|---|
| gen55 | 18.6% | 83.4% | No: Black vs gen55's White −6.2 pp |
| v29 | 10.6% | 87.4% | (baseline) |
| gen54 | 17.5% | 82.3% | No: Black vs v29's White −8.7 pp |
| gen53 | 13.8% | **91.3%** | **Yes**, all four cells at or above v29 |

**a. Swap extra teacher gen54 for gen53** (recommended).

- gen53 has the best Black of the six models measured. gen54 has the
  weakest Black of the three recent models.
- gen55 already supplies the best White.
- The cost is the same.

**b. Make selection's Black reference v29's cells, not the teacher's**
(recommended).

- Selection and the proposed promotion rule would then test the same thing.
- Under the current rule, a candidate with gen55's Black passes selection
  but fails promotion on the cell where gen55 fails.

**d. Per-colour teachers** (owner, October 13: "different teachers for
different colors?"). Recommended in place of a.

- Replace the two extra teachers' self-play book games (v29 700, gen54 700)
  with 1,400 book games between a White teacher and a Black teacher.
- **Whites:** gen55 (best White mean, 18.6%) and gen54 (best White against
  v29's Black, 18.8%).
- **Blacks:** gen53 (91.3%) and v29 (87.4%), the two best Blacks.
- Four pairings × 350 games. Each position's games are dealt evenly over the
  pairings.
- Both sides' moves are policy targets. The value labels are the game
  results, so White learns its outcomes against the Blacks that refute it.
- **Why:** self-play hides a model's holes. gen55's White beats gen55's own
  Black in e1e2 e2e3 and e2e3 f2f4 (21/40, 18/40), yet loses 19/20 and 20/20
  there to v29's Black.
- **Cost:** about the same.
- **Generator:** opt-in `book_pairings` in `tools/stateful_generation.py`,
  tested. Old recipes are unchanged.
- gen55's own 1,400 book self-play games stay as they are.

**c. gen53's held-out mean on the repetition-aware engine: done, still fails.**

- Driver: `tools/heldout_remeasure_20261013.py`.
- Evidence: `benchmarks/heldout_remeasure_20261013/`.
- Settings: 160 games per match at 3,200 sims, 16 sampled plies.

| Model | vs B2 | vs v27 | Mean | Old engine |
|---|---:|---:|---:|---:|
| gen53 | 100.0% | 75.9% | **88.0%** | 88.3% |
| v29 | 98.8% | 86.9% | 92.8% | 92.7% |

- The bar is 92.8% − 1.0 = 91.8%. gen53 misses it by 3.8 pp.
- The whole gap is gen53's White against v27: 41/1/38 W/D/L. v29's White
  went 38/42/0 there. gen53's Black won all 160 games.
- So gen53 stays ineligible: its White still loses at the normal start to
  v27's Black. This is the known v27 hole.
- This doesn't affect option d, which uses only gen53's Black.

## Decisions (owner, October 13)

"Yes, try it, and then start gen 56", after option d was explained.

1. **Data plan:** approved, with option d. The per-colour teacher games replace
   the v29 and gen54 self-play book games. Option a (gen53 as a self-play
   teacher) is subsumed.
2. **Selection rule:** as written, with Black within 2 pp of the teacher's.
   Option b (v29's cells as the Black reference) was not taken.
3. **Promotion rule:** not decided. The campaign reports eligibility under both
   the current rule and the proposed per-colour rule; the owner decides at
   sign-off.
4. **Launch:** approved. A fresh rehearsal runs first, with option d, under the
   new rehearsal namespace `rehearsal_gen56e_20261013`. Production follows
   automatically if it passes.
