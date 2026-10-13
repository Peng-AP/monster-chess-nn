# Gen56 plan (draft, October 13, 2026; awaiting the owner)

Owner, October 12, on the census recommendations: "Yes, work on it".

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
| Extra teacher **v29**, book starts | 700 | **New teacher.** It refutes gen55's d4 + f4 and draws lines gen55's White loses |
| Extra teacher **gen54**, book starts | 700 | Replaces Arm R and LR. A different Black repertoire |
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

## Decisions needed

1. Approve the data plan: book starts, v29 and gen54 as extra teachers, and
   the pool.
2. Approve the per-colour selection rule.
3. Approve or amend the promotion-rule revision.
4. Approve launch. The driver is ready to rehearse once the per-colour
   baselines finish (they hold the GPU until about 7 AM).
