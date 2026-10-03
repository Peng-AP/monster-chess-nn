# Teacher selection for gen53 (declared October 3, 2026, before any game)

Owner: "Choosing the right teacher is important, spend time making sure we
have the best on hand."

A teacher is used at three depths when generating gen53's data
(`docs/plans/GEN53_PLAN.md`, gen52 recipe):

- self-play games at **1,600** simulations;
- deep-value continuations at **6,400**;
- reanalysis at **12,800**.

Every comparison so far was at 3,200, except the depth ladders, which rated
each model against the field. Teaching quality is measured at the depths the
teacher works at.

## Candidates

The five that are best or competitive on current evidence:

| Candidate | Why |
|---|---|
| v29 | release; gen52's teacher |
| gen52 Arm R | best against the group at 3,200; only model gaining +149 to 12,800 |
| gen52 Arm LR | beats v29 72% head-to-head; 70% in its gate legs |
| gen52 Arm L | +34 against the group |
| gen52 Arm C | beats v29 63% head-to-head |

Arms A and B are dominated (each is below one of the above on every
measure) and left out.

## Measurement

`tools/teacher_rr.py`, three independent 5-player round robins, all 10
pairings each, sampled openings (16 temperature plies), colours split
evenly:

| Depth | Games per pairing | Seed block |
|---:|---:|---|
| 1,600 | 100 | 4.02e9 |
| 6,400 | 100 | 4.04e9 |
| 12,800 | 60 | 4.06e9 |

Each round robin is fitted separately (Bradley–Terry, Elo relative to
v29 = 0, 1,000-rep bootstrap).

## Selection rule, fixed now

1. **Eligibility:** strong-games value bias within ±0.03
   (`tools/value_colour_audit.py`). The audit showed that a teacher's bias
   passes to its students, so a biased teacher would re-teach the problem
   Arm R fixed. Current values:

   | Candidate | Bias | Eligible |
   |---|---:|---|
   | v29 | +0.017 | yes |
   | R | −0.005 | yes |
   | LR | −0.007 | yes |
   | L | +0.071 | **no** |
   | C | +0.049 | **no** |

   L and C still play, so the rating of every player is measured, but they
   cannot be selected.
2. **Score:** mean Elo across the three depths, equal weights, among
   eligible candidates.
3. **Decision:** the highest score is the recommended teacher. If the
   bootstrap 95% interval of the score difference between the top two
   eligible candidates includes 0, they are declared **tied**, and the tie
   is broken by Elo at 12,800. That is the depth of the deep-value and
   reanalysis data that produced the last real gain.
4. This **recommends** a teacher. Launching gen53 (which its own rule 1
   currently blocks) remains the owner's decision.

## Conduct and time

- Driver `tools/teacher_chain_20261003.py`: the three depths in order, then
  `tools/teacher_select.py` writes `benchmarks/teacher_selection_20261003/selection.json`.
- One managed job. Each step is resumable. Evaluation only; nothing is
  trained or promoted.
- **Guide:** 1,600 about 0.8 h, 6,400 about 3.5 h, 12,800 about 4 h; about
  8–9 h in all, re-estimated from the running job.
