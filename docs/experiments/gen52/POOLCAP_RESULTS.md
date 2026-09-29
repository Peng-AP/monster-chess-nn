# Gen52 Arm C (capped deep-value share): results, completed September 29, 2026

Plan `docs/plans/GEN52_POOLCAP_PLAN.md` (criteria fixed before any result).
Driver `tools/gen52_poolcap_campaign.py`. Production ran as one managed job,
about 13.9 h, finishing at 15:06, with no failures or restarts. Evidence is in
`benchmarks/gen52_program/gen52_poolcap_20260929/production/`.
**Nothing was promoted, and gen53 was not launched.**

## Verdict: inconclusive (by the pre-declared criteria)

| Criterion (plan) | Arm C | Arm B | Met |
|---|---:|---:|---|
| 1. Held-out mean (B2, v27) ≥ Arm B | 86.9% | 91.4% | **no** |
| 2. Combined gate-v4 score vs the teacher @3,200 > Arm B | 65.3% | 47.4% | yes |
| 3. ≥ 50% vs Arm B @3,200 (400 games) | 41.9% | — | **no** |

- "Supported" needs all three, so the hypothesis is not supported.
- "Contradicted" needs Arm C worse than Arm B on both 1 and 2. Arm C is worse
  on 1 only, so the result is **inconclusive**.
- **Gate v4 against the teacher: FAIL.** Arm C is therefore not a gen53
  teacher candidate.

**What the numbers show:** capping the deep-value share did not restore
White. Arm C is still clearly weaker than the teacher as White (−15.6 pp
against the teacher's self-par), about the same as Arm B (−13.4 pp).

Criterion 2's 65.3% comes almost entirely from Black wins in openings the
teacher repeats:
- 711 of the 800 gate games repeat an earlier game.
- Counting each distinct game once, Arm C scores 49.8%, against Arm B's
  53.6% / 58.0% per leg.

## The change

- Deep-value rows now carry **12.4%** of value-training weight (gen52's own
  increment only), against **22.3%** in Arm B (gen51's rolled-forward source
  plus gen52's). gen51's deep-value arm had 13.0%.
- Everything else is Arm B's recipe with the same seeds:
  - replay gen45–gen52 plus the 1,200 pool games;
  - training from scratch, seed 3173, 30 epochs, patience 10.

## Selection (vs the teacher, gen51 deep-value = v29)

| Epoch | Screen @3,200 (200 g) | Probe @12,800 (80 g) | Nominee |
|---|---:|---:|---|
| 15 | 66.75% (lowest colour delta −13.25 pp) | 49.4% (W 0/36/4, B 3/37/0) | **yes** (fallback: none passed the deep guard) |
| 1 | 61.75% | 33.75% | |
| 14 | 60.0% | 30.0% | |

## Gate v4 against the teacher

| Leg | Score | White W/D/L | Black W/D/L | Distinct games only |
|---|---:|---|---|---:|
| vs_bar @3,200 (400) | 65.0% | 24/86/90 (33.5%) | 186/14/0 (96.5%) | 51.7% (66 games) |
| vs_bar_confirm @3,200 (400) | 65.6% | 29/78/93 (34.0%) | 189/11/0 (97.25%) | 52.6% (51 games) |
| deep_guard @12,800 (160) | 44.7% | 1/59/20 (38.1%) | 2/78/0 (51.25%) | 42.5% (27 games) |

- **Verdict: FAIL.** White is below par − 5 pp on both 3,200 legs. The guard
  aggregate is below 47.5%, and Black is below deep par − 10 pp.
- Combined 800-game colour deltas against the teacher's 3,200 self-par:
  White −15.6 pp (−19.2 to −12.1), Black +46.3 pp (+44.7 to +47.8).
  - This is the colour profile of Arm A (White −26.4, Black +46.4), not of
    Arm B (White −13.4, Black +8.1).
- At 12,800 the picture flips: Arm C's White beats the teacher's deep par
  (+19.4 pp), and its Black falls short of it (−30.0 pp).

## Diagnostics (3,200 simulations, 160 games; W/D/L from Arm C's side)

| Opponent | Arm C | Arm B | Arm A | Teacher |
|---|---:|---:|---:|---:|
| **B2 (held out)** | 81.6% (W 48/16/16, B 69/11/0) | 87.8% | 65.9% | 99.06% |
| **v27 (held out)** | 92.2% (W 63/9/8, B 80/0/0) | 95.0% | 95.9% | 86.25% |
| **Held-out mean** | **86.9%** | **91.4%** | 80.9% | 92.7% |
| gen49 (pool member) | 66.9% (W 1/57/22, B 75/5/0) | 60.6% | 61.9% | 75.6% |
| v28 (pool member) | 68.75% (W 10/48/22, B 72/8/0) | 70.9% | 58.75% | 74.9% |

Self-play, counted by the colour actually played (White W/D/L):

| | @3,200 (200 g) | @12,800 (160 g) |
|---|---|---|
| Arm C | 2 / 107 / 91 (27.8%) | 8 / 51 / 101 (20.9%) |
| Arm B | 1 / 137 / 62 (34.7%) | 39 / 57 / 64 (42.2%) |
| Teacher | 6 / 383 / 11 of 400 (49.4%) | 1 / 58 / 101 (18.8%) |

## Arm C vs Arm B head-to-head (the direct measurement)

| Simulations | Games | Arm C score | Arm C as White W/D/L | Arm C as Black W/D/L |
|---|---:|---:|---|---|
| 3,200 | 400 | **41.9%** | 1/130/69 (33.0%) | 52/99/49 (50.75%) |
| 12,800 | 160 | 53.1% | 25/35/20 (53.1%) | 27/31/22 (53.1%) |

At 3,200 Arm B is better, entirely through Arm C's White (1 win in 200). At
12,800 Arm C is slightly ahead: 53.1% over 160 games, about 1 SE from 50%
(not significant).

## Conclusions

1. **The deep-value-share hypothesis is not supported.** Halving the share
   (22.3% → 12.4%) did not recover White against the teacher or in
   self-play. It moved held-out results down, not up (86.9% vs 91.4%).
2. **White regression is common to all three gen52 trainings** (A, B, C),
   whatever the deep-value share. The remaining hypothesis from
   GEN52_RESULTS is the teacher itself: at 12,800 its Black beats its own
   White (18.8%), and students inherit that White pessimism through
   self-play value labels. This is untested.
3. **v29 remains the release and the strongest model on the evidence.** No
   gen52 arm passed gate v4 against it.

## Options for the owner (none started)

- **Test the teacher hypothesis cheaply.** Retrain Arm B's data with value
  labels for White positions taken from a different evaluator (e.g. v28
  reanalysis), or down-weight teacher self-play value rows.
- **Pause training** and spend the GPU on a round robin (v29, gen52 A/B/C,
  v28, gen49, B2, v27). That would show whether any gen52 arm is stronger
  than it looks in chained matches.
- **gen53 per GEN53_PLAN** stays blocked by its own rule 1 (no teacher
  candidate passed gate v4).
