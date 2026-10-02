# Gen52 Arm R (deep-value data on the main game-result labels): results, completed October 2, 2026

Plan `docs/plans/GEN52_RAMP_PLAN.md` (criteria fixed before any result).
Driver `tools/gen52_ramp_campaign.py`. Production ran as one managed job,
October 1 14:37 → October 2 05:27, with no failures. Evidence is in
`benchmarks/gen52_program/gen52_ramp_20261001/production/`.
**Nothing was promoted.**

## Verdict: the labels help (pre-declared rule, against Arm B)

| Criterion | Arm R | Required | Met |
|---|---:|---|---|
| Score vs Arm B at 3,200 (400 g) | **59.1%**, 95% range 56.8–61.4% | lower end > 50% | yes |
| Distinct-game score vs Arm B | **53.1%** (87 distinct games) | > 50% | yes |
| Held-out mean (B2, v27) | **91.9%** | ≥ 91.4% − 1.0 pp | yes |

The driver records this as `capacity_helps`, because it reuses Arm L's
verdict function and that is the function's name for a pass. For this arm
it is the plan's "labels help".

**The audit confirms the mechanism.** Strong-games bias:

| Model | Bias (95%) |
|---|---|
| Arm R | **−0.005** (−0.013 to +0.003) |
| Arm B | +0.063 (0.054–0.072) |
| v29 | +0.017 (0.009–0.024) |

Putting the deep-value rows on the same labels as everything else removed
the White shift entirely. The deep-value share of value weight is unchanged
at 21.6%.

## Against v29: still not stronger

- **Elo** (joint fit with the round robin): Arm R **2583** (2528–2649), level
  with v29's 2582 (2528–2648). Arm B is 2557.
- **Gate v4 against v29: FAIL.**

| Leg | Score | White W/D/L | Black W/D/L | Distinct games only |
|---|---:|---|---|---:|
| vs_bar @3,200 (400) | 39.75% | 7/66/127 | 38/162/0 | 60.0% (51) |
| vs_bar_confirm @3,200 (400) | 37.6% | 2/63/135 | 34/166/0 | 55.1% (45) |
| deep guard @12,800 (160) | 50.9% | 1/1/78 | 80/0/0 | 60.7% (11) |

- Over distinct games Arm R scores well above 50% against v29 (59.4%
  combined).
- The game-weighted score is 38.7% because **v29's Black repeatedly beats
  Arm R's White in the openings the sampler draws most often**. As White,
  Arm R lost 262 of 400 games against v29 at 3,200.
- The gate fails on the aggregate and on White, at both depths.

## Selection

The nominee is epoch 21, a fallback: none of the three finalists passed the
deep guard. Its 12,800 probe against v29 was 50.0%, with every game won by
Black (White 0/0/40, Black 40/0/0). The other finalists, epochs 30 and 23,
probed 40.0% and 43.75%.

## Diagnostics (3,200 simulations, 160 games)

| Opponent | Arm R | Arm B | Arm L |
|---|---:|---:|---:|
| B2 (held out) | 90.0% | 87.8% | 90.9% |
| v27 (held out) | 93.75% | 95.0% | 94.1% |
| gen49 | 66.6% | 60.6% | 64.1% |
| v28 | 74.1% | 70.9% | 70.9% |

Head-to-head, Arm R against Arm B:

| Simulations | Arm R | as White W/D/L | as Black W/D/L | Distinct games only |
|---:|---:|---|---|---:|
| 3,200 | 59.1% | 2/145/53 | 124/76/0 | 53.1% |
| 12,800 | 69.7% | 30/31/19 | 55/22/3 | 52.4% |

Self-play, White W/D/L by the colour actually played:

| Simulations | White W/D/L | White score |
|---:|---|---:|
| 3,200 | 0/145/55 | 36.3% |
| 12,800 | 2/152/6 | 48.75% (very drawish) |

## Reading

1. **The label fix worked as designed.** The bias is gone (−0.005), and Arm R
   beats its own baseline (Arm B) clearly at both depths. That is the largest
   head-to-head margin of any gen52 arm against Arm B.
2. **It does not make a stronger engine than v29.** Elo is level and the gate
   fails.
   - Against v29 the result depends on how games are counted: 59% over
     distinct games, 39% weighted by how often the sampler draws each
     opening.
   - Arm R's White loses repeatedly to v29's Black in v29's preferred lines.
3. Every gen52 arm's gain over Arm B comes as Black. As White, all are below
   v29's self-play White. Whether that is a fault or the game's balance (the
   owner's point) is not settled by these numbers.

Per `docs/plans/OVERNIGHT_20261002_PLAN.md`, this verdict selects **Arm LR**
(the wide tower on Arm R's data) as the next step. The chain initially
mis-read the verdict name and chose L2; that was corrected at 05:36, before
the variant step started (commit `f5692b0`).
