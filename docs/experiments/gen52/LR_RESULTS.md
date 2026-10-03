# Gen52 Arm LR (wide tower on Arm R's relabelled data): results, completed October 3, 2026

Step 2 of `docs/plans/OVERNIGHT_20261002_PLAN.md`, selected by Arm R's
"labels help" verdict. Driver `tools/gen52_variant_campaign.py --variant lr`:
Arm L's receipted train command with only the data and model directories
changed. It ran in the overnight chain, October 2 10:15 → October 3 04:12,
with no failures. Evidence is in
`benchmarks/gen52_program/gen52_largeramp_20261002/production/`.
**Nothing was promoted.**

## Verdict: null (pre-declared rule, against Arm L)

| Criterion | Arm LR | Required | Met |
|---|---:|---|---|
| Score vs Arm L at 3,200 (400 g) | 50.4%, 95% range 47.6–53.1% | lower end > 50% | **no** |
| Distinct-game score vs Arm L | 53.4% (114 distinct games) | > 50% | yes |
| Held-out mean (B2, v27) | **89.7%** | ≥ 92.5% − 1.0 pp | **no** |

**Consistent labels did not measurably help the wide network against its own
baseline, and its held-out scores slipped.** As 1.9M networks, the same
relabelling gave +9 points (Arm R vs Arm B); on the 4.0M tower it gave
nothing measurable.

## Gate v4 against v29: FAIL, on the White floor only

| Leg | Score | White W/D/L | Black W/D/L | Distinct games only |
|---|---:|---|---|---:|
| vs_bar @3,200 (400) | **69.6%** | 23/113/64 | 198/2/0 | 58.7% (61) |
| vs_bar_confirm @3,200 (400) | **70.8%** | 24/120/56 | 198/2/0 | 56.0% (53) |
| deep guard @12,800 (160) | **75.3%** (pass) | 51/16/13 | 43/37/0 | 58.9% (23) |

- **Passes:** the aggregate on both 3,200 legs and the whole deep guard.
  - The guard's White is +55 pp above v29's deep self-par.
  - The guard's Black is −4.4 pp, inside the −10 pp floor.
- **Fails:** White on both 3,200 legs, at −9.6 pp and −7.4 pp against v29's
  self-par, where the floor is −5 pp.
- **Not eligible for promotion anyway:** the held-out mean (89.7%) is below
  v29's 92.7% − 1 pp.

This is the narrowest gate miss of any candidate so far.

## Selection

The nominee is epoch 8, by the gen51 rule (best screen rank among checkpoints
passing the deep guard).

| Epoch | 12,800 probe vs v29 |
|---|---:|
| 8 (nominee) | 78.75% |
| 11 | **88.75%** |
| 13 | 73.75% |

Epoch 11 probed higher, but it ranked below epoch 8 on the 3,200 screen. The
rule picks by screen rank, so epoch 8 stands.

## Diagnostics (3,200 simulations, 160 games)

| Opponent | Arm LR | Arm L | Arm R |
|---|---:|---:|---:|
| B2 (held out) | 89.1% | 90.9% | 90.0% |
| v27 (held out) | 90.3% | 94.1% | 93.75% |
| gen49 | 64.4% | 64.1% | 66.6% |
| v28 | 73.1% | 70.9% | 74.1% |

Head-to-head:

| Match | Score | as White W/D/L | as Black W/D/L | Distinct games only |
|---|---:|---|---|---:|
| LR vs Arm L @3,200 (400) | 50.4% | 17/108/75 | 65/131/4 | 53.4% |
| LR vs Arm L @12,800 (160) | 64.4% | 59/11/10 | 25/27/28 | 48.3% |
| LR vs Arm R @3,200 (400) | **45.1%** | 5/90/105 | 62/137/1 | 49.2% |

Self-play, White W/D/L by the colour actually played:

| Simulations | White W/D/L | White score |
|---:|---|---:|
| 3,200 | 1/153/46 | 38.75% |
| 12,800 | 80/40/40 | **62.5%** |

**Elo placement** (3,200, joint fit with the round robin): Arm LR **2589**
(2536–2658), v29 2559 (2507–2624), Arm C 2566, Arm B 2560. LR is the
highest point estimate, but the intervals overlap.

## Reading

1. **The two gains do not stack.** On the 1.9M network the label fix beat its
   baseline by 9 points. On the 4.0M network it is a null, and Arm LR loses
   to Arm R (1.9M, same data) 45.1% at 3,200.
2. **LR is the strongest-looking engine against v29** (70% in the gate legs,
   75% in the deep guard). Its White is still below v29's self-par at 3,200,
   and its held-out scores slipped. It is not eligible for promotion.
3. **Deep search helps LR's White a great deal** (self-play White 62.5% at
   12,800 against 38.75% at 3,200). The depth-scaling run (morning chain) tests
   whether that holds against the field.
