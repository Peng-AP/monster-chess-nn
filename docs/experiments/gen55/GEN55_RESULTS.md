# Gen55 results (teacher gen54 + gen53, Arm R, Arm LR), completed October 9, 2026

- Plan: `docs/plans/GEN55_PLAN.md` (owner, October 7: "Should add one more
  teacher to the pool. Then start 55").
- Driver: `tools/gen55_campaign.py`.
- Run: one managed job, October 7 13:39 → October 9 05:24, including the
  rehearsal. No failures.
- Evidence: `benchmarks/gen55_program/gen55_20261007/production/`.
- First generation on the **repetition-aware engine**. Every par was measured
  under it.

**Nothing was promoted. gen55 is not eligible:** it fails gate v4 against v29 on
the White floor, as gen54 did.

Nominee: `models/candidates/bootstrap_main_gen_0055/arena_selected.pt`
(epoch 15), SHA256 `76d348e8…`.

## Recipe

| Source | Games | Value weight |
|---|---:|---:|
| gen54 self-play (2,800 + 400 forks; reanalysis 24k → 12k at 12,800) | 3,200 | 8.7% (gen55 increment) |
| Deep-value continuations (gen54 and gen55; ramped; reference v29) | 768 roots × 2 each | **23.9%** (11.5% + 12.4%) |
| Extra teachers: gen53, Arm R, Arm LR | 700 each | 2.1% each |
| Pool: gen54 vs v29, gen52 A/B/C/L/LR/R, gen53 | 1,376 | 4.0% |
| Self-play replay gen48–gen54 | — | 7.8–8.6% each |

The rest was unchanged: 1.9M tower, scratch, seed 3173. Training stopped early
at epoch 15.

**Training self-play is overwhelmingly Black wins for every teacher** (1,600
simulations, 30 half-moves of exploration):

| Teacher | White wins | Draws | Black wins |
|---|---:|---:|---:|
| gen54 | 12.9% | 9.1% | 78.0% |
| gen53 | 14.6% | 16.4% | 69.0% |
| Arm R | 13.4% | 15.0% | 71.6% |
| Arm LR | 14.0% | 12.1% | 73.9% |

## Selection (the new White check against v29)

v29's par on this engine: White 47.2% at 3,200; White 24.1% at 12,800.

| Epoch | Screen vs gen54 | Deep probe vs gen54 (12,800) | White vs v29 at 3,200 | Passes both |
|---|---:|---:|---:|---|
| **15** | 65.5% | 65.6% (White +25.9 pp) | **−17.2 pp** | no |
| 5 | 59.0% | 60.0% (White +13.4 pp) | −30.4 pp | no |
| 7 | 58.8% | 51.9% (White +22.2 pp) | −29.7 pp | no |

No epoch passed the White check. The pre-declared fallback took the best White
check, epoch 15, which was also the best screen.

## Gate v4 against the teacher (gen54): PASS

| Leg | Score | White W/D/L | Black W/D/L |
|---|---:|---|---|
| vs_bar @3,200 (400) | **68.8%** | 51/53/96 | 195/5/0 |
| vs_bar_confirm @3,200 (400) | **70.6%** | 61/49/90 | 194/6/0 |
| deep guard @12,800 (160) | 65.9% | 0/73/7 | 58/22/0 |

Against gen54's self-par:

- **3,200 legs:** White +32.1 / +36.1 pp, Black +5.4 / +5.1 pp.
- **Deep guard:** White +26.6 pp, Black +5.3 pp.

## Gate v4 against the release (v29): FAIL

| Leg | Score | White W/D/L | Black W/D/L |
|---|---:|---|---|
| vs_bar @3,200 (400) | 62.9% | 1/101/98 | 200/0/0 |
| vs_bar_confirm @3,200 (400) | 67.4% | 6/127/67 | 200/0/0 |
| deep guard @12,800 (160) | 64.1% | 22/1/57 | 80/0/0 |

- **Failures:** the White floor in both 3,200 legs. White is about −21.5 and
  −13 pp against v29's 47.2% par.
- gen55 won all 480 of its Black games against v29.
- gen54 failed the same way (White −19 / −21 pp).

**Where the White games against v29 go** (both 3,200 legs, first 6 half-moves):

| Line | gen55 W/D/L | gen54 W/D/L |
|---|---|---|
| e4 d4 …d5 c4 Ke2 …dxe4 | 1/34/2 (37 games) | **6/75/60** (141) |
| e4 d4 …d5 Ke2 e5 …f5 | 0/30/7 | — |
| **d4 f4 …f5 Kf2 Ke3 …e6** | **0/0/35** | — |
| e4 d4 …d5 e5 Ke2 …f5 | 5/18/9 | — |

- gen55 repaired gen54's problem line, c4 + Ke2.
- It now plays a d4/f4 setup that loses every time (35 of 35).
- The rest of its White losses are spread over 38 openings.

## Diagnostics (160 games each at 3,200, unless noted)

| Opponent | gen55 | White W/D/L | Black W/D/L | gen54 |
|---|---:|---|---|---:|
| B2 (held out) | 93.1% | 66/6/8 | 80/0/0 | 92.2% → 98.4% (new engine) |
| v27 (held out) | 91.9% | 61/12/7 | 80/0/0 | 92.2% |
| gen49 | 74.1% | 18/41/21 | 80/0/0 | 80.6% |
| v28 | 82.2% | 35/33/12 | 80/0/0 | 71.9% → 76.9% (new engine) |

**Held-out mean: 92.5%**, which meets the rule's 91.7%.

**Self-play (actual colours):**

| Simulations | Games | White wins | Draws | Black wins | White score | gen54 (same engine) |
|---:|---:|---:|---:|---:|---:|---:|
| 3,200 | 200 | 53 | 51 | 96 | 39.3% | 6.6% |
| 12,800 | 160 | 0 | 55 | 105 | 17.2% | 19.1% |

At 3,200, gen55's own White holds up far better against its own Black than
gen54's did.

## The v27 position

Against v27, 10 games per side:

- gen55 as White: 10 wins in 10;
- gen55 as Black: 10 wins in 10 (gen54 drew all 10).

The hole stays fixed.

## Value audit

Strong games, prediction minus target:

| Model | Bias | Interval | MSE |
|---|---:|---|---:|
| gen55 | +0.0047 | −0.0025 to +0.0124 | **0.0628** |
| gen54 | −0.0055 | −0.0131 to +0.0027 | 0.0664 |
| v29 | +0.0166 | +0.0090 to +0.0243 | 0.0804 |

gen55 is unbiased and has the most accurate value head yet.

## Promotion eligibility (`docs/protocols/PROMOTION_RULE.md`)

| Criterion | gen55 | Required | Met |
|---|---:|---|---|
| Gate v4 PASS vs release v29 | FAIL (White floor) | PASS | **no** |
| Held-out mean (B2, v27) | 92.5% | ≥ 91.7% | yes |

**Not eligible.** v29 remains the release.

## Reading

- gen55 is stronger than gen54 by every head-to-head measure (about 70%), and
  its self-play is much more balanced.
- **Its White against v29 at 3,200 is still the blocker.** Two generations of
  changes did not fix it: the teacher mix, the pool and the White check in
  selection.
- Each generation repairs one losing White line and finds another. gen54's was
  c4 + Ke2; gen55's is d4 + f4.
- **Common to every teacher:** training self-play is 69–78% Black wins at 30
  half-moves of exploration, while match self-play is far more balanced
  (v29: 93% draws). The training data teaches "White loses" whoever the teacher
  is.

That is the leading hypothesis. It is not yet measured.

## Next steps (proposed; the owner decides)

1. **Exploration test** (GPU, about 1–2 h). Play training-style self-play from
   one teacher at 16 vs 30 half-moves of exploration, and compare the colour
   outcomes. If 16 is far more balanced, the exploration depth is skewing the
   training data.
2. **If so, gen56 with less exploration** (or exploration only for Black), with
   everything else as gen55. It needs a plan and approval.
3. **Site:** add gen55 with a fitted rating, as gen54 was.
