# Gen52 results — completed September 28, 2026

Campaign `tools/gen52_campaign.py` (plan `docs/plans/GEN52_PLAN.md`).
Production 2026-09-27 03:08 → 2026-09-28 18:03 (39 h), one managed job, no
failures or restarts. Evidence
`benchmarks/gen52_program/gen52_20260927/production/`; every journal
replay-audited. **Nothing was promoted.**

## Bottom line

1. **Neither arm passed gate v4 against the teacher** (gen51's deep-value
   nominee), so **gen52 is not an improvement**, and by the pre-declared rule 1
   in `docs/plans/GEN53_PLAN.md` **gen53 does not launch automatically**.
2. **Both arms got weaker as White** against the teacher (Arm A −26 pp, Arm B
   −13 pp against the teacher's self-par). Both are stronger or equal as
   Black. The teacher itself is extremely drawish in self-play (383 of 400
   draws at 3,200).
3. **The owner's model pool helped where it was meant to:**
   - Arm B's held-out mean (B2, v27) is 91.4% vs Arm A's 80.9%, about level
     with the teacher's 92.7%.
   - Arm B is much sturdier as White against v28 and gen49.
   - Counting each opening once, Arm B scores 53.6% / 58.0% against the teacher.
     It fails only the frequency-weighted gate, because its losses sit in a
     few openings the teacher repeats.
4. **Arm A regressed on the held-out B2** (65.9%, mostly draws, vs 99% for
   gen51). The draw-heavy deep-value hypothesis below is plausible but
   untested.

## What changed from gen51

- Teacher: gen51's deep-value nominee (by the pre-declared rule).
- Self-play recipe unchanged: 30-ply exploration, 400 forks, reanalysis at
  12,800.
- Deep-value continuations with this teacher (768 roots × 2 at 6,400); gen51's
  deep-value increment rolls forward.
- **Arm B only:** 1,200 teacher-vs-pool games (v28, gen49, gen48, v26).
  **B2 and v27 held out.**

Data checks:

- gen52 self-play (1,600 sims): White 21.3%, Black 54.1%, draws 24.6%, so
  outcomes were not draw-starved.
- Diversity held: 157,579 distinct positions, 2.82 value rows per position,
  middlegame top-100 share 3.4% (gen50: 3.77 and 23.6%).
- Arm A replay 4,895,272 rows (gen45–gen52 plus both deep-value sources);
  Arm B adds 163,136 pool rows.

Teacher par (self-play):

- At 3,200: White 49.4% (6 W / 383 D / 11 L).
- At 12,800: White 18.8% (1 W / 58 D / 101 L).

## Selection (vs the teacher)

| Arm | Epoch | Screen @3,200 (200 g) | Probe @12,800 (80 g) | Nominee |
|---|---|---:|---:|---|
| A | 16 | 66.5% (W −13.8, B +46.8 pp) | 41.25% | |
| A | 28 | 65.25% | 36.9% | |
| A | 18 | 60.75% (W −25.7 pp) | 46.9% | **yes (fallback: none passed)** |
| B | 8 | 47.25% (W −12.8, B +7.3 pp) | **54.4%** | **yes** |
| B | 10 | 44.0% | 46.25% | |
| B | 16 | 35.75% | 48.1% | |

## Gate v4 against the teacher

| Leg | Arm A (epoch 18) | Arm B (epoch 8) |
|---|---:|---:|
| vs_bar @3,200 (400) | 59.75% (W 23.3 / B 96.3) | 46.88% (W 34.7 / B 59.0) |
| vs_bar_confirm @3,200 (400) | 60.25% (W 22.8 / B 97.8) | 47.88% (W 37.2 / B 58.5) |
| Endpoint-unique, both legs | 46.1% / 49.0% | **53.6% / 58.0%** |
| deep_guard @12,800 (160) | 43.75% (FAIL) | 51.25% (pass) |
| **Verdict** | **FAIL** (White floor, guard) | **FAIL** (aggregate ≤ 50%, White floor) |

Combined 800-game colour deltas against the teacher's 3,200 self-par:

- Arm A: White −26.4 pp (−29.0 to −23.7), Black +46.4 pp (+44.8 to +48.0).
- Arm B: White −13.4 pp (−16.1 to −10.7), Black +8.1 pp (+6.0 to +10.2).

## Diagnostics (3,200 simulations, 160 games unless noted)

| Opponent | Arm A | Arm B | gen51 deep-value (teacher) |
|---|---:|---:|---:|
| **B2 (held out)** | 65.9% (W 29/49/2, B 24/56/0) | 87.8% (W 57/17/6, B 70/10/0) | 99.06% |
| **v27 (held out)** | 95.9% (W 71/7/2, B 79/0/1) | 95.0% (W 70/4/6, B 80/0/0) | 86.25% |
| **Held-out mean** | **80.9%** | **91.4%** | **92.7%** |
| gen49 (pool member for Arm B) | 61.9% (W 3/54/23, B 60/18/2) | 60.6% (W 15/55/10, B 37/35/8) | 75.6% |
| v28 (pool member for Arm B) | 58.75% (W 3/27/50, B 75/5/0) | 70.9% (W 11/48/21, B 77/3/0) | 74.9% (gate) |

Actual-colour self-play (White W/D/L):

| | @3,200 (200 g) | @12,800 (160 g) |
|---|---|---|
| Arm A | 1 / 131 / 68 (33.3%) | 2 / 123 / 35 (39.7%) |
| Arm B | 1 / 137 / 62 (34.7%) | 39 / 57 / 64 (42.2%) |
| Teacher | 6 / 383 / 11 of 400 (49.4%) | 1 / 58 / 101 (18.8%) |

**The gen51 White hole (e4+d4 …d5 c4+Ke2):** both gen52 arms moved away from
it. Arm A now meets …d5 with e5 + Ke2, which draws against gen49 (20 D /
1 L), and opens more often with e4-e5. Against that, gen49's …f6 produced
most of Arm A's White losses (0 W / 19 D / 16 L). The hole moved rather than
closed.

## Why gen52 did not improve: evidence and hypotheses

- **Deep-value weight nearly doubled.** gen51's deep-value source rolled
  forward alongside gen52's new one, each at value weight 4. Deep-value rows
  carry **22.3%** of Arm A's value-training weight (10.9% + 11.4%), against
  13.0% in gen51's deep-value arm. About a third of those rows are draw labels
  (30.1% / 36.7%).
- **The continuation data itself was not much drawier** (gen52: 26.6% of
  games drawn, gen51: 23.6%). So the change is in *how much* of it trains
  the value head.
- **Hypothesis, untested:** the larger draw-heavy strict-label share pulled
  the value head toward "drawn", producing draw-prone play (Arm A drew 105 of
  160 games against B2) and a passive White. The pool data diluted it (Arm B
  better against B2).
- **Alternative or additional:** a very drawish teacher whose Black beats its
  own White at depth (teacher self at 12,800: White 18.8%) passes that
  White-pessimism to its students through self-play value labels.

## Decisions and recommendations

- **Rule 1 (GEN53_PLAN):** no gen52 arm passed gate v4, so **gen53 is not
  launched**. The owner is notified.
- **Promotion:** unchanged. gen51's deep-value nominee remains the strongest
  candidate on the evidence (passed gate v4 vs v28, 99% vs B2, beat the
  gen51 control 67.8% at 12,800) and is the recommended playtest.
- **Options for the owner (none started):**
  1. **Cheap causal test (about 10–11 h, no new games):** retrain gen52 Arm B
     with the deep-value share capped near gen51's 13% (drop the rolled-forward
     gen51 source, or weight 2). Screen and gate it against the same teacher.
     Tests the hypothesis directly.
  2. **gen53 with the lessons applied** (about 36 h): teacher gen51
     deep-value; pool kept (it helped on held-out opponents); deep-value
     source from the current generation only (no roll-forward), or weight 2.
  3. **Release decision first:** playtest gen51's deep-value nominee, and
     decide whether it becomes v29, before further generations.
