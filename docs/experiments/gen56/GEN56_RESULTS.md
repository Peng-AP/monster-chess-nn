# Gen56 results (census-book starts + per-colour teachers), completed October 15, 2026

- **Plan:** `docs/plans/GEN56_PLAN.md`. Owner, October 12: "Yes, work on it".
  Owner, October 13, on option d: "Yes, try it, and then start gen 56".
- **Driver:** `tools/gen56_campaign.py`.
- **Run:** production from October 13, 18:30, to October 15, 11:02.
  - The first option-d rehearsal caught a model-preload bug in
    `tools/stateful_generation.py`: workers loaded only the first task's
    models.
  - The bug was fixed and tested, and the second rehearsal passed.
- **Evidence:** `benchmarks/gen56_program/gen56_20261013/production/`.
- **Nominee:** `models/candidates/bootstrap_main_gen_0056/arena_selected.pt`
  (epoch 24), SHA256 `935959b4…`.

**Nothing was promoted.**

| Rule | Eligible | Why |
|---|---|---|
| Current (`PROMOTION_RULE.md`) | **no** | Gate v4 against v29 fails on the White floor: −6.1 and −5.6 pp against a −5 pp floor. The held-out mean passes (97.7%). |
| Proposed per-colour rule (`GEN56_PLAN.md`; not adopted) | **yes** | All four per-colour cells are at or above v29's, and the held-out mean passes. |

**The caveat on both:**

- At 12,800 simulations, gen56's White loses almost every game to gen55's
  Black (0/2/78 W/D/L).
- In the same opening lines, gen55's White holds draws against that Black.
- gen56 failed the deep guard against gen55, its teacher, though it passed the
  deep guard against v29.

## Recipe

| Source | Games |
|---|---:|
| gen55 self-play, normal start (+ 400 forks; reanalysis 24k → 12k) | 1,400 |
| gen55 self-play from 70 census book lines | 1,400 |
| **Per-colour teachers** (book starts, both sides' policy targets; 350 games per pairing) | 1,400 |
| Pool: gen55 vs v29, gen54, gen53 and gen52 B, C, L, LR, R | 1,376 |
| Deep-value continuations (gen55, reference v29; ramped) | 768 roots × 2 |
| Self-play replay, gen49–gen55 | — |

1.9M tower, scratch, seed 3173. Training stopped early at epoch 24. The best
validation loss was 1.495, at epochs 9–14.

**The per-colour teacher games** (1,600 sims, 30 half-moves of exploration;
White's result):

| White | Black | W/D/L | White score |
|---|---|---|---:|
| gen55 | gen53 | 51/19/280 | 17.3% |
| gen55 | v29 | 54/30/266 | 19.7% |
| gen54 | gen53 | 49/27/274 | 17.9% |
| gen54 | v29 | 67/25/258 | 22.7% |

## Selection

**Screen against gen55** (200 games at 3,200 sims, deltas against gen55's
self-par):

| Epoch | Black | White |
|---|---:|---:|
| 24 | +24.3 pp | +19.2 pp |
| 15 | +0.8 | +24.7 |
| 16 | +14.3 | −9.2 |
| 14 | +18.3 | −29.2 |

**Deep probe** (against gen55 at 12,800 sims, 80 games) **and colour check**
(the census key book, 14 positions × 6 copies, against gen55 and v29):

| Epoch | Deep probe | Deep White Δ | Colour White | Colour Black | Deep guard |
|---|---:|---:|---:|---:|---|
| **24** | 40.6% | −17.2 pp | **22.8%** | 89.7% | fail |
| 15 | 45.0% | −17.2 pp | 19.1% | 82.9% | fail |
| 16 | 40.0% | −18.4 pp | 19.1% | 90.8% | fail |
| gen55 (teacher) | | | 16.8% | 83.9% | |

- No epoch passed the deep guard.
- The pre-declared fallback took the best colour-check White, epoch 24. It was
  also the best screen.

## Gate v4 against the teacher (gen55): FAIL (guard)

gen55's par on this engine: White 36.9% at 3,200 (95/105/200), 18.4% at
12,800 (0/59/101).

| Leg | Score | White W/D/L | Black W/D/L |
|---|---:|---|---|
| vs_bar @3,200 (400) | **73.2%** | 122/4/74 | 143/52/5 |
| vs_bar_confirm @3,200 (400) | **72.9%** | 114/7/79 | 153/42/5 |
| deep guard @12,800 (160) | 43.8% | **0/2/78** | 58/22/0 |

- **3,200 legs:** White +25.1 / +21.9 pp and Black +21.4 / +23.9 pp, both
  against gen55's self-par. This is the largest primary margin of the chain.
  gen55 scored 68.8% / 70.6% against gen54.
- **Deep guard:** White −17.2 pp against a band of −10, and aggregate 43.8%
  against a floor of 47.5%. The guard fails.

**Where the deep White games go** (first four half-moves; White's W/D/L):

| Line | gen55's White vs gen55 | gen56's White vs gen55 (deep probe) |
|---|---|---|
| e4 d4 …d5 e5 (either order) | 0/17/1 | 0/1/28 |
| e4 d4 …d5 Ke2 (either order) | 0/17/0 | 0/0/8 |
| d4 f4 (any) | 0/0/90 | not played |

- In the census main line, gen55's White holds a draw against gen55's Black at
  12,800. gen56's White loses it almost every time.
- At 3,200 the same model's White is +25 pp. **gen56's White gain does not hold
  up under deep search** against gen55's Black.
- gen56 dropped d4 + f4, which gen55's White lost 90/90.

## Gate v4 against the release (v29): FAIL (White floor)

v29's par on this engine: White 48.4% at 3,200 (7/373/20), 20.9% at 12,800
(1/65/94).

| Leg | Score | White W/D/L | Black W/D/L |
|---|---:|---|---|
| vs_bar @3,200 (400) | 71.1% | 14/141/45 | 200/0/0 |
| vs_bar_confirm @3,200 (400) | 71.4% | 17/137/46 | 200/0/0 |
| deep guard @12,800 (160) | 56.2% | 9/2/69 | 80/0/0 |

- **White:** −6.1 / −5.6 pp against the −5 pp floor. It fails by 1.1 and
  0.6 pp. gen55 missed by 13–21 pp, gen54 by 19–21.
- **Black:** won all 480 games against v29.
- **Deep guard:** passes. White is −8.4 pp against a band of −10.

## Per-colour instrument (the new primary measure)

Same book, sampling and sims as `docs/experiments/game_value/COLOUR_BASELINES.md`
(14 key positions × 20 copies, 8 sampled half-moves, 3,200 sims). 560 games
against each reference. Draws count half.

| Model | White vs gen55 | White vs v29 | Black vs gen55 | Black vs v29 | White mean | Black mean |
|---|---:|---:|---:|---:|---:|---:|
| **gen56** | 21.2% | **20.0%** | 86.4% | 91.6% | **20.6%** | 89.0% |
| gen55 | 21.7% | 15.5% | 78.3% | 88.6% | 18.6% | 83.4% |
| v29 (release) | 11.4% | 9.7% | 84.5% | 90.3% | 10.6% | 87.4% |
| gen54 | 16.2% | 18.8% | 83.0% | 81.6% | 17.5% | 82.3% |
| gen53 | 13.2% | 14.5% | 90.5% | 92.1% | 13.8% | **91.3%** |

**Paired differences** (position-weighted, pp, SE in brackets):

| gen56 minus | White vs gen55 | White vs v29 | Black vs gen55 | Black vs v29 |
|---|---:|---:|---:|---:|
| v29 | +9.8 (2.7) | +10.3 (2.4) | +2.0 (2.5) | +1.3 (1.8) |
| gen55 | −0.4 (2.5) | +4.5 (2.9) | +8.1 (2.2) | +3.0 (2.3) |
| gen53 | +8.0 (2.7) | +5.5 (2.7) | −4.1 (2.2) | −0.5 (2.0) |

- **gen56 has the best White measured**, and the only one at or above v29 in
  all four cells.
- Its Black is second to gen53's.
- gen55's Black lost or drew several lines against gen55's White. gen56 wins
  them: e1e2 e2e3 (47.5% → 100%), e2e3 f2f4 (55% → 92.5%), e2e4 e4e5
  (61% → 100%) and d4 f4 (50% → 90%).
- Two lines got worse against gen55's White: c4 d4 (66% → 25%) and c4 e4
  (74% → 45%). Each rests on 20–40 games; quote the totals, not the lines.

**Proposed rule**, against v29's cells with 3 pp tolerance:

- **Campaign's own scoring:** +7.6, +2.9, +10.2 and +1.3 pp, so all pass.
- **Draws counted as half:** +9.8, +1.9, +10.3 and +1.3 pp. This is the same
  verdict.
- **Scoring note:** the campaign's colour matches score a ±0.5 move-limit
  relabel as 0.75 or 0.25, not as a draw. Match.py, the census and the
  baselines count it as a draw. It moves cells by 0–2 pp. It affects selection
  equally for teacher and candidates, and it changes no verdict here. The
  drivers are pinned, so the fix goes into the next driver.

## Diagnostics (160 games each at 3,200, unless noted)

| Opponent | gen56 | White W/D/L | Black W/D/L | gen55 |
|---|---:|---|---|---:|
| B2 (held out) | **100.0%** | 80/0/0 | 80/0/0 | 93.1% |
| v27 (held out) | **95.3%** | 69/7/4 | 80/0/0 | 91.9% |
| gen49 | 75.6% | 15/52/13 | 80/0/0 | 74.1% |
| v28 | 81.2% | 50/6/24 | 77/0/3 | 82.2% |
| self @3,200 (200) | 49.5% | 0/57/43 | | |
| self @12,800 | 50.3% | 0/1/79 | | |

- **Held-out mean: 97.7%.** That is the best of the chain. The rule bar is
  91.7%, and the current-engine bar is 91.8% (v29 re-measured at 92.8%).
- gen53's v27 White hole (41/1/38) is closed: gen56's White went 69/7/4
  against v27.
- **Self-play at 12,800:** White 0/1/79. gen56's own Black beats its White at
  depth.

**Position probe** (the v27 endgame position, 10 games per side against v27):

- As White: 0/10/0.
- As Black: 0/0/10.
- gen55 won 10/10 as both colours.
- This is a regression on the one probe position.

**Value audit** (bias = target − prediction; positive means White-pessimistic;
MSE):

| Model | Bias (all) | MSE | Bias (strong games) | MSE |
|---|---:|---:|---:|---:|
| v29 | +0.036 | 0.091 | +0.017 | 0.080 |
| gen55 | +0.021 | 0.080 | +0.005 | 0.063 |
| gen56 | +0.028 | 0.085 | +0.009 | 0.069 |

gen56 is slightly more White-pessimistic and slightly less accurate than
gen55, but better than v29.

## Reading

**1. Option d did what it was meant to on the census positions.**

- gen56 is the first model at or above the release in all four per-colour
  cells.
- It has the best White measured, and it repaired gen55's Black holes against
  gen55's White.

**2. It is the strongest model at 3,200 by every normal-start measure here:**

- 73% against gen55;
- 71% against v29, with Black 400/400;
- the best held-out result.

**3. Its weakness is White at deep search.**

- Against gen55's Black at 12,800, it loses lines that gen55's White draws.
- It also loses its own self-play at 12,800 (White 0/1/79).
- Against v29 at 12,800 the guard passes, so the problem shows against the
  strongest Black, not every Black.

**4. The current rule's White floor at 3,200 is missed by about 1 pp.** Under
the proposed per-colour rule, gen56 is eligible.

## For the owner

**Promotion:**

- The current rule says no.
- The proposed rule says yes.
- The deep White result against gen55 is the reason for caution.
- This is the owner's decision. Nothing was promoted.

**Next-step candidates** (not started):

- Look at the e4 d4 …d5 e5 positions at 3,200 and at 12,800: where gen56's
  White choices differ from gen55's, and which one the deeper search prefers.
- Test whether book and teacher games at 1,600 sims teach lines that only hold
  at shallow depth.
- Fix the colour-match scoring in the next driver.
