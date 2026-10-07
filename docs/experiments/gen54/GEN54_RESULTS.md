# Gen54 results (teacher mix: gen53 + gen52 Arms R and LR), completed October 7, 2026

Plan: `docs/plans/GEN54_PLAN.md` (owner decision of October 5: "a mix of
teachers as well as the other proposed changes. 1 arm only"). Driver:
`tools/gen54_campaign.py`. Follow-up chain: `tools/gen54_followup_chain.py`.

- Production ran October 5 16:55 → October 6 20:24 as one managed job, with no
  failures.
- The follow-up ran October 6 20:24 → October 7 00:29.

Evidence: `benchmarks/gen54_program/gen54_20261005/{production,vs_release}/`,
`benchmarks/position_probe_gen54/` and `benchmarks/value_colour_audit_gen54/`.

**Nothing was promoted. gen54 is not eligible:** it fails gate v4 against v29
on the White floor.

Nominee: `models/candidates/bootstrap_main_gen_0054/arena_selected.pt`
(epoch 13), SHA256 `1f5d53ba…4eae35`.

## Recipe

| Source | Games | Notes |
|---|---:|---|
| gen53 (main teacher) | 2,800 + 400 forks | 1,600 simulations, 30-ply exploration; reanalysis 24k → 12k at 12,800 |
| gen53 deep-value | 768 roots × 2 at 6,400 | Reference v29; ramped game-result labels |
| gen52 Arm R | 700 | Extra increment (policy and value) |
| gen52 Arm LR | 700 | Extra increment (policy and value) |
| Hole-scan pool | 1,204 | gen53 against v29 and gen52 Arms A, B, C, L, LR, R; only gen53's moves carry policy weight |

The value-weight share of each source in the training replay, computed from
`value_weights.npy`:

- **deep-value 22.6%**: gen53's 11.2% and gen54's 11.4%;
- Arm R 2.0%, Arm LR 2.0%, pool 3.2%;
- the self-play generations gen47–gen54, the rest (gen47 13.7%, the others 7.8–8.5% each).

The same computation gives gen53's replay 23.5% deep-value. (GEN53_RESULTS
quotes 21.9% by another count.)

The rest was unchanged: the 1.9M tower trained from scratch, seed 3173, and
replay gen47–gen54. Training stopped early around epoch 20.

## Selection

The rule picks the best screen rank among the epochs that pass the deep probe
at 12,800 against gen53. Only epoch 13 was eligible.

| Epoch | Deep probe vs gen53 | White W/D/L | Black W/D/L | White vs par |
|---|---:|---|---|---:|
| **13 (nominee)** | 70.0% | 0/32/8 | 40/0/0 | −8.1 pp |
| 22 | 68.1% | 0/29/11 | 40/0/0 | −11.9 pp |
| 4 | 61.9% | 0/20/20 | 39/1/0 | −23.1 pp |

## Gate v4 against the teacher (gen53): PASS

| Leg | Score | White W/D/L | Black W/D/L | Distinct games only |
|---|---:|---|---|---:|
| vs_bar @3,200 (400) | **79.9%** | 68/113/19 | 190/10/0 | 64.4% |
| vs_bar_confirm @3,200 (400) | **78.5%** | 64/113/23 | 187/13/0 | 60.8% |
| deep guard @12,800 (160) | 70.0% | 1/63/16 | 79/1/0 | 66.4% |

Against gen53's self-par:

- **3,200 legs:** White +17.4 / +15.4 pp, Black +42.4 / +41.6 pp.
- **Deep guard:** White −7.5 pp, Black +47.5 pp.

## Diagnostics (160 games each at 3,200, unless noted)

| Opponent | gen54 | White W/D/L | Black W/D/L | gen53 |
|---|---:|---|---|---:|
| B2 (held out) | 92.2% | 76/4/0 | 59/21/0 | 98.4% |
| **v27 (held out)** | **92.2%** | 59/17/4 | 80/0/0 | **78.1%** |
| gen49 | 80.6% | 23/53/4 | 79/1/0 | 89.7% |
| v28 | 71.9% | 24/44/12 | 58/22/0 | 84.7% |

**Held-out mean (B2, v27): 92.2%**, which meets the rule's 91.7%. gen53 had 88.3%.

**Self-play (actual colours):**

| Simulations | Games | White wins | Draws | Black wins | White score | gen53 White score |
|---:|---:|---:|---:|---:|---:|---:|
| 3,200 | 200 | 1 | 30 | 169 | 8.0% | 45.25% |
| 12,800 | 160 | 0 | 134 | 26 | 41.9% | 48.4% |

**What the drops against the old models are made of** (from the journals):

- **As Black** gen54 never lost. Its draws are repetitions in which it held
  at least a rook more material and White still had pawns:
  - 21 against B2, only 4 distinct games (one game sampled 10 times);
  - 22 against v28, 17 distinct games.

  The forced-capture solver did not run in these, because it only starts
  when White has at most one piece besides the king. **Whether Black was
  winning in these positions is not established.** A material lead is not a
  forced win against a double-moving king. Measuring this is the first next
  step below.
- **As White** gen54 won much less often: 23 vs 55 wins against gen49, and 24
  vs 37 against v28. Its draws are mostly repetitions in which gen54 was
  reduced to a bare king and the older engine did not convert (27 and 29
  distinct games).

## Gate v4 against the release (v29): FAIL

v29's par was measured afresh. The par from gen53's gate was rejected by the
runtime-identity check, as expected. v29's self-play White score: 49.5% at
3,200, 27.8% at 12,800.

| Leg | Score | White W/D/L | Black W/D/L | Distinct games only |
|---|---:|---|---|---:|
| vs_bar @3,200 (400) | 64.75% | 9/103/88 | 197/3/0 | 63.0% |
| vs_bar_confirm @3,200 (400) | 63.9% | 4/105/91 | 198/2/0 | 60.3% |
| deep guard @12,800 (160) | 58.1% | 0/65/15 | 41/39/0 | 59.4% |

Deltas against v29's par:

| Leg | White | Black | Floor (−5 pp) |
|---|---:|---:|---|
| vs_bar @3,200 | −19.25 pp | +48.75 pp | **White fails** |
| vs_bar_confirm @3,200 | −21.25 pp | +49.0 pp | **White fails** |
| deep guard @12,800 | +12.8 pp | +3.4 pp | Passes |

- **The primary verdict is FAIL** on the White floor in both 3,200 legs.
  The guard passes.
- gen54's White loses about 45% of its games to v29's Black at 3,200.
  gen53's White against v29 was 3/182/15.
- At 12,800 gen54's White is fine against v29 (+12.8 pp): the weakness is at
  the ordinary depth.

## The v27 position: FIXED

This is the endgame that cost gen53 32 of 32 losses to v27. From it, with 10
games per side against v27:

- **gen54 as White wins 10/10** (gen53 lost 10/10; Arm R and v29 won 10/10);
- gen54 as Black draws 10/10.

The v27 diagnostic agrees: gen54's White lost 4 games to v27, against gen53's 33.

## Value audit: no White pessimism

Prediction minus target, with the 95% interval, on strong games:

| Model | Bias | Interval | MSE |
|---|---:|---|---:|
| gen54 | −0.0055 | −0.0131 to +0.0027 | **0.0664** |
| gen53 | −0.0059 | −0.0137 to +0.0021 | 0.0704 |
| v29 | +0.0166 | +0.0090 to +0.0243 | 0.0804 |

gen54's value head is as unbiased as gen53's and the most accurate of the
three. **The White weakness is not a value bias.**

## Rating (for the site, provisional)

The rating is gen54's own matches at 3,200, fitted with the ladder ratings
held fixed:

| Opponent | Fixed rating | gen54's score | Games |
|---|---:|---:|---:|
| v29 | 2450 | 64.3% | 800 |
| gen53 | 2555 | 79.2% | 800 |
| v28 | 2326 | 74.4% | 320 (old and new engine) |
| gen49 | 2301 | 80.6% | 160 |
| B2 | 2171 | 95.3% | 320 (old and new engine) |
| v27 | 2064 | 92.2% | 160 |

**Result: 2641** (likelihood 95% range 2625–2658). The data is non-transitive:

- the v29 result alone implies about 2550;
- the gen53 result alone implies about 2785.

A round-robin extension, like gen53's, would settle the rating. The site lists
gen54 as experimental with this provisional rating.

## Promotion eligibility (`docs/protocols/PROMOTION_RULE.md`)

| Criterion | gen54 | Required | Met |
|---|---:|---|---|
| Gate v4 PASS vs release v29 | FAIL (White floor) | PASS | **no** |
| Held-out mean (B2, v27) | 92.2% | ≥ 91.7% | yes |

**Not eligible.** v29 remains the release, and gen53 remains on the site as
the strongest experimental model.

## Reading

- gen54 is the strongest Black yet:
  - it never lost as Black in any match;
  - it scored 99% as Black against v29.
- It closed gen53's v27 hole.
- Its White at 3,200 is the problem. It lost 169 of 200 to its own Black and
  about 45% of its games to v29's Black, and it wins less against old models.
- Beating gen53 by 79% hid this, because gen53's Black did not punish it the
  way v29's does.
- The value audit rules out a pessimism bias. The White weakness is in what
  it plays, not in how it evaluates.
- Single arm: the teacher mix, the pool and the labels changed together, so
  the cause cannot be attributed from this run.

## Next steps (proposed October 7; the owner decides)

1. **Measure the drawn endings** (CPU only). Run the forced-capture solver
   at depth 6–7 over the final stretch of every distinct draw, and sort them
   into proven-and-missed wins, no forced win within the horizon, and unknown.
   This decides whether the engine-side fix (repetition awareness and a wider
   solver trigger) is worth an engine-identity change.
2. **Locate gen54's White losses against v29** (no GPU). Check whether the 179
   losses are concentrated in a few openings, compare with gen53's White on
   the same seeds, and check for shared lines with its self-play and v28
   losses.
3. **Data audit** (no GPU). Compare the White-win and White-loss shares of each
   value source with gen53's.
4. **Epoch check** (GPU, about 1–2 h; needs approval). Play the other screened
   epochs as White against v29, as a diagnostic only.
5. **Gen55 options** after steps 1–4:
   - select and gate against v29 as well as the teacher;
   - put v29 in the pool;
   - run two arms, so the changes can be attributed.
