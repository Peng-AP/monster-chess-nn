# Overnight run — 2026-08-01/02

Active report for the unattended session that began 2026-08-01 ~22:20. Written
as results land; every claim cites its artifact in `benchmarks/`.

**Nothing here promotes anything.** Promotion needs the owner's playtest, and
no gate threshold was moved.

---

## 1. What ran

| # | work | status |
|---|---|---|
| 1 | M3 referees | done — ramp's optimism is miscalibration, §2 |
| 2 | D1 value-weight pipeline | done, §3 |
| 3 | D2 Merge-K / Merge-B corpora | done, §4 |
| 4 | Phase 3 corpus ladder: control → O → K → B | **3 of 4 beat the bar**, §5 |
| 5 | Cliff conversion, K-vs-B, calibration | done, §6 |
| 6 | M4 cliff-vs-sims curve | done — **the cliff is search-limited**, §7 |
| 7 | D3 cliff self-play generation | launched, time-boxed, §8 |

### The three findings that matter

1. **Data moved what architecture never did.** A control arm on `combined_v17`
   reproduced the historical 0.300 Black leg against ramp exactly; adding your
   27 uncorpused games took it to 0.450, and ps_monster to 0.650–0.700. The
   v18 programme's architecture and duplication levers were measured nulls;
   the corpus was the variable all along.
2. **ps_monster's value labels buy game strength and cost calibration.** K and
   B convert identically while B's value head is 2.4× more miscalibrated.
3. **The cliff is now search-limited, not knowledge-limited.** B converts
   0.36 → 0.84 as its search grows 200 → 1600. The knowledge is in there; 400
   sims does not extract it.

**Nothing is promoted.** Two arms meet both of the owner's criteria and are
waiting on his playtest.

---

## 2. M3 — ramp's Black-optimism is miscalibration. Answered.

§4.5 asked whether ramp's unusually optimistic value for Black in
promotion-defense positions is insight or error. It is error, and the
experiment that settles it is a controlled one: **each model plays Black
itself, from the same 400 positions, against the same White.**

| model | mean predicted (Black POV) | mean realized | bias | Black conversion |
|---|---|---|---|---|
| ramp | **−0.129** | −0.647 | **+0.518** | **20.75%** |
| v17 | **−0.537** | −0.667 | **+0.130** | **19.75%** |

The two models convert at the same rate — 1 point apart on n=400, about half a
standard error — while their beliefs sit 0.41 apart. Ramp is not seeing
something v17 misses. It converts no better and simply believes it is better
off, with four times v17's calibration error.

Artifacts: `benchmarks/promotion_defense_outcomes_rampblack_20260801_214726.json`,
`..._v17black_*.json`.

**Why the first attempt at this was worthless, and worth remembering.** The
referee originally specified was heuristic-vs-heuristic. It converts Black in
**3 of 400** positions (0.75%), so every predicted band mapped to realized
−1.00 and the table showed all four models wildly optimistic with ramp worst.
That reading was unusable: a referee too weak to convert cannot distinguish "the
model is wrong" from "the model is right and the referee cannot play." Only
when the model plays its own Black does the comparison mean anything. Kept as
`promotion_defense_outcomes_20260801_210716.json`.

**Caveat that survives the result.** Heuristic-White is weak, so ~20%
overstates conversion against real opposition and the *magnitude* of both
biases is a floor, not a measurement. The comparison between models is sound —
identical positions, identical White. The strong-White ceiling
(`--playout-white ramp --playout-black ramp`) was queued and dropped when the
direction was settled; it can only push realized value further below prediction.

**Consequence for Phase 2.** Ramp stays a legitimate *opponent* — its games are
labelled by outcome, not by its beliefs — but **its search values are not
trustworthy training signal**, so D3's cliff self-play should not take value
targets from ramp. This was the decision M3 existed to make.

---

## 3. D1 — the value-weight pipeline

`data_processor.py` emits `value_weights.npy` beside `policy_weights.npy`;
`train.py` applies it to the scalar value loss **and** the WDL cross-entropy,
so "teach policy, not value" cannot leak back in through `--value-head wdl`.

Two guarantees, both tested (`tests/test_value_weights.py`, 15 tests):

1. **Default is the old behaviour, exactly.** Verified beyond unit tests: the
   M1 command re-run after the change reproduces its recorded first three
   epochs **digit for digit** (train 3.9353 / 2.6762 / 2.1442…, val 3.0664 /
   2.8073 / 2.8078). Had this drifted, every v17/v18 number on record would
   have stopped being comparable.
2. **Weight 0 is zero value gradient** — not "downweighted". A masked record
   cannot change the loss value, cannot move the gradient, and an all-masked
   batch still backpropagates without error.

Batch tuples have now grown twice, so `_unpack_loader_batch` disambiguates
length-5 batches on dtype (weights are float, WDL labels are long) and every
historical shape still unpacks. Five pre-existing tests were updated for the
new arity.

---

## 4. D2 — the knowledge-vs-belief fork, as a single variable

`tools/merge_source.py` merges a raw source into a corpus and can stamp
`value_weight` on every merged record. Both arms carry the *same* 829
ps_monster games; the only difference is whether they teach the value head.

| | positions | train | policy masked | value masked |
|---|---|---|---|---|
| `combined_v19_K` | 243,542 | 195,292 | 64,956 | **87,862** (every ps record, mirrored) |
| `combined_v19_B` | 243,542 | 195,292 | 64,956 | **0** |

Identical in every other respect. The stamp has a large, real effect on
training — one epoch each, same seed:

| arm | value loss | policy loss |
|---|---|---|
| K | **0.1386** | 3.5466 |
| B | **0.2386** | 3.5178 |

So the A/B is not a null by construction.

**A confound to remember when reading offline metrics:** K and B share a
validation set that contains ps positions, and K is never trained to predict
them. B's value MAE and sign accuracy are therefore flattered mechanically —
after one epoch B already reads val sign 83.4/83.4 against K's 77.9/79.3. This
is law 7 again (offline metrics and play strength are decoupled, demonstrated
four times). **The gate decides; these numbers do not.**

---

## 5. Phase 3 — the corpus ladder

Each rung adds exactly one thing, so a difference can be attributed:

| arm | corpus | what it adds |
|---|---|---|
| control | `combined_v17` | the v18-era corpus |
| O | `combined_v19_base` | + 27 owner games |
| K | `combined_v19_K` | + ps_monster, policy only |
| B | `combined_v19_B` | + ps_monster, full value |

Recipe frozen across all four (ramp labels r50h60, scalar head, 15ch, seed 42,
30 epochs / patience 10 — M1 measured no headroom). Gate is `tools/gate.py`
unchanged: bar is ramp, per-side floor 0.40, passing arms replay the bar leg on
a fresh opening seed.

**Training, all four (same recipe, same seed):**

| arm | best epoch | stopped | train positions |
|---|---|---|---|
| control | **24** | cap 30 | 116,324 |
| O | 8 | early, 18 | 124,924 |
| K | 5 | early, 15 | 195,292 |
| B | 6 | early, 16 | 195,292 |

`v19_control` reproduces M1's best epoch 24 and selection value −1.2594 exactly,
which is a second confirmation that D1 left training untouched.

The merged arms converge far earlier — more positions per epoch, so fewer
epochs — and then early-stop. Their `best_selection_value` is *lower* than
control's (K −1.0994, B −1.1139, O −1.2217 vs control −1.2594), but **that
comparison is not meaningful**: each arm's validation set is drawn from its own
corpus, so control is scored on v17 positions only while K and B are scored on
a set containing 43,939 ps records. Different exams, not different grades. The
gate is the only cross-arm comparison in this report that is like-for-like.

### Gate results — three arms passed, and the ladder is monotone

Per-side totals across every leg, and the pooled bar legs (`vs_ramp` +
`vs_ramp_confirm`, 40 games per side), which is where §12 says to read:

| arm | verdict | White total | Black total | **vs ramp, pooled W / B** |
|---|---|---|---|---|
| control | **FAIL** | 0.700 (50) | 0.440 (50) | 0.550 / **0.300** |
| O | PASS | 0.807 (70) | 0.500 (70) | 0.775 / 0.450 |
| K | PASS | 0.800 (70) | 0.671 (70) | 0.725 / **0.700** |
| B | PASS | **0.900** (70) | **0.686** (70) | **0.875** / 0.650 |

Noise floor: SE 0.112 on a 20-game leg, **0.079** on a pooled 40-game bar leg.

**The control reproduces the historical failure exactly.** Its Black leg
against ramp is **0.300** — the same 0.30 the v18 gap, spatial and dup1 arms
died on (§4.1, §4.2). Same corpus family, same number. That is the strongest
evidence the setup is measuring what it claims: the ladder's baseline rung
lands precisely where the record says it should.

**Reading the rungs against the noise floor** (Black vs ramp, n=40 each):

| step | change | Δ | in SE |
|---|---|---|---|
| control → O | + 27 owner games | +0.150 | 1.9 |
| O → K | + ps_monster, policy only | +0.250 | 3.2 |
| O → B | + ps_monster, full value | +0.200 | 2.5 |
| **K vs B** | value labels on/off | **+0.050** | **0.6** |

1. **The owner's 27 games moved the deciding leg on their own** — 0.300 → 0.450
   at 1.9 SE. Suggestive rather than proven, but in the right direction and it
   was enough to flip the verdict from FAIL to PASS.
2. **ps_monster is the big lever, at 2.5–3.2 SE.** HANDOFF called it "the only
   untried lever". It was worth trying: data moved what architecture (§4.2, a
   measured null) and duplication (§4.1) never did.
3. **K vs B is a null at 0.6 SE.** The knowledge-vs-belief fork is *not*
   resolved — the belief worry did not materialise (B is not worse) and the
   knowledge advantage did not either (K is not better). The honest statement
   is that ps_monster helps a great deal either way, and whether its outcome
   labels reach the value head does not detectably matter.

Every arm faced identical openings on the corresponding leg (seed derived per
leg index), so these comparisons are paired; and the passing arms cleared the
bar twice, on two independent opening sets.

**What this does not yet establish.** The gate's opening phase is 16 sampled
plies, and ps_monster is 829 games of human openings — so part of the gain may
be opening knowledge rather than the pawn-phase conversion the campaign is
actually about. The cliff measurement in §6 is what separates those, and it is
the second of the owner's two criteria for a candidate reaching him.

Artifacts: `benchmarks/gate_v19_{control,O,K,B}_*.json`,
`benchmarks/phase3_summary_20260802_023748.json`.

---

## 6. The cliff moved — and it is not just opening knowledge

The gate could have been won on openings alone (16 sampled plies, and
ps_monster is 829 games of human openings). This separates them: every model
plays **Black** from the same 150 pawn-phase starts against the same heuristic
White. No opening is involved — these are mid-game positions.

| model as Black | conversion | mean realized (Black POV) | mean plies |
|---|---|---|---|
| v17 | 0.280 | −0.533 | 104 |
| control | 0.327 | −0.457 | 103 |
| ramp | 0.347 | −0.417 | 112 |
| K | **0.473** | −0.150 | 100 |
| B | **0.527** | −0.023 | 117 |

SE is 0.041 per model, so 0.058 on a difference:

| comparison | Δ | in SE |
|---|---|---|
| B − ramp | +0.180 | **3.1** |
| K − ramp | +0.126 | **2.2** |
| B − control | +0.200 | 3.4 |
| B − K | +0.054 | 0.9 |
| control − v17 | +0.047 | 0.8 |

So the pawn-phase gain is real and it is not an opening artifact. B converts
positions ramp loses, from identical starts against an identical opponent.

**Two things this does not say, and both matter.**

1. **These absolute rates are not law 1's 7–14%.** The deck is harvested from
   games Black *won* (`--require-black-win`), so it over-represents winnable
   positions by construction, and heuristic-White is weak. Both inflate every
   number in the table. The deck exists to compare models on identical
   positions, which it does fairly; it is not an estimate of how often Black
   converts in general, and it must not be quoted as one.
2. **B vs K is still a null** (0.9 SE here, 0.6 SE at the gate). Two
   independent measurements agreeing that they cannot be separated is worth
   more than either alone — but it is agreement on "cannot separate", not on
   "equal".

Artifacts: `benchmarks/promotion_defense_outcomes_cliff_*.json`.

### 6.1 K vs B, decided by playing them against each other

Comparing two models through their scores against a third is weak. A direct
match is not, and it separates them where the gate and the cliff deck both
returned nulls — **B beats K 0.65–0.35 over 40 games**, and B is better on
*both* sides:

| | as White | as Black |
|---|---|---|
| B | **0.95** | **0.35** |
| K | 0.65 | 0.05 |

(Each figure is against the other model's opposite side.) Note both models are
far stronger as White than Black even against each other — the standing White
bias, unchanged.

This does not overturn §5's null; it refines it. K and B are close enough that
40 games against ramp could not order them, and 40 games against *each other*
could. **B is the leading candidate.**

Artifact: `benchmarks/match_v19_K_vs_v19_B_20260802_041900.json`.

### 6.2 A flaw in the leading candidate: B is the most optimistic model yet

The promotion-defense probe, same 400 positions as M2:

| model | capture rate | mean value (Black POV) |
|---|---|---|
| control | 0.593 | −0.408 |
| K | 0.605 | −0.154 |
| **B** | 0.588 | **+0.028** |
| _ramp, for reference_ | 0.585 | −0.129 |

**B is more optimistic about Black than ramp was** — and M3 established that
ramp's optimism was miscalibration, not insight. That is exactly the pattern
that would show up in a playtest as "it thinks it is fine when it is not."

### 6.3 The D2 fork, answered: the value labels buy strength and cost calibration

Turning those predictions into calibration errors the way M3 did — each model
plays Black itself, same deck, same White:

| model | predicted | realized | **bias** | conversion |
|---|---|---|---|---|
| ramp | −0.129 | −0.647 | +0.518 | 0.207 |
| v17 | −0.537 | −0.667 | +0.130 | 0.198 |
| **B** | +0.028 | −0.255 | **+0.283** | **0.405** |
| **K** | −0.154 | −0.273 | **+0.119** | 0.395 |

Read carefully, because the naive reading of §6.2 was wrong:

* **B's optimism is largely earned.** It converts 0.405 against ramp's 0.207 —
  roughly double — so a much higher value is partly justified. Its bias is
  +0.283, about half ramp's.
* **But K converts the same** — 0.395 vs 0.405, a difference of 0.01 on n=200
  against SE ≈ 0.035, indistinguishable — **with less than half the
  calibration error** (+0.119 vs +0.283). K is the best-calibrated model
  measured, marginally better even than v17, while converting twice as often.

Same positions, same opponent, same conversion, very different beliefs. That
is exactly the M3 logic applied to K vs B, and it isolates the effect of the
value labels: **taking ps_monster's outcome labels into the value head does not
improve conversion; it inflates optimism.**

So §7.1's fork has an answer, and it is not the clean one either side
predicted:

| | K (policy only) | B (full value) |
|---|---|---|
| beats ramp at the gate | yes, 0.7125 pooled | yes, 0.7625 pooled |
| Black leg vs ramp | **0.700** | 0.650 |
| head-to-head | loses 0.35 | **wins 0.65** |
| cliff conversion | 0.473 | **0.527** |
| promotion-defense conversion | 0.395 | 0.405 |
| **calibration error** | **+0.119** | +0.283 |

**B wins more games today; K believes the truth.** Which matters more is an
owner call, and there is a specific reason to think it is not obvious: ramp was
rejected for *play* faults — gifting pawns in the opening, missing a
promotion — of the kind an over-optimistic value head produces. The D1 pipeline
that makes K expressible is therefore load-bearing, not merely available.

Artifacts: `benchmarks/promotion_defense_outcomes_pd_{B,K}_*.json`.

---

## 7. M4 — the cliff is search-limited

B plays Black from the same 100 cliff starts while **only Black's search
varies**; White is fixed at heuristic@400. (The probe originally applied one
`--sims` to both players, which would have strengthened White alongside Black
and made the curve meaningless; it now takes `--white-sims`/`--black-sims`.)

| Black sims | conversion | SE |
|---|---|---|
| 200 | 0.360 | 0.050 |
| 400 | 0.520 | 0.050 |
| 800 | 0.670 | 0.050 |
| 1600 | **0.840** | 0.050 |

+0.48 across an 8× range, monotone, ~10 SE. By the directive's own reading
rule this is the "knowledge is present, search-limited" branch: **the data
offensive should now target value sharpness — arm S, cliff self-play — rather
than more policy teaching.**

**What the curve does not say.** White is held at 400 sims while Black climbs
to 1600, so part of the rise is simply Black outsearching White. It does not
mean B at 1600 beats a *strong* White at 1600. The finding is about where B's
ceiling sits relative to what 400 sims extracts, which is what "search-limited"
means and what decides the next lever.

**A side observation worth checking.** The standing rule is that the owner
should play at 800 sims and never more, because the model was "measurably worse
at 2000–5000". B shows no such pathology up to 1600 — it improves monotonically
throughout. That warning was measured on v17-era models; it may not transfer to
these arms. Untested above 1600, so the 800-sim playtest rule stands until
someone measures it.

Artifacts: `benchmarks/promotion_defense_outcomes_m4_B_s{200,400,800,1600}_*.json`.

## 8. D3 — cliff self-play, launched and time-boxed

Deck: `data/start_fens/cliff_starts_v2.jsonl`, 412 wP≥3 starts from games Black
won (200 ps_monster / 212 owner, wP 3 and 4 evenly split). An uncapped sample
came out 89% ps_monster and buried the owner's own conversions, which are the
existence proof the deck is for; `--cap-per-source` fixes it.

Generation is **v17 vs v17 at 400 sims**, into
`data/raw/nn_v19_cliff_selfplay`. Two deliberate choices:

* **v17, not ramp, produces the values** — M3 measured ramp at +0.518
  calibration error, so its search values are not trustworthy training signal.
  v17 is the best-calibrated model available (+0.130).
* **400 sims is a time-box, not the right answer.** The directive specifies
  1600 and §7 has just shown why sharper search matters; 1600 would need ~4 h
  and the window was ~1 h. **A future run should regenerate at 1600.**

## 9. Where to pick this up

**Two candidates are ready for your playtest.** Both clear the gate twice on
independent openings and both convert the pawn phase far better than ramp.
They are not interchangeable:

* **`models/candidates/v19_B`** — the stronger player. Beats ramp 0.7625
  pooled, beats K head-to-head 0.65, cliff conversion 0.527. Carries a +0.283
  value bias.
* **`models/candidates/v19_K`** — the honest one. Beats ramp 0.7125 pooled with
  the **better Black leg (0.700 vs 0.650)**, same promotion-defense conversion
  as B, and less than half the calibration error (+0.119).

Play B first if you want to know whether the ceiling moved; play K if you want
to know whether the *judgement* improved. Given ramp was rejected for faults an
optimistic value head produces, K may survive your eye better than its gate
numbers suggest. **At 800 sims per the standing rule** — though see §7, that
rule was measured on v17-era models and B shows no high-sim pathology to 1600.

**The next lever is arm S**, not more policy data. §7 says the knowledge is
already in there and search extracts it, so the payoff is in sharper value
targets. D3's generation is the first step; regenerate at 1600 sims when there
is a night for it.

**The question I could not answer and you can.** Ramp trained on
`combined_v16`; every v18 arm that lost to it trained on `combined_v17`; the
whole ladder above inherits v17. `data/raw/combined_v16` is on neither this box
nor the transfer drive. One directory copied from the CPU box would let us
reproduce ramp from source and settle whether the v16→v17 recipe change cost
anything — the one confound this run could not control for.

## 10. Open, unchanged

- **The v16/v17 corpus confound.** Ramp trained on `combined_v16`; all three
  v18 arms that died on its Black leg trained on `combined_v17`; the whole
  ladder above inherits v17. `data/raw/combined_v16` is on neither this box nor
  the transfer drive — one directory copy from the CPU box would let us
  reproduce ramp and run the clean A/B.
- **§7.4** — the hand-corrected label precedent on `white_2026_07/game_00013`,
  which is inside every corpus in this ladder.
