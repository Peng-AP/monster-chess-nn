# Overnight run — 2026-08-01/02

Active report for the unattended session that began 2026-08-01 ~22:20. Written
as results land; every claim cites its artifact in `benchmarks/`.

**Nothing here promotes anything.** Promotion needs the owner's playtest, and
no gate threshold was moved.

---

## 1. What ran

| # | work | status |
|---|---|---|
| 1 | M3 referees (v17-as-Black; ramp-both-sides deferred) | see §2 |
| 2 | D1 value-weight pipeline | done, §3 |
| 3 | D2 Merge-K / Merge-B corpora | done, §4 |
| 4 | Phase 3 corpus ladder: control → O → K → B | see §5 |
| 5 | M4 cliff-vs-sims curve | see §6 |

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

## 6. M4 — cliff conversion vs search depth

_Pending._

---

## 7. Open, unchanged

- **The v16/v17 corpus confound.** Ramp trained on `combined_v16`; all three
  v18 arms that died on its Black leg trained on `combined_v17`; the whole
  ladder above inherits v17. `data/raw/combined_v16` is on neither this box nor
  the transfer drive — one directory copy from the CPU box would let us
  reproduce ramp and run the clean A/B.
- **§7.4** — the hand-corrected label precedent on `white_2026_07/game_00013`,
  which is inside every corpus in this ladder.
