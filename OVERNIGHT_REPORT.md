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

## 2. M3 — is ramp's Black-optimism insight or miscalibration?

_Filled in as referees land. Run 1 (ramp-as-Black) is in DIRECTIVE §M3._

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

_Results filled in as they land._

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
