# DIRECTIVE — the v19 campaign (2026-08-01)

**Objective, in the owner's words: make the models good enough to beat him.**
Concretely: a Black that converts the 3–4-pawn positions he currently wins 100%
of, and a White that stops gifting pawns. This document is the complete plan —
nothing parked, nothing shelved. It supersedes the "queued for the GPU box"
fragments in HANDOFF §7/§8 by absorbing all of them.

**Authorization.** The owner's instruction of 2026-08-01 ("effect changes that
will actually get this project off the ground… no shelving") is treated as the
standing go for every run in this directive, including overnight and
multi-worker jobs — with two carve-outs that stay absolute: no workers launch
while he is playing, and **promotion still requires his playtest**. No gate
threshold moves. Ever.

---

## 0. What the GPU changes — measured, not assumed

| cost | old box (CPU) | this box (5060 Ti) | factor |
|---|---|---|---|
| training epoch (v18 recipe) | 8.25 min | **~10–16 s** (measured 2026-08-01: load+epoch+val+test = 16.0 s) | ~40–50× |
| 30-epoch run | ~4.5 h | **~6–8 min** | " |
| 80-epoch run | ~11 h (never affordable) | **~15–20 min** | " |
| full-length game @400 sims, 1 worker | ~10 min¹ | **121 s** (was 631 s before the clone fix) | ~5× |
| aggregate match throughput @400 sims | — | **7.1 decisions/s** at 8 workers | — |
| full 100-game gate (3 legs) | ~3 days | **see §0.1** | — |

¹ the old box's recorded `sec_per_decision ~1.3` came from `benchmark.py`
anchor runs, which are cheaper per decision than NN-vs-NN; it is not a like
for like comparison and the CPU-box game figure was never measured directly.

Every "we can only afford one arm" constraint in the project's history is void.
The campaign below is designed for the new economics: **factorial arms instead
of single recipes, 40-game gates instead of 10, and reinforcement cycles that
were never affordable on CPU.**

## 0.1 The match bottleneck was a bug, not the hardware (2026-08-01)

The GPU barely matters to match cost. Profiling one late-game decision at 400
sims: **18.6 s of 22.6 s was `chess.Board.copy()`** — python-chess copies the
*entire* move stack, one `copy.copy` per ply (3.0M calls), and MCTS clones once
per node expansion. NN forward was 0.63 s, **2.8%**. Clone cost grew with game
length, which is why late-game decisions cost ~5× opening ones and why the
games that reach the 150-turn cap — the common case in NN-vs-NN — were the
expensive ones.

`MonsterChessGame.clone` now copies `CLONE_HISTORY_PLIES = 8` plies instead of
all of them. Only `mcts._own_previous_moves` reads history, at offsets
−1/−3/−4. **Verified move-for-move identical**, not just same-result: the same
seeded 225-ply game replays with an identical move list, `563 s → 121 s`
(4.64×); two shorter games likewise identical (1.31×, 1.50× — less history to
copy). `tests/test_clone_history_depth.py` pins the depth against the offsets.

This is why the five-arm Phase 3 budget survives. At the old clone cost a
100-game gate was ~8 h per candidate and the five arms ~40 h.

**Also fixed: every worker default was set to crash this box.** 14 workers dies
during CUDA init with `fatal : Memory allocation failure` and leaves 14
orphaned ~1.4 GB processes. `data_generation.py` defaulted to `os.cpu_count()`
= 16, `tools/promotion_probe.py` and `tools/match.py` to `cpu_count() - 2` =
14 — so M2 tonight and the Phase 2 self-play generation would both have hit it
unattended. All three now share `config.DEFAULT_GAME_WORKERS = 8`, pinned by
`tests/test_worker_defaults.py`. Throughput plateaus there anyway: 4 workers
5.39 decisions/s, 8 → 7.11, 12 → 7.38.

Calibration artifacts: `scratchpad gpu_epoch_bench` (epoch-1 losses reproduce
the v18 gap arm's recorded CPU trajectory within nondeterminism noise — train
3.9353 vs 3.9464, val 3.0664 vs 3.0631 — which **validates the rebuilt corpus**
as equivalent to the original `combined_v17_r50h60`).

**Known discrepancy to re-derive, not blocking:** law 3 records ramp-label
saturation |v|>0.9 at 10.3%; the rebuilt corpus measures 24.5% (24.6% train
split). Since epoch-1 training reproduces exactly, the corpus is right and the
10.3% was measured on some other basis. Find that basis (likely a different
denominator) and correct the law's wording in memory. Do not re-litigate the
ramp itself — its gate results stand.

## 1. Phase 0 — foundation (DONE 2026-08-01, except the last item)

- [x] Artifacts transferred from old box via USB, verified against HANDOFF
      numbers (v17, 15 rejected models, combined_v17, ps_monster 829/43,939,
      human_games 2026-07, playstrategy cache).
- [x] `data/processed/combined_v17_r50h60` rebuilt (train=116,324 — matches the
      v18 manifests byte-for-count) and validated by epoch-1 reproduction.
- [x] `--patience` added to `train.py` (default 10, recorded in the run
      manifest via `vars(args)`). 166/166 tests green.
- [x] GPU training calibration (above).
- [x] Match calibration: **179 games/hour**, so a full 100-game gate is
      **34 min** and five Phase 3 arms cost ~2.8 h. Protocol fixed at 20 games
      per side per leg; the 0.40 floor untouched. `tools/gate.py` runs all
      three legs through the single `match.run_match` schema, keeps the
      thresholds as constants no CLI flag can reach, and cannot report PASS
      from a rehearsal. See `PHASE0_REPORT.md` — the calibration run doubled as
      a real gate on the sparring partner and **its 0.70 h2h did not replicate
      at n=20/side (0.575)**, which is §7.3 demonstrating itself.

## 2. Phase 1 — decisive measurements (first GPU day)

These aim the campaign; all are cheap now and none needs owner time.

**M1 — DONE 2026-08-01, and the answer is "no headroom".** Ran verbatim at
`--epochs 80 --patience 10` (`models/candidates/v19_m1_ep80`): **best epoch 24,
early stopped at 34.** The recorded 30-epoch run's best was 25, so lifting the
cap from 30 to 80 moved the best epoch by −1, inside GPU/CPU nondeterminism.

Under-training is therefore **excluded**: every v18-era negative stands on its
own merits, and Phase 3 arms take a **30–40 epoch budget with patience 10**.
The parked question from HANDOFF §4.6 is closed — it was cheap to ask and the
answer was the unglamorous one. (Retire the retraction note in the
`training-run-audit-facts` memory: the record now has the measurement, not an
inference from cap-vs-patience.)

_Original specification, kept for the record:_ Rerun the v18 GAP
arm verbatim — `--data-dir data/processed/combined_v17_r50h60 --target
game_result --value-head scalar --seed 42` — at `--epochs 80 --patience 10`.
StepLR is epoch-indexed and the shuffle is seeded per-epoch, so the first 30
epochs are the same trajectory: this *extends* the recorded run rather than
re-rolling it. Gate at 20/side. Two possible worlds:
- Best epoch stays ≤27 → under-training is excluded; all v18-era negatives
  stand; proceed with 30–40 epoch budgets.
- Best epoch >27 and gate legs move → **every prior arm was cut short and every
  Phase 3 arm inherits the 80/10 budget.** At 15 min/run this costs nothing to
  know.

**M2 — DONE 2026-08-01: dup1's refusal is not a population defect.** 400
positions through search (400 sims, no noise), all four models. Evidence:
`benchmarks/promotion_defense_search_20260801_210436.json`.

| model | capture rate | capture never visited | refused despite higher capture Q | mean value (Black POV) |
|---|---|---|---|---|
| v17 | **0.565** | 83 | 12 | −0.537 |
| ramp | 0.585 | 67 | 10 | **−0.129** |
| gap | 0.603 | 65 | 10 | −0.399 |
| dup1 | 0.595 | 57 | 13 | −0.377 |

dup1 is mid-pack on every column and the **incumbent** is the worst — lowest
capture rate, most positions where search never visits the capture at all.
§4.4 closes: the failure was n=1, now confirmed in Q and not only in priors.

The probe reproduces the §4.4 anchor position to four decimals (capture priors
dup1 0.0908 / gap 0.1149 / ramp 0.1169 / v17 0.5835 against the recorded
0.0907 / 0.1147 / 0.1172 / 0.5834, and dup1 alone plays `a8b8`), which is what
licenses trusting the rest of the run.

Deck and probe are committed — `tools/make_promotion_defense_deck.py`,
`data/start_fens/promotion_defense_deck_v1.jsonl` (400 of 960 qualifying
positions), `tools/promotion_defense_probe.py`. The original deck lived in a
scratchpad on the old box and is gone, so absolute values are **not**
comparable to the recorded §4.4/§4.5 numbers; orderings are.

**Two metric bugs were found and fixed mid-run; both would have inverted a
conclusion.** (1) `MCTSNode.q_value` returns 0.0 for an unvisited child, and Q
here is mostly negative, so "best non-capture" read 0.0 in nearly every
position — the first pass showed a negative mean Q gap, i.e. the exact opposite
of the corrected +0.10 to +0.13. (2) `NNEvaluator.evaluate_with_policy` returns
a **White**-perspective value while a root child's Q is **root side-to-move** —
mixing them flips the sign of the headline. Everything above is Black-POV.

**M3 — DONE 2026-08-02: ramp's Black-optimism is MISCALIBRATION, not insight.**

Each model played Black itself over the same 400 positions against the same
White. **Conversion is identical — ramp 20.75%, v17 19.75% (n=400, ~0.5 SE
apart) — while their predictions sit 0.41 apart.** Ramp's calibration error is
+0.518 against v17's +0.130. It converts no better; it just believes it is
better off.

**Consequence, and the decision this measurement existed to make: D3's cliff
self-play must not take value targets from ramp.** It remains a legitimate
opponent (games labelled by outcome, not beliefs) and remains the gate bar —
being miscalibrated about Black is not the same as being weak, and it still
beats v17 ~0.65. Full write-up and the discarded first referee in
`OVERNIGHT_REPORT.md` §2.

_Original specification and the partial state, kept for the record:_

§4.5's ordering **reproduced on an independent deck**: mean predicted value
(Black POV) ramp −0.129, dup1 −0.377, gap −0.399, v17 −0.537, against the
recorded ramp −0.02 / dup1 −0.33 / gap −0.35 / v17 −0.47. Ramp is by far the
most Black-optimistic, and it is now the bar.

**The first ground truth was invalid and is kept only as a cautionary
artifact** (`promotion_defense_outcomes_20260801_210716.json`). Heuristic on
both sides converts Black in **3 of 400** positions (0.75%), so every predicted
band maps to realized −1.00 and the calibration discriminates nothing. It would
have produced a confident table declaring all four models wildly optimistic —
equally consistent with the models being right and the referee being unable to
play Black. **A referee too weak to convert cannot adjudicate optimism about
converting.**

**Referee run 1 of 3 done** (`..._outcomes_rampblack_20260801_214726.json`):
ramp itself as Black vs heuristic White, 400 positions, 400 sims, 37.5 min.

| | |
|---|---|
| Black conversion | **0.2075** (vs 0.0075 under heuristic-Black — 28×) |
| outcomes | 314 losses, 56 Black-dominant draws, 27 wins, 3 draws |
| mean realized (Black POV) | **−0.647** |
| ramp's mean prediction | **−0.129** |

Ramp is measured against *its own play*, against a White weak enough to
flatter Black, and still reads **+0.52 optimistic**. That points at §4.5 being
miscalibration rather than insight — but see the caveat below before acting.

**Not yet run** (each ~37.5 min, the third slower because it is NN-vs-NN):
- `--playout-black models/fresh_start_v17/...` — does the M2 ordering survive
  contact with realized outcomes? v17 predicts −0.537; if it converts near
  that it is calibrated and ramp is not.
- `--playout-white ramp --playout-black ramp` — the strong-White ceiling. Can
  only push realized value further below prediction, so it sharpens the
  magnitude and cannot flip the direction.

**Caveat that must survive the pause:** heuristic-White is weak, so 0.2075
*overstates* what Black converts against real opposition. The conclusion
"ramp's value head is miscalibrated" is well supported in direction; the
magnitude is not pinned until a strong White is measured.

**What this decides:** whether Phase 2's D3 cliff self-play may use ramp as a
player. If ramp's value head is miscalibrated, it stays a fine *opponent*
(games labelled by outcome, not by its beliefs) but its search values are not
trustworthy training signal.

**M4 — DONE 2026-08-02: the cliff is SEARCH-limited. Arm S is the indicated
next lever.**

`v19_B` as Black from 100 cliff starts, **only Black's search varied** (White
fixed at heuristic@400): **0.360 → 0.520 → 0.670 → 0.840** at 200 / 400 / 800 /
1600 sims, SE 0.050. Monotone, +0.48 across 8×.

By the reading rule stated below, that is the "conversion rises with sims →
knowledge is present, search-limited → target **value sharpness**, arm S
dominates" branch. The policy-teaching lever (ps_monster) has already been
pulled and delivered; the remaining gap at 400 sims is search, not ignorance.

Caveat: White is held at 400 while Black climbs, so part of the rise is Black
outsearching White — this locates B's ceiling relative to what 400 sims
extracts, which is the question that picks the next lever, and is not a claim
about B at 1600 against a strong White.

Also: B shows **no** "worse at high sims" pathology up to 1600, contrary to the
v17-era observation behind the 800-sim playtest rule. Untested above 1600, so
the rule stands until measured.

_Original specification:_

**M4 — cliff-vs-sims conversion curve.** From the cliff deck (see D3), Black
conversion rate at 200/400/800/1600/3200 sims, v17 and ramp, vs both
NN-White and heuristic-White. HANDOFF says 800 sims → 0.70 h2h ruled out
"search depth" at one point on one metric; this is the full curve on the
metric that matters. Reading:
- Conversion rises with sims → knowledge is present, search-limited → data
  offensive targets *value sharpness* (self-play arm S dominates).
- Flat curve → knowledge absent → policy teaching dominates (ps_monster arm K).
This curve's 800-sim point becomes the campaign's tracked progress number. It
is a diagnostic, **not** a promotion proxy — the owner's playtest remains the
only gate that promotes (binding rule: his eval is not replaceable).

## 3. Phase 2 — the data offensive

The corpus is 9.4% pawn phase; the cliff is a pawn-phase problem; the owner's
repertoire is the corpus's only opening variety (law 3.1). Attack all three
deficits at once. **Every source below lands in `data/raw/` as its own dir; the
merged corpora are built per-arm with `set_human_duplication.py`-style tooling
and audited with `tools/pretrain_check.py` + `tools/corpus_audit.py` before any
training run spends on them.**

**D1 — value-weight pipeline (the §7.1 fork-splitter, scoped and built).**
`data_processor.py` emits `value_weights.npy` mirroring `policy_weights.npy`;
`train.py` applies it in the value loss exactly as the policy path does (the
weighted-loss machinery already exists at `src/train.py:542`). Contract tests
pin: default all-ones (bit-identical behavior on old corpora), zero-weight
records contribute zero value gradient. This makes "teach policy, not value"
expressible — the capability HANDOFF said was "bounded; not yet scoped."

**D2 — DONE 2026-08-02. ps_monster is the lever, and the fork has a nuanced
answer: the value labels buy game strength and cost calibration.**

Both arms beat the bar. Against ramp, pooled over the bar leg and its
confirmation (40 games/side): **K 0.7125 aggregate (Black 0.700), B 0.7625
(Black 0.650)**, against a control on `combined_v17` that scored 0.425 with a
Black leg of **0.300** — the identical 0.30 the v18 gap/spatial/dup1 arms died
on. Cliff conversion rose from ramp's 0.347 to K 0.473 / B 0.527.

Then the fork itself, isolated: on the promotion-defense deck K and B convert
**identically** (0.395 vs 0.405, n=200, SE 0.035) while B's value head is
**+0.283 miscalibrated against K's +0.119**. Same positions, same opponent,
same results, different beliefs. Head-to-head B beats K 0.65–0.35.

So: ps_monster's outcome labels do not improve conversion, they inflate
optimism — and B nonetheless wins more games. Full tables in
`OVERNIGHT_REPORT.md` §5–6. **Which to prefer is an owner call**, sharpened by
the fact that ramp was rejected for exactly the sort of play faults an
over-optimistic value head produces.

_Original specification, kept for the record:_
- **Merge-K (knowledge):** combined_v17 + ps_monster at 1×, ps records
  `value_weight=0` — 43,939 records of policy-only teaching, 41.2% pawn phase,
  829 games of opening structures the corpus has never seen, zero risk of
  importing 1600-blitz outcome beliefs into the value head.
- **Merge-B (belief):** same merge, full value labels (ramp-shaped from
  `game_result`). If B beats K, the belief worry was wrong and the labels
  helped; if K beats B, the fork is answered with evidence.

**D3 — cliff self-play: manufacture the missing phase with real search values.**
ps_monster carries `mcts_value=0.0` — only self-play can produce pawn-phase
records with search-grounded values. Steps:
1. **Re-aim `make_promo_deck.py`** (HANDOFF §8.3 — 153/163 positions currently
   have the owner promoting as *Black*, the wrong side). Build two decks:
   *Black-defends* (White pawn advanced, Black must stop it) and
   *Black-converts* (wP≥3, material for Black — the cliff itself).
2. **Harvest cliff starts**: every wP≥3 position from the owner's won-as-Black
   human games (his conversions are the existence proof of the technique),
   plus wP≥3 positions sampled from ps_monster games that Black won, plus the
   existing curriculum tiers. Target ≥300 distinct starts.
3. **Generate**: `data_generation.py` from these decks at **1600 sims** (GPU
   makes this the new normal — the old 400 was a CPU compromise), Dirichlet
   root noise on, v17-vs-v17 plus cross-play v17-vs-ramp (subject to M3: if
   ramp's value head is miscalibrated, it still plays as an *opponent* whose
   games are labeled by outcome, not by its beliefs). Volume target: lift the
   merged corpus's wP≥3 share from 9.4% to **25–30%**.

**D4 — owner-game intake. DONE 2026-08-01. Bigger than it looked: 27 games,
not 2** (25 from March plus the 2 known July orphans). Audited
2026-08-01 by content (corpus files store the newer games at 6× in-file
duplication, so filename and record-count matching both lie). Owner games on
this box that **no corpus contains**:

| directory | games | positions | wP≥3 | outcome | replay |
|---|---|---|---|---|---|
| `human_games/black_2026_03` | **21** | 798 | **27.1%** | owner-as-Black won all 21 | 21/21 clean |
| `human_games/white_2026_03` | **4** | 68 | **92.6%** | owner-as-White won all 4 | 4/4 clean |
| `black_2026_07/game_00023` | 1 | 38 | — | the dup1 echo game | clean |
| `white_2026_07/game_00014` | 1 | 19 | — | the refused-capture game | clean |

The March directories are on this box only — they were never on the CPU box
(HANDOFF recorded "4+3 from 2026-03"; there are 4 and **21**). They replay
without a single error under the current post-2026-07-04 ruleset, so the rules
correction did not invalidate them.

**This is the highest-density owner data the project has.** The corpus is 9.4%
pawn phase; these are 27% and 93%, and the 21 Black games are 21 more instances
of the owner *converting* the phase Black converts 7–14% of — the existence
proof D3 wants cliff starts from. Against a human set of 96 games, this is a
26% increase.

**Folded in** by `tools/add_owner_games.py` at 6× duplication (matching the
multiple the newer human games already carry, so the new corpus differs from
`combined_v17` by exactly "these 27 games were added"):

```
data/raw/combined_v19_base            human_games 96 -> 123 files
data/processed/combined_v19_base_r50h60   (ramp floor 0.5 / horizon 60, 15ch, seed 42)
```

`combined_v17` is untouched — v17, ramp and every v18 arm trained on it, and
mutating it would make those runs irreproducible. **`combined_v19_base` is the
new base every Phase 3 arm builds on**; the K/B/S/KS merges stack on top of it,
not on combined_v17.

The `game_00013` hand-corrected label is included as documented — **owner: say
the word if you do not bless the §7.4 precedent and it comes out.**

## 4. Phase 3 — training arms

All arms: ramp labels (r50h60), scalar head, 15ch, seed 42, epoch budget from
M1 (80/10 if headroom is real, else 30/10). Naming: `models/candidates/v19_*`.

**RUN 2026-08-02.** The ladder was built on `combined_v19_base` (v17 + the 27
owner games), not `combined_v17`, and a `control` rung was added so each step
attributes one change. Results — verdicts are from `tools/gate.py`, unmodified:

| arm | corpus | verdict | vs ramp pooled (W/B) | Black, all legs |
|---|---|---|---|---|
| control | combined_v17 | **FAIL** | 0.550 / **0.300** | 0.440 |
| **O** | + 27 owner games | **PASS** | 0.775 / 0.450 | 0.500 |
| **K** | + ps(value_weight 0) | **PASS** | 0.725 / **0.700** | 0.671 |
| **B** | + ps(full value) | **PASS** | **0.875** / 0.650 | 0.686 |

Steps against the noise floor (SE 0.079 on a pooled 40-game leg): owner games
+0.150 (1.9 SE), ps_monster a further +0.20–0.25 (2.5–3.2 SE). **Data moved
what architecture (§4.2, a measured null) and duplication (§4.1) never did.**

Still unrun: **S** (cliff self-play — deck built, `data/start_fens/cliff_starts_v2.jsonl`,
412 starts; generation must take values from v17, not ramp, per M3),
**KS**, and **C** (capacity; report *tower* params, not total — v18_cap moved
total by 8% because `policy_fc` is 81.5% of the model).

_Original arm specification:_

| arm | corpus | tests |
|---|---|---|
| **K** | combined_v17 + ps(value_weight 0) | knowledge hypothesis |
| **B** | combined_v17 + ps(full value) | belief hypothesis |
| **S** | combined_v17 + cliff self-play | value-sharpness hypothesis |
| **KS** | combined_v17 + ps(v0) + cliff self-play | the max-data arm |
| **C** | best-of-above's corpus, tower 128→192 (`--res-channels`) | capacity — the one lever law 6 leaves standing, done properly this time (v18_cap moved total params 8% because `policy_fc` is 81.5% of the model; report *tower* params, not total) |

Five arms ≈ 2 hours of training total at GPU rates. The budget lives in the
gates, not the training — which is exactly how it should be.

**Gate protocol, every arm identically** — `py -3 tools/gate.py --model ...`:
- **vs `fresh_start_v18_ramp` — the bar** — 40 games (20/side), 400 sims
- vs `fresh_start_v17` — 40 games (20/side), 400 sims
- heuristic anchor — 20 games, 400 sims
- **confirmation**: a candidate that clears all of the above replays the bar
  leg on a fresh opening seed and must clear it again (+23 min, passing
  candidates only).
- Per-side floor **0.40** on every leg; aggregate must beat 0.50 on **both**
  model legs.

**Owner, 2026-08-01: "Every model should be better than the last,
definitively. Last should be ramp."** The bar is ramp, not the incumbent —
v17 holds the version number, ramp holds the strength. This is a hard bar:
v17 itself scores **0.275** against ramp. Arms will die here while comfortably
beating the incumbent, and that is the intent. The confirmation leg exists
because two independent 40-game reads of one fixed matchup came out 0.575 and
0.725 (PHASE0_REPORT §5.1) — one leg over 0.50 is not "definitively".
- **Read per process note §12:** total per-side scores across all legs against
  the noise floor (SE ≈ √(n·0.25)) before believing any direction. n=20/side
  halves the false-floor-trip rate that §4.1 demonstrated at n=10.
- Report every leg per-side in the run report. Aggregates hide collapses
  (law 8, burned four times).
- Promotion-defense deck rate (from M2) reported per candidate as context,
  never as a gate.

## 5. Phase 4 — close the loop (the part CPU could never afford)

Supervised training on a static corpus is how the project got a model the
owner's repertoire echoes back at him. The fix for exceeding your data is the
standard one:

1. Best gated candidate generates fresh cliff self-play (D3 decks, 1600 sims,
   noise on) — a few hours of GPU.
2. Rebuild corpus: union with prior self-play, keep human + ps sources fixed.
3. Retrain (same recipe), regate (same protocol).
4. Repeat while gate scores climb; stop on the first cycle that fails to beat
   its parent on the totalled per-side score (no cherry-picking a leg).

Two to three cycles ≈ two to three GPU days. Each cycle's evidence goes to
`benchmarks/` before the next starts. If cycles flatten immediately, that is a
result: the ceiling is the recipe, not the iteration count, and arm C's
capacity axis becomes the next variable.

## 6. Phase 5 — v19 and the owner gate

A candidate reaches the owner when it (a) passes the full gate protocol, and
(b) beats the campaign's tracked cliff-conversion baseline meaningfully (not
one leg, totalled). Then:

- **Owner playtest at 800 sims** (never more — the model is measurably worse
  at 2000–5000; HANDOFF/memory addendum). Suggested menu: his cliff positions
  with him as White vs candidate-Black — the "beat me" test, literally — plus
  free games both colors.
- Promotion per CONTEXT.md: candidate → `fresh_start_v19`, new incumbent.
  Rejection → `models/rejected/`, observations recorded verbatim (they have
  been confirmed by measurement every single time), folded into the next cycle.

## 7. Standing discipline (unchanged, non-negotiable)

- Laws 1–10 and the §9 do-not-do list stand. Note their precise scope: law 5
  forbids blending/fine-tuning/label-reshaping **on the existing corpus** — the
  Phase 2/3 arms are *data* changes with fixed labels, which is the allowed
  axis. No side-specialized heads. No proxy scorecards. No threshold changes.
- Commits: owner identity, one-line, no co-author trailer. No push unless
  asked. Root carries only this directive + the active run report; concluded
  drivers retire to history.
- Runs detached per HANDOFF §10.1 when they must survive a session; watch the
  artifact, not the log grep (§10.2).
- Every claim in run reports cites its JSON in `benchmarks/`. Distrust any
  success rate not tested against the hard case (§12).

## 8. Sequence and clock

| when | what | GPU-hours |
|---|---|---|
| today | Phase 0 close-out (match calibration, gate driver) | <1 |
| tonight | M1 headroom + M2/M3 probes (unattended batch) | ~2 |
| day 2 | M4 curve; D1 value-weights + tests; D3 deck builds | ~3 |
| day 2–3 | D2 merges + corpus audits; cliff self-play generation | overnight |
| day 3–4 | Phase 3: five arms trained + gated | ~6 |
| day 4–6 | Phase 4: 2–3 reinforcement cycles | ~8/cycle |
| day 6–7 | v19 candidate report → **owner playtest** | his call |

The bottleneck is now match/self-play throughput, not training. If the match
smoke shows games are still slow, the fix list is: batched-leaf sizes, worker
count vs GPU contention, and per-worker CUDA context sharing — measured before
optimized.

*Written 2026-08-01 on the GPU box. This is the active campaign document; it
retires when v19 is decided.*
