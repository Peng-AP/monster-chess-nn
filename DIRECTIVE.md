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

**M1 — epoch headroom (HANDOFF §4.6, no longer parked).** Rerun the v18 GAP
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

**M2 — MCTS promotion-defense probe at population scale (HANDOFF §8.1).** The
§4.4 refused-capture failure lives in Q, not priors; the static probe measured
priors. Run the 400-position deck through **search** (400 sims, no noise) for
dup1/gap/ramp/v17: capture-chosen rate, Q(capture) vs Q(best-refusal) per
model. Output: whether dup1's n=1 failure is a population defect, and a
reusable promotion-defense eval deck for Phase 3 reporting.

**M3 — ramp optimism audit (HANDOFF §4.5).** Same 400 positions: calibration
of predicted value vs realized outcome (bucket predictions, plot realized win
rate). Ramp reads +0.58 for Black with a White pawn on b7; if that is
miscalibration, ramp remains a fine *opponent* but its self-play data would be
value-tainted — which decides whether Phase 2's self-play uses v17, ramp, or
both as players.

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

**D2 — ps_monster merges, both forks.** The knowledge-vs-belief fork stops
being a debate and becomes an A/B:
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

**D4 — owner-game intake.** Add the two orphaned 2026-07-26 playtest games
(`black_2026_07/game_00023`, `white_2026_07/game_00014`) to the corpus human
set. The `game_00013` hand-corrected label is included as documented — **owner:
say the word if you do not bless the §7.4 precedent and it comes out.**

## 4. Phase 3 — training arms

All arms: ramp labels (r50h60), scalar head, 15ch, seed 42, epoch budget from
M1 (80/10 if headroom is real, else 30/10). Naming: `models/candidates/v19_*`.

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
