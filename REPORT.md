# MASTER REPORT — through 2026-08-15: native engine to bootstrap successor

Replaces the 2026-08-03 run report (git history holds it). Covers the owner's
rewrite directive (`DIRECTIVE.md`, 2026-08-03) from its writing through today:
**five of six phases closed inside two days** against the directive's own
1.5–2-week optimistic bound, the first LC0-derived search changes, the data
intake, and the answer to the question the whole campaign was premised on.

Every claim cites its artifact in `benchmarks/` (`benchmarks/INDEX.md` maps the
active evidence set). Every bug below is pinned by a regression test. **Suite: 606 passing,
verified 2026-08-15 by full discovery; native Rust: 5 passing.**
Long runs log to `logs/`; `py -3 tools/runs.py status` shows active/recent
progress (`status --all` includes older history).

---

## 1. Where things stand

| phase | status | headline evidence |
|---|---|---|
| E0 calibration | **done** | curve + forced-capture spike (§5, §6) |
| E1 rules core | **done** | exact-order movegen parity; lockstep 31,807 plies |
| E2 heuristic + encoding | **done** | **bit-identical** on 148,272; byte-equal tensors |
| E3 native MCTS | **done** | gates: **400/400**, 0.525 (0.71 SE), 6.31× |
| E4 integration | **done** | clean 240-game overnight, healthy distribution |
| E5 re-baseline | **done** | 10-pair cross-table + independent bar confirmation |
| E6 exploit | started | solver + reuse measured (§8) |

Why two days against a two-week estimate: the parity regime did what it was
designed to do. Every overrun surfaced as a **failed gate with a specific
number attached**, not as silent wrongness — nine substantive bugs were caught
this way (§4), each cheap to find because a gate pointed at it.

---

## 2. The frame: only a king capture is a win (2026-08-03, compressed)

The owner's scoring ruling reversed a promotion on identical games: `v20w`
gated PASS at 08:12 and FAIL at 10:52 on the same 40 games — its Black leg was
0.50 only because move-limit relabels counted as wins
(`gate_v20w_20260803_{081212,105202}.json`). Both v20 arms rejected. The turn
cap tested dead: 2.7× the moves, identical captures
(`promotion_defense_outcomes_capraise_20260803_103416.json`).

**Standing consequence: no number measured before 2026-08-03 is comparable to
one measured after.** E5 discharged that debt in one captures-only native
cross-table (`rebaseline_20260804_155035.json`, 40 games/pair):

| row engine | vs v17 | vs ramp | vs v19 | vs v19_B | vs heuristic |
|---|---:|---:|---:|---:|---:|
| v17 | — | 0.425 | 0.350 | 0.263 | 0.637 |
| ramp | 0.575 | — | 0.350 | 0.312 | 0.588 |
| v19 | 0.650 | 0.650 | — | **0.425** | 0.738 |
| v19_B | 0.738 | 0.688 | **0.575** | — | **0.762** |

The direct v19/v19_B result repeated **0.575 for v19_B** on a second fully
disjoint 40-game read. E5 therefore settles the bar at `v19_B`; thresholds are
unchanged. `fresh_start_v19` remains the numbered incumbent until an owner
promotion, but every new candidate must beat the strongest engine on record.

---

## 3. The port is faithful — the parity record

| layer | scale | result |
|---|---|---|
| pseudo-legal movegen vs python-chess | 248k positions (sets), 16,684 (**exact list order**) | 0 mismatches (`movegen_parity_*`) |
| Monster action APIs (all four) | 60k positions | 0 (`monster_api_parity_*`) |
| game state machine, lockstep | 31,807 plies, every field incl. heuristic + cap relabel | 0 (`state_machine_parity_*`) |
| heuristic evaluator | 148,272 positions, 48,535 pending-aware | **bit-identical, worst Δ = 0.0** (`eval_parity_20260803_211157.json`) |
| tensor encoding, 15ch + 17ch | 98,272 each | 0 byte-unequal (`encoding_parity_20260803_221918.json`) |
| corpus replay | 968/991 games, 48,272 plies | clean; 23 legacy games carved out with one verified cause (`replay_parity_e1_full_*`, `data/legacy_unreplayable_games.json`) |
| E3 gate (a): selected-move agreement | 400 deck positions, 400 sims | **400/400** (`search_agreement_20260803_231929.json`) |
| E3 gate (b): self-match | 200 games | 0.525, **0.71 SE** from 0.50 (`engine_self_match_20260804_000702.json`) |
| E4: overnight generation | 240 games, 700 sims | 240/240 saved, W174/B44/cap22, fullmove 5–75, zero timeouts |

Deliberately not ported: the atomic-pair prior path (unreachable for
`MonsterChessGame` — documented, not dead code) and solver/reuse into the UCB1
reference path.

---

## 4. The bug ledger — what the gates caught

Each of these would have corrupted training data or measurements silently.
None would have crashed.

1. **ep capturer rank** — native offered White en-passant-capturing its *own*
   double-push square (`c2c4,d2c3`), because Monster Chess forces `board.turn`
   back to WHITE between halves. 220 differential failures.
2. **Castling rights outlive a captured king** — python-chess voids a side's
   rights when its king leaves home; in this variant the king is *captured*.
   The FEN drops `kq` the instant a piece lands on e8; the port kept them.
3. **Generation order** — `truncate_wins` returns the *first* winning move
   found, and python-chess scans squares descending. Same winner either way,
   different recorded FEN and policy target. Caught in lockstep at ply 4,695;
   fixed by mirroring `scan_reversed` and section order exactly.
4. **Value-head sign flip at the NN bridge** — the model speaks in
   side-to-move perspective; the bridge forwarded it raw, so every
   Black-to-move leaf backpropagated the wrong sign. Gate (a) read 90% with
   rank-5 disagreements; the tightened 99% bar (directive said 95%) is what
   made it un-ignorable. Fixed → 400/400.
5. **Hybrid value source** — `HybridEvaluator` takes values from the heuristic
   and only policy from the NN, and skips the forward at |h| ≥ 0.95.
   The native search initially couldn't express it. **Correction (2026-08-16):
   generation does NOT use it.** `data_generation` defaults `hybrid_eval` to
   False and neither `iterate.py` nor `generation_driver.py` passes
   `--hybrid-eval`, so the flag is available but unused. It substitutes
   heuristic values for network values, which made sense when the network was
   weak and is very unlikely to now.
6. **Frozen RNG, within a run** — a constant seed re-created an identical
   stream per decision: temperature sampling returned the same move 10/10,
   Dirichlet noise the same vector 5/5.
7. **Frozen RNG, across games** — harnesses re-seed the *global* RNG once per
   game but build engines once per worker; an engine holding its own seed
   never saw it. Symptom: 240/240 White at fullmove 6–7 vs python's spread to
   75. Fix: default follows the global RNG per decision, exactly as `MCTS`
   does. Post-fix, native generation matches python's distribution
   (W174/B44/cap22 vs W14/B10 shape).
8. **A FEN under-determines a Monster Chess state, four ways** —
   `white_half_pending` (different action set), `turn_count` (the cap),
   move history (the oscillation penalty), and ep-with-no-legal-capture
   (python-chess writes `en_passant="legal"`; 1 in 148k positions evaluated
   differently through its own FEN round-trip). Every instance failed
   silently. `Game`/`Tree` now take the whole state; **E4+ never passes bare
   FENs between engines.**
9. **The post-timeout pool hang** — the historic fix existed; what was missing
   was proof. `tests/test_pool_teardown.py` forces the exact sequence (hang →
   timeout → teardown → fresh pool) and pins the subtle half: `_processes`
   must be snapshotted *before* `shutdown()`.

---

## 5. Speed, measured — and the honest ceiling

Components (`native_speedup_20260803_211854.json`): movegen 19–26×, clone
83×, heuristic 47×, tree search 58×, composite playout 25.6×.

Whole decisions, warmed, same model:

| sims | batch | python | native | speedup |
|---|---|---|---|---|
| 400 | 16 | 551 ms | 180 ms | 3.07× |
| 800 | 64 | 1026 | 163 | **6.31×** (E3 gate ≥5×: met) |
| 1600 | 256 | 2152 | 216 | **9.98×** (E4 gate ≥10×: met) |

Pipelines: full 3-leg gate **34 min → 6.5 min** (390s,
`gate_fresh_start_v19_20260804_003547.json`); 16-game match 3.65×; the PPC
curve point at 1600 sims 1230s → 706s.

**Law 18 does not transfer to the native engine** — its "NN forward = 14%" was
measured where tree work dominated. Native at batch 16 spends **65%** of a
decision in the network (ceiling 4.7×); batch 256 forwards cost 0.037
ms/position vs 0.294 at batch 16. The batch width the engine *operates* at is
now the lever, not more Rust.

**Depth is the limit speed cannot move**: ~0.6–0.9 PV plies per doubling of
sims; the endgame PV is pinned at 4 plies even at 51,200 sims; policy priors
already quadruple PV depth over UCB1 at equal budget (13 vs 3). This is the
measured case behind the owner's queued pruning discussion (E6).

---

## 6. The sims curve is answered: it plateaus

Re-measured end to end on the native engine, one engine for all six points
(`promotion_defense_outcomes_ppccurve_native_s*_20260804_*.json`; 100
promotion-defense starts, White fixed at 400 sims, captures-only throughout):

| Black sims | 200 | 400 | 800 | 1600 | 3200 | **6400** |
|---|---|---|---|---|---|---|
| true king captures | 0.19 | 0.17 | 0.24 | 0.37 | 0.51 | **0.55** |
| dominant, unfinished | 11 | 26 | 26 | 24 | 20 | 18 |
| White wins | 0.70 | 0.57 | 0.50 | 0.39 | 0.29 | 0.27 |
| earlier Python read | 0.09† | 0.15† | 0.20† | 0.30† | 0.51 | — |

The 200→400 change is −0.02; gains from 400 upward are +0.07, +0.13,
+0.14, then **+0.04** (≈1 SE at n=100).
**The knee is at 3200.** The directive's §5 contingency is triggered: raw
simulations cap out around 0.55 on this deck, so the exploit order shifts from
"more sims" to the finisher/solver and to data. The rewrite's justification
stands on throughput: the first directly comparable full binding gate at the
3200 operating point took **48.5 minutes** (25.8 minutes for its 40-game bar
leg), versus the earlier 2.3-hour reference — not on sims buying conversion
indefinitely. The earlier 12-minute estimate was not a measured full gate.

† **The apparent low-sim engine gap was a deck confound, found in the artifact
audit on 04 Aug.** The Python 200/400/800/1600 rows used
`postpromo_starts_v1.jsonl`; every native row used
`promotion_defense_deck_v1.jsonl`. The decks overlap on only 10 of 400 FENs.
The sole same-deck Python/native point is 3200, where both read **0.51 exactly**.
The earlier low-sim rows therefore cannot support an engine-divergence claim.
A same-deck Python 200-sim control replaces the queued first-divergence trace;
only a real residual gap would earn tracing. E5 is unaffected — it is
single-engine throughout.

---

## 7. The oracle repairs (owner-directed, 2026-08-04)

Diagnosis first: `verify_scripted_mate`'s 9/12 was deterministic, and all
three failures were **material loss** — Black hangs heavies to the double-move
king from turn 5–23 onward. The depth-1 fence search cannot see a king that
captures two squares away.

Two fixes, both defaulted on per the owner's instruction:

- **Material guard** — refuse any move after which White can win a heavy or
  the king, when an alternative exists; stands down when everything hangs
  (same discipline as the king-safety override). **9/12 → 11/12**, and the
  last failure changed character: it reaches the cap with queen and both rooks
  intact — the finishing problem, not a blunder.
- **Forced-capture preflight** (exact AND/OR, depth 3) — on the E0(b)
  walked-past positions the fence heuristic converted **1/8; with the
  preflight 8/8**, each in the proven minimum
  (`forced_capture_v20asym_b1600_d3_20260803_140521.json`).

E0(b)'s wider finding stands: **29–30% of dominant-unfinished games held a
forced capture Black walked past** (`forced_capture_v19K_d3_20260803_133041.json`),
and the walked-past hits were the oracle's own moves — recorded into corpora
at policy 1.0. `tools/finish_unconverted.py` exists to *resume* those −0.5
games with the repaired tools and keep only real captures (a relabel would
assert plies that never happened; the ramp target is a function of
`plies_to_end`). Not yet run at scale.

---

## 8. LC0-derived search changes

**MCTS-Solver (certainty propagation)** — implemented, flagged, off by
default. Proofs only from king captures — the cap's ±0.5 relabel is an opinion
and a test keeps it from ever hardening into a proof. Proofs live in White's
perspective so the logic survives White's non-alternating half-pair.
Demonstrated: at 800 sims on a mate-in-2, solver-off reports +0.98 (the owner's
July complaint about "non-1 evals for mate positions"); solver-on proves it
and reports exactly +1.00. The decisive measurement is now complete: true
captures moved **0.37 -> 0.39 at 1600** and **0.51 -> 0.53 at 3200** over the
same 100 starts (`promotion_defense_outcomes_ppccurve_solver_s{1600,3200}_*`).
Paired by FEN, those are only 4 gained / 2 lost captures and 5 gained / 3 lost.
**Result: null.** Certainty propagation remains off by default; the deeper
alpha-beta prover is a separate E6 idea, not a reason to enable this flag.

**Tree reuse across moves** — implemented with derived rebasing: a node's Q
lives in its *parent's* frame, so rerooting changes exactly one node's frame —
negate the new root iff the side to move changed. (The python engine's
"would require rebasing every stored value" premise was wrong: it's one
value.) Measured in three stages:

| question | result |
|---|---|
| carry-over | 337→1,381 nodes/search at 200 sims; 591→4,258 at 400 |
| strength at equal sims | **0.5000** over 200 games — null (`reuse_benefit_20260804_...`) |
| wall-clock | **233 vs 301 ms/decision (−23%)** with early stop; −6% without |
| strength at equal time (516 vs 400 sims) | 0.5833 ± 0.0456, **+1.83 SE**, n=120 (`reuse_benefit_20260804_134606.json`) |

The equal-sims null has a clean mechanism: an inherited tree arrives with the
previous search's visit lead banked, so `_should_stop_early` fires sooner —
reuse converts into *time saved*, not depth. The −23% is the bankable claim;
+1.83 SE is suggestive but under this project's definitive bar. Off by
default; enable via `MONSTER_REUSE=1` (`MONSTER_SOLVER=1` likewise).

**Moves-left signal** — the forbidden version remains forbidden: reshaping the
value target is a do-not-do (law 5; a global per-ply discount was rejected in
v13 for taxing Black's long wins). The LC0 version is now implemented as a
separate, opt-in auxiliary head. Targets are remaining recorded decisions per
segment; capped draws and policy-only imports are masked, Black weighting (when
requested) carries through, and weighted Huber loss is isolated from the value
target. Old checkpoints and the `(value, policy)` inference ABI remain intact.

Three other low-complexity LC0-derived candidates are implemented behind
flags, all off by default: legal-move masking in policy loss/metrics (packed at
512 bytes/position), a compact source-to-destination attention policy head, and
EMA validation/checkpoint weights. PUCT's policy-prior temperature, `c_puct`,
and FPU are also instance parameters in both engines; `tools/search_sweep.py`
runs resumable same-checkpoint screens ranked by Black score, with the existing
White and aggregate floors retained. **WDL head** — machinery
exists (`train.py`), but law 4: label shaping is invisible under WDL; interacts
with the ramp.

**LC0-derived training screen — successor found.** Exact-v19_B one-variable
arms found the attention policy head strong while moves-left, legal masking,
SE blocks, and training-hyperparameter search did not produce a robust
successor. Plain attention improved over a self-calibrated v19_B in three
independent two-color reads (240 candidate games total): **+0.271 as Black,
+0.100 as White, +0.185 pooled**, but missed the binding gate's Black floor by
one half-point in its first small bar leg. The focused follow-up combined the
same 32-channel attention head with EMA 0.999, leaving v19_B's corpus, seed,
optimizer recipe, scalar target, and epoch count unchanged.

That `lc0b_attention_ema` candidate passed the unchanged binding gate and its
automatic fresh v19_B replay:

| binding leg | overall | as White | **as Black** |
|---|---:|---:|---:|
| v19_B | **0.7125** | **0.925** | **0.500** |
| v19_B fresh confirmation | **0.7375** | **0.925** | **0.550** |
| ramp | **0.8250** | **0.925** | **0.725** |
| fixed heuristic anchor | **1.0000** | **1.000** | **1.000** |

Artifact: `gate_lc0b_attention_ema_20260805_052437.json`; packaged summary:
`lc0b_attention_ema_successor_20260805.json`. This is the strongest automated
result on record. The owner's playtest then approved it as the interim
playing-strength bar; v19_B remains preserved as historical evidence.

**Immediate successor cycle (2026-08-05).** The owner then passed
`lc0b_attention_ema` qualitatively: strong in both colors, especially vicious
as White, with only occasional spotty Black conversion. A focused successor
screen rejected value emphasis, Black-policy reweighting, full and policy-only
fine-tuning, checkpoint interpolation, and a spatial value head. Widening the
attention channels from 32 to 64 was the sole repeatable improvement. Against
the approved 32-channel model, `lc0b_attention_ema_wide64` scored calibrated
**+0.050 Black / +0.025 White / +0.0375 overall** at 40 games and 400 sims,
then **+0.1625 / +0.0625 / +0.1125** on a fresh 80-game, 800-sim
confirmation. On 2026-08-05 the owner promoted this exact checkpoint as
`models/fresh_start_v20`; it is now both the numbered incumbent and formal
gate bar. The source campaign checkpoint remains preserved. Artifacts:
`lc0b_successor_wide_{screen,confirmation}_*` and
`lc0b_attention_ema_wide64_successor_20260805.json`.

The promoted v20 self-match baseline used identical checkpoints and search on
both sides for 80 native games at 400 sims. White won 47, Black won 16, and 17
drew: **0.6938 White score vs 0.3063 Black**, a 0.3875 color-score gap toward
White. This quantifies the operating-point skew; it does not compare two model
strengths. Artifact: `match_fresh_start_v20_vs_fresh_start_v20_20260805_150239.json`.

**Search-parameter screen — closed null.** Same-checkpoint `v19_B` screens at
400 sims changed one prior/PUCT setting at a time. The coarse read's best Black
result was `c_puct=1.2` at 0.35 (White 0.95, overall 0.65), still below the 0.40
floor. A disjoint fine sweep then read only 0.10 for the same setting; nearby
1.0 / 1.1 / 1.3 read 0.20 / 0.10 / 0.15. Policy temperatures 0.8–1.25 and FPU
0.2/0.4 were all Black <=0.20. Artifacts:
`search_sweep_20260804_black.json` and
`search_cpuct_fine_20260804_black.json`. Defaults remain unchanged.

**The tablebase analogue** — LC0 doesn't search endgames, it looks them up.
Our equivalent is the exact forced-capture solver + repaired oracle; the E6
candidate is an alpha-beta *prover* over the terminal objective ("forced king
capture within N"), where eval quality doesn't matter because lines bottom out
in captures. Native movegen at 19–26× should push the affordable proof depth
from 3 toward 5–6.

---

## 9. Data intake and post-E5 ladder result

- **`ps_monster_e1500`** — the full playstrategy corpus was on this box all
  along (`data/playstrategy/ps_games.json`, 2,966 validated games); only 829
  had been imported, at `--min-elo 1600`. Imported the 1500 tier as a separate
  source: **2,132 files / 103,414 positions**. The pre-training overlap audit
  then found 825 existing games plus 20 internal duplicates; the actual ladder
  arm adds **1,287 unique games / 59,434 positions**, policy-only, rather than
  silently double-weighting the old tier.
- **`combined_v19_K_owner41`** — 41 owner games (the whole top level of
  `human_games/`) had never been incorporated; corpus human files 123 → 164.
  L4 remains the best per-record lever on record (+0.150 from 27 games).
- **`tools/finish_unconverted.py`** — resumes −0.5 games rather than
  relabelling them (§7).

The three-arm native 3,200-sim ladder is complete; all failed `v19_B` and the
additions moved Black in the wrong direction on the binding bar leg:

| arm | overall vs bar | as White | **as Black** |
|---|---:|---:|---:|
| frozen-corpus base | 0.4750 | 0.700 | **0.250** |
| + unique e1500, policy-only | 0.3875 | 0.675 | **0.100** |
| + 41 owner games | 0.2125 | 0.350 | **0.075** |

Artifacts: `gate_post_e5_{base,e1500,owner41}_20260804_*` and
`post_e5_ladder_20260804_190131.json`. This does not overturn the established
owner-game effect in prior controlled contexts; it says these fresh-start
whole-corpus arms did not reproduce it against the much stronger `v19_B` bar.
That ladder result left `v19_B` strongest at the time. The later controlled
architecture screen in §8 produced `lc0b_attention_ema`, which has now passed
the automated gate; promotion is open only for the required owner playtest.

The legal-mask implementation also exposed **8 enabled illegal policy rows out
of 89,622 (0.009%)** in `combined_v19_K`: 7 legacy PlayStrategy rows (including
Black moves stamped as White's second half) and 1 self-play distribution row.
`policy_legality_combined_v19_K_20260804.json` records every source and sample.
Processing now masks those exact rows, records the count in corpus metadata,
and the training loss still fails if any enabled illegal target survives.

---

## 10. Infrastructure

- **`logs/` + `tools/runs.py`** — every managed long run gets a named, live
  log; `status` shows active/recent alive/elapsed/latest progress, while
  `status --all` includes older history and `tail` reads one. Progress
  streaming added to the probes (`imap_unordered` — a run that reports only at
  the end is indistinguishable from a hung one).
- **Storage cleanup 2026-08-05** — 74.833 GiB of reproducible rejected
  processed corpora, March-era model runs, failed HPO checkpoints/blends, and
  one explicit raw duplicate were sent to the Windows Recycle Bin. The active
  `combined_v19_B_r50h60` corpus, raw sources, benchmark evidence, and all
  approved/confirmed checkpoints remain. Completed logs and superseded setup
  JSON moved into archives; exact targets and recovery notes are in
  `logs/archive/cleanup_20260805_manifest.md`.
- **`benchmarks/INDEX.md`** — 224 artifacts, 30 families, newest marked,
  headline extracted. Files are never moved: docs cite them by exact name.
- **`tools/build_native.ps1`** — vcvars + PYO3_PYTHON + build + install +
  import check as one step; fails loudly when the `.pyd` is locked instead of
  silently testing the previous build.
- `models/` cleaned (~350 entries → 20); scratch data dirs removed; run logs
  archived.

---

## 11. Corrections ledger

Claims I made during this campaign and retracted, kept here because the
record should show its own repairs:

1. **E3's ≥10× gate was impossible as written** — Amdahl from law 18; then law
   18 itself proved non-transferable (§5). Recalibrated twice, finally from
   measurement.
2. **Stage-2 server "5×"** → 0.87× on search work; the first benchmark
   compared wall-clocks and credited the server with 8 model loads it had
   merely moved before the timer. Real win: 1 resident model instead of 8.
3. **"Tree reuse buys nothing"** → measured one axis (strength at fixed sims)
   and reported it as both. Its benefit is time (−23%).
4. **Curve-gap-was-the-RNG-bug** → falsified by an identical deterministic
   re-run; gap remains open (§6).
5. **"Native stronger under noise" at +1.74 SE, n=40** → over-read; the trees
   were identical (3,016 nodes both).
6. **Moves-left as "a one-flag experiment"** → it's a forbidden label change
   (law 5, v13); the real version is a model head.
7. **First bridge benchmark showed native 0.8× (slower)** → CUDA init inside
   the timed region.

---

## 12. V21 two-hour architecture/value screen (2026-08-05)

V20 remains champion. Seven controlled checkpoints were trained from the
exact `combined_v19_B_r50h60` split/seed/optimizer recipe. The regenerated
capture corpus was byte-identical on every frozen core array after restoring
the 16 legacy policy weights that current preprocessing now masks. No model
cleared repeatable positive deltas in both colors.

| candidate | calibrated result vs v20 |
|---|---|
| attention width 96 | Black -0.025, White +0.000 |
| side-specific attention adapter | Black -0.025, White -0.025 |
| attention width 128 | screen +0.025/+0.050; confirmation **-0.138/+0.000** |
| capture-only WDL value | Black +0.050, White **-0.250** |
| mixed scalar + capture-WDL, weight 0.10, late checkpoint | Black -0.125, White -0.400 |
| same mixed model, recovered epoch 9 | Black +0.025, White -0.050 |
| mixed auxiliary weight 0.03 | seed 1: Black **+0.225**, White +0.000; repeat: Black +0.075, White **-0.300** |

The capture-only signal consistently pushed Black in the desired direction,
but did so by sacrificing White/general play. It is therefore useful as an
auxiliary diagnostic, not as a replacement value target. The long mixed run
also exposed a checkpoint-selection defect: a tiny decisive-score increase at
epoch 25 overwrote epoch 9 even though validation policy CE had deteriorated
from 1.99 to 2.84; arena strength then collapsed. Future selection must apply
a policy-loss/top-1 guard or Pareto rule instead of summing policy and sign
minima without a regression bound.

Implemented and tested infrastructure from this screen:

- raw terminal provenance now emits `capture_results.npy` (`+/-1` only for
  captures; move-cap and other non-captures are draws);
- scalar value can retain the proven discounted target while a separately
  sourced capture-WDL head trains as a low-weight auxiliary;
- auxiliary checkpoints carry an explicit marker so inference cannot mistake
  the WDL head for the engine's primary value;
- optional zero-initialized side-specific attention policy adapters load and
  round-trip correctly.

Evidence: `v21_initial_screen_20260805.json`,
`v21_attention128_confirmation_20260805.json`, and the five
`v21_*capture_wdl*20260805.json` artifacts. Verification at that point:
**532 passed, 10 warnings, 3 subtests**. The best Black-leaning lead is
`v21_mixed_capture_wdl_w003` (SHA-256
`9b29cf111497483c44527edae64e6b29adbe81502aa14ac8c8f42e056e68c9af`), but
it is rejected, not v21.

---

## 13. Promotion-aware policy screen (2026-08-05)

The old 4096-action ABI encoded only source and destination, so all four
promotion pieces shared one training cell and one network prior. The new
opt-in ABI retains those 4096 legacy logits and adds 192 promotion cells:
two colors, eight source files, three destination directions, and q/r/b/n.
Old checkpoints still load unchanged, and the native/Python engines select the
extended index only when a checkpoint carries the new head.

The controlled corpus preserved every v20 position, value, split, policy
weight, and non-promotion target exactly. Only 3,790 augmented promotion rows
(1,895 originals) changed. `v21_promotion_policy_exact` loaded all 117 v20
tensors unchanged, froze their parameters and BatchNorm buffers, and trained
only a zero-initialized 1,548-parameter promotion delta.

Held-out promotion-row metrics improved sharply:

| split | full CE v20 -> candidate | full top-1 v20 -> candidate | promotion-choice CE |
|---|---:|---:|---:|
| validation (384) | 2.566 -> 1.665 | 7.8% -> 57.8% | 1.472 -> 0.982 |
| test (344) | 2.470 -> 1.314 | 4.1% -> 64.5% | 1.453 -> 0.718 |

The binding arena did not promote it. Against v20 it scored 0.5125 initially
(White 0.575, Black 0.450), then 0.5000 on the required fresh confirmation
(White 0.675, Black **0.325**). It scored 0.9375 against ramp and 0.950 against
the heuristic, so general strength was retained, but the fixed 0.40 Black floor
failed. V20 remains incumbent. Artifacts:
`promotion_policy_metrics_20260805_185746.json` and
`gate_v21_promotion_policy_exact_20260805_185732.json`.

---

## 14. Conversion-adapter closeout (2026-08-05)

The proposed phase-specific conversion adapters were tested and rejected. They
did not create a repeatable Black improvement and would encode a narrow game
given that normal training/search should learn:

| candidate | games | overall | White | Black | result |
|---|---:|---:|---:|---:|---|
| broad owner conversion adapter | 40 | 0.4875 | 0.725 | 0.250 | reject |
| micro post-promotion adapter | 40 | 0.4500 | 0.700 | 0.200 | reject |
| deep defense distillation, screen | 20 | 0.6750 | 0.750 | 0.600 | advanced |
| deep defense distillation, binding v20 leg | 40 | 0.4500 | 0.675 | 0.225 | reject |
| immediate-capture distillation, screen | 20 | 0.5750 | 0.750 | 0.400 | advanced |
| immediate-capture distillation, confirmation | 40 | 0.4250 | 0.600 | 0.250 | reject |
| V20-only capture teacher, screen | 20 | 0.6000 | 0.800 | 0.400 | reject: White-led variance |

The immediate-capture probe itself rose from 3/20 correct captures for V20 to
11/20 for the candidate, demonstrating that the isolated target was learnable.
That did not translate to general strength or Black conversion. The conclusion
is not to add a more elaborate tactical gate: obvious capture/promotion facts
must be learned as part of the general policy. These artifacts are retained as
negative evidence. V20 remains incumbent.

---

## 15. Iterative bootstrap pipeline (2026-08-05)

`src/iterate.py` was rebuilt as a resumable, manifest-driven state machine:

`generate -> reanalyze -> process -> compose -> train -> checkpoint_screen ->
offline_gate -> binding_gate -> high_fidelity_gate -> self_skew -> promote`

The new design keeps the exact v19_B/V20 processed recipe as an immutable replay
anchor, mixes recent promoted generations, generates balanced league experience,
and uses deeper champion search on ordinary positions to create policy-only
teachers. Teachers are ranked by policy divergence, value change, and changed
top action; the pipeline reserves 60% for Black without selecting a tactical
motif. Conservative end-to-end fine-tuning starts from the incumbent. The
moves-left head is available but deliberately off for generation one.

Every phase records commands, outputs, logs, timings, and status in
`iterations/gen_NNNN/state.json`. Resume rejects statistical recipe drift. An
offline policy/value comparison runs before the binding gate and records
warnings, but is advisory by default: imperfect teacher imitation is not a
playing-strength oracle, so every successfully trained candidate reaches actual
games. `--reject-on-offline-regression` restores the legacy hard behavior. Only an
explicit `--promote-on-pass` can update `models/bootstrap/champion.json`, only a
full gate can promote, and no numbered release checkpoint is overwritten.
Numbered V21 still requires the existing owner playtest.

The first production candidate demonstrated why that distinction matters. It
improved held-out winner-sign accuracy by 1.53 points overall and 3.14 points on
Black, while policy top-1 fell 1.15 points overall and 1.52 on White. The old
1-point hard threshold rejected it without a game. Recovered play-testing then
showed exactly why games must decide: the first 40-game V20 leg scored 0.650
overall / 0.725 White / 0.575 Black, but the independent confirmation fell to
0.3875 / 0.575 / 0.200. Across all 140 gate games it scored 0.793 as White and
0.579 as Black. The candidate was correctly rejected for unstable real play,
not for imperfect teacher imitation.

The production loop now also honors the repository's measured eight-worker
default instead of silently applying the obsolete four-worker NN cap. On this
5060 Ti, prior 400-simulation measurement was 5.39 decisions/s at four workers,
7.11 at eight, and 7.38 at twelve; fourteen exhausted memory. Eight therefore
uses more of the available GPU while keeping useful failure headroom.

**Four-generation production result (2026-08-06).** Every successfully trained
generation received the binding game protocol. None promoted from its original
validation-selected checkpoint. Generation two reached 0.5875 against V20 but
fell to 0.5000 overall / 0.325 Black on confirmation; generations three and
four scored 0.4750 and 0.4875 against V20. Generation four nevertheless exposed
an actionable selection error: its preserved epoch two, not the proxy-selected
epoch four, passed an independent full binding gate at 0.525 / 0.600 / 0.450,
then 0.6875 / 0.800 / 0.575 on the fresh V20 confirmation. Across its 140 gate
games it scored 0.821 White and 0.650 Black.

The stronger 80-game, 800-simulation calibrated read correctly blocked
promotion: epoch two improved White by 0.275 and aggregate by 0.119, but Black
was 0.4375 versus the seeded V20 calibration's 0.475 (-0.0375). Epoch one's
same-opening 800-simulation match was worse at 0.600 White / 0.3125 Black.
Epoch two's 80-game self-match was also more color-skewed than V20: pooled White
score 0.781 versus V20's recorded 0.694. This is evidence of a general White
drift, not merely a weak conversion tail.

The pipeline now arena-screens every unique `selected_epoch_*.pt` checkpoint,
ranking by the minimum calibrated color delta before aggregate and Black
tie-breaks. A binding winner must additionally pass the calibrated 80x800
two-color confirmation. Thus all preserved models receive play, small screens
cannot promote by themselves, and the generation-four epoch-two false positive
would be rejected automatically.

A live native smoke run completed generate -> reanalyze -> process: two games
(one win per color), 200 self-play positions, two positions deep-searched, one
teacher retained, 201 raw rows / 402 augmented rows, and no illegal targets.

---

## 16. Next

1. Owner-playtest `bootstrap_gen5_teacher3200_full` epoch two. It has cleared
   every automated gate but remains a candidate; V20 remains the numbered
   incumbent until that decision.
2. Register the candidate as the bootstrap champion only after acceptance, then
   generate the next replay generation from it. Bootstrap acceptance is not
   automatically numbered V21.
3. Keep moves-left opt-in: the isolated arm improved White but did not clear
   the Black stability bar. Do not tune it into the gate from this result.
4. Keep the staged epoch arena and two independent V20 gates. Use general
   failure-position reanalysis rather than hand-authored tactical rules.

---

## 17. Bootstrap production contracts (2026-08-06)

The three prerequisites for cumulative model iteration are implemented:

1. Processed champion data is entered into a run-local `accepted_data.json`
   registry before training. Candidate rejection no longer discards its source
   self-play or deep-search teachers.
2. Replay preserves validation/test membership and smooths the fixed-size
   training index budget across side, true outcome, and corpus-derived
   material-phase thirds. The default square-root smoothing (`alpha=0.5`)
   avoids both raw-majority domination and full rare-stratum equalization.
3. Every training run evaluates the resume checkpoint on the same validation
   rows. Epochs are ranked by worst-color policy/sign gains over that fixed
   incumbent, with policy-CE and per-side top-1 guards also fixed to it. A run
   with no safe epoch ends as `rejected_training` with a structured report.

Explicit `--continue-after-reject` runs can now accumulate accepted data while
V20 remains champion; automatic promotion and numbered releases retain all
existing gates.

The production audit on 2026-08-06 found that standalone deep-search teacher
files were being independently split: 10/40 demo-generation-one teachers and
61/180 generation-two teachers crossed their source game's split. Teachers now
form a split group with the source game; the same reproducer reads 0/40 and
0/180. The audit also added no-progress timeouts to every worker phase, strict
generated-batch completion checks, atomic staging for reanalysis/replay,
non-overlapping phase seed ranges, full accepted-artifact hashes, a run-root
lock, stale-checkpoint archiving on training retries, and memory-mapped replay
loading. Run status now validates both the exact Windows PID column and process
creation time, so PID reuse cannot resurrect dead entries. Full unittest
discovery passed 522 tests at that point (562 as of 2026-08-06).

Open owner decisions: §7.4's hand-corrected label; the `combined_v16` copy;
the 23 legacy unreplayable games. The bar is no longer open.

---

## 18. Weighted deep-teacher recovery (2026-08-06)

The generation-five failure was data dilution, not evidence that V20 was the
wrong initialization. The full replay held 625,456 rows, while the 1600-sim
policy-only teachers represented only about 1--2% of enabled policy weight.
The unweighted candidate's strict 80x800 read gained +0.2375 White but changed
Black by -0.0875. Giving those teachers 4x policy weight produced a near-pass:
+0.025 Black / +0.0375 White at 80x800, but its fresh binding confirmation
fell to 0.500 overall / 0.275 Black. Restricting replay to anchor plus latest
generation was worse: all five full-screen finalists lost Black strength.

The controlled recovery kept full replay, V20 initialization, 4x teacher
weight, 12/5 training, and the architecture fixed, while increasing teacher
search from 1600 to 3200 simulations. The new pass sampled 8,000 ordinary
positions (60% Black), retained 4,000, and raised action-change rate from
77.9% to 86.4% and mean policy JS from 0.291 to 0.323. After augmentation and
replay balancing, 7,109 teacher rows contributed 6.01% of effective training
policy weight, essentially the same share as the prior 5.94% experiment.

All seven epochs received games. Epoch two won the full checkpoint screen at
+0.100 Black / +0.050 White / +0.075 overall. It then passed the independent
calibrated 80-game, 800-simulation read at **+0.0125 Black / +0.1375 White /
+0.075 overall**. Its full binding results were:

| leg | overall | White | Black |
|---|---:|---:|---:|
| V20, initial seed | 0.650 | 0.825 | 0.475 |
| ramp | 0.900 | 1.000 | 0.800 |
| heuristic | 0.950 | 1.000 | 0.900 |
| V20, fresh confirmation | 0.550 | 0.675 | 0.425 |

The 80-game self-match measured pooled White 0.5875 / Black 0.4125, reducing
V20's 0.3875 color gap to 0.175. On the 400-position Black promotion-defense
search deck, capture choice rose 0.640 -> 0.6675 and capture visit share rose
0.5232 -> 0.5678; refusals despite a higher capture Q rose 10 -> 14, so this is
a broadly positive diagnostic rather than a perfect conversion result. The
first 12-worker binding attempt encountered a transient CUDA error and emitted
no result; the exact protocol and seed completed at the measured stable
eight-worker default. Full discovery passed 522 tests at that point (562 as of
2026-08-06).

This checkpoint is the first bootstrap successor to clear every automated
gate. It remains an owner-playtest candidate, not V21. Evidence:
`gen5_teacher3200_full_checkpoint_screen_20260806.json`,
`gen5_teacher3200_full_epoch2_800_confirmation_20260806.json`,
`gen5_teacher3200_full_epoch2_binding_20260806.json`,
`gen5_teacher3200_full_epoch2_self_skew_20260806.json`, and
`promotion_defense_search_20260806_140554.json`.

---

## 19. Moves-left isolated A/B (2026-08-06)

With the fixed-architecture successor established, moves-left was tested as
one isolated change. The arm used the same V20 initialization, 3200-teacher
replay, seed, optimizer, 12/5 schedule, and per-epoch shuffle; only the
8,321-parameter auxiliary head and its 0.01 Huber loss were enabled. Trusted
remaining-length labels covered 71.7% of training rows.

All eight epochs received games directly against the fixed successor. Epoch
six won the full screen at +0.100 Black / +0.200 White / +0.150 overall, and
epoch seven also read +0.050 / +0.200. Epoch six then passed the independent
80x800 A/B at **+0.025 Black / +0.100 White / +0.0625 overall**. It did not
clear the absolute binding gate:

| leg | overall | White | Black |
|---|---:|---:|---:|
| V20, initial seed | 0.7125 | 0.900 | 0.525 |
| ramp | 0.825 | 0.975 | 0.675 |
| heuristic | 1.000 | 1.000 | 1.000 |
| V20, fresh confirmation | 0.6125 | 0.850 | **0.375** |

The one failure is binding: fresh-confirmation Black missed the unchanged
0.400 floor by half a point. The earlier epoch three was separately tested
because both its offline metrics and full screen were more Black-leaning; its
independent 80x800 A/B failed at -0.050 Black / +0.1375 White. The head learns
a real length signal and may improve White/general play, but this run does not
show a stable Black improvement. The fixed-architecture epoch-two successor
remains the owner-playtest candidate. Evidence:
`gen5_teacher3200_full_mlh_vs_control_checkpoint_screen_20260806.json`,
`gen5_teacher3200_full_mlh_epoch6_800_vs_control_20260806.json`,
`gen5_teacher3200_full_mlh_epoch6_binding_20260806.json`, and
`gen5_teacher3200_full_mlh_epoch3_800_vs_control_20260806.json`.

---

## 20. V21 promoted; the conversion thesis tested and rejected (2026-08-06/07)

### 20.1 V21

The owner playtested the gen-5 teacher-3200 epoch-two candidate ("it is a very
strong player") and promoted it as **`models/fresh_start_v21`**. The promoted
checkpoint's SHA-256 was verified byte-identical to the one the gate scored
(`06d89341...`); `arena_selected.pt` and `selected_epoch_002.pt` are the same
file. The gate bar moved to `vs_v21` and **no threshold moved** -- floor 0.40,
aggregate 0.50, sims 400 unchanged. Two tests pin the bar (`test_gate_protocol`,
`test_post_e5_driver`) and both required a deliberate edit, which is what they
exist for. Manifest: `models/fresh_start_v21/promotion_manifest.json`.

Recorded caveats: Black gained **+0.0125** at 80x800, inside noise, so the
measured improvement is White-weighted like every recent arm; and the gain
traces to the teacher recipe (3200 sims, 4x weight), not to iteration --
generations one through four on the same recipe were all rejected.

### 20.2 Progression tournament

Seven models, 21 pairings, 30 games each, 630 games in 36.9 min
(`benchmarks/tournament_progression.json`, `tools/tournament.py`):

| player | overall | as White | as Black | colour gap |
|---|---:|---:|---:|---:|
| v21 | 0.736 | 0.867 | 0.606 | 0.261 |
| lc0b_attn_ema | 0.728 | 0.872 | 0.583 | 0.289 |
| v20 | 0.700 | 0.845 | 0.556 | 0.289 |
| v19_B | 0.486 | 0.750 | 0.222 | 0.528 |
| v19 | 0.389 | 0.572 | 0.205 | 0.367 |
| v18_ramp | 0.244 | 0.406 | 0.083 | 0.322 |
| v17 | 0.217 | 0.372 | 0.061 | 0.311 |

**The ladder is monotonic with no inversions.** Black rose 0.061 -> 0.606, the
largest single step landing on the attention+EMA change (v19_B -> v20). But
**the colour gap stopped closing**: 0.528 -> 0.289 -> 0.261, so the last two
promotions raised both colours together rather than fixing the asymmetry.
v2-v5 could not be entered; they predate the current `DualHeadNet`.

### 20.3 Leaf-parallel width settled

Batch 16 vs batch 64, same net, 1600 sims, 200 games: **0.720 to batch 16 at
6.22 SE** -- roughly 165 Elo (`benchmarks/match_fresh_start_v20_vs_fresh_start_v20_20260806_185310.json`).
Wide in-tree batching is badly harmful here. An earlier recommendation of 64
was made on *fill rate* (74-91% of the requested batch), which measures how
full a batch is and says nothing about whether its leaves were well chosen.
`benchmark.py`'s note that only a match settles it was correct.

### 20.4 The unconverted-game finisher

`tools/finish_unconverted.py` had never been run and was broken: it replayed
each game forward from its opening record, but generation retains records
side-selectively, so consecutive records are not consecutive plies and every
game returned "unreplayable". It now resumes from the final record (`fen` plus
`half` pin the position exactly) and runs 8-way parallel, verified
byte-identical to serial output.

Two counting corrections matter for reading the results. `extra_turns` counts
`turn_count` increments and a full round increments it **twice**, so the flag
is 2x the Black moves allowed. And deduplicating by file content was too weak:
631 matching files hold 208 content-distinct games but only **143 distinct
resume positions**, and resume state is the whole experiment.

| run | Black | White | budget | rate |
|---|---|---|---|---|
| v20 | 1600 | 800 | 20 moves | 23.8% (34/143) |
| v21 | 1600 | 800 | 80 moves | **39.9%** (57/143) |
| v21 | 1600 | 1600 | 80 moves | **39.9%** (57/143) |

Raising the budget was decisive: **24 of the 57 conversions needed more than 20
Black moves**, out to 69, and mean depth went 5.5 -> 22.4. The symmetric run
converted the same count and **46 of the same positions**, so doubling White's
search changes which marginal positions fall, not how many. **~40% of the
`-0.5` corpus is genuinely won against a full-strength opponent.**

### 20.5 The thesis, tested

The standing argument was that `-0.5` collides in the value target with real
wins, and that supplying decisive conversions would move Black. Tested
directly: 57 conversions stamped `value_weight`/`policy_weight` 4 (the
sanctioned per-record lever -- `data_processor` bans source quotas by design),
composed additively onto v21's exact five sources at 12.25% effective share,
trained from v21 on the unchanged recipe.

**It failed.** Offline, only epoch 1 improved (+0.0071) before five straight
declines into early stopping. At the gate, every leg passed and the
confirmation replay did not: **Black 0.350 < 0.400**, aggregate 0.4875
(`benchmarks/gate_v22_conversions_w4_20260807_001755.json`). The symmetric
finisher rules out the obvious excuse -- the conversions are wins against a
full-strength White, not against a handicapped one.

The remaining excuses were then tested and all failed:

| conversion arm | bar leg | confirmation | confirm Black |
|---|---:|---:|---:|
| weight 4, lr 1e-4 | 0.6250 | 0.4875 | 0.350 |
| weight 1, lr 1e-4 | 0.5500 | 0.4750 | 0.375 |
| weight 4, lr 3e-5 | 0.6750 | 0.4250 | 0.325 |

A 4x change in effective share moved the outcome 0.025, and a 3x slower
learning rate made it worse. All three cleared the first bar leg and collapsed
on the fresh opening seed.

**Status: the conversion axis is closed.** The *data* claim survives -- ~40% of
the `-0.5` corpus is genuinely won against a full-strength opponent, confirmed
by the symmetric finisher -- and the *training* claim is rejected across three
independent variants. Better data on this axis is not what Black is short of.

Two known imperfections do not rescue it: 22 of the 57 games still carry a
competing `-0.5` original in the anchor (which weakens rather than reverses the
correction), and 10,669 rows may simply be too few at any weight -- but weight
1 and weight 4 landing 0.025 apart argues the share was never the binding
constraint. **Do not revisit without a genuinely new reason**; four attempts
(v22 plus three sweep arms) have now failed the same leg the same way.

### 20.6 Infrastructure

- `tools/tournament.py` -- round-robin with per-colour splits, resumable.
- `tools/overnight_sweep.py` -- unattended train->gate->record queue,
  resumable, deadline-aware, never touches a threshold.
- `tools/queue_after.py` -- chains a run behind another by name.
- `runs.py`: `status` went 13.7s -> 1.19s (one `tasklist` snapshot instead of
  one spawn per record); added `stop` (tree-kill) and `prune` (archive).

Two liveness bugs, both of which had already caused real damage. `runs.py`
treated an **unqueryable recycled pid as alive** -- a finished run's pid became
`svchost.exe` and read as RUNNING forever, which would have made `queue_after`
wait for it indefinitely and silently skip a night's work. And plain `taskkill`
on a `queue_after` wrapper **orphans the child it already launched**: an old
finisher kept writing into the same output directory as its replacement for 24
minutes, and a match fired early into a busy GPU.

`gate.py` seeded legs `seed + 100*i` while `run_match` requires >=100,000
separation. The current 40/40/20 legs are disjoint (verified: zero shared
per-game seeds, bar and confirm 423,223 apart), but the margin was 81 and any
leg past ~100 games would have overlapped -- two legs replaying the same
openings read as independent agreement. The stride is now derived from leg
size, with tests at both current and 400-game sizes.

### 20.7 What the confirmation-leg pattern is

Confirmation legs look systematically worse than first legs. Across all five
gate runs holding both, confirm was worse on Black in **3 of 5, mean -0.035**,
and `v19_B` went *up* +0.200 -- noise at ~0.08 SE per leg. The real mechanism
is conditional selection: the confirmation only runs if the first legs pass, so
every observed confirmation is conditioned on an above-threshold first leg and
regresses to the mean. That is why it is the valid test, not a suspect one.

---

## 21. Overnight replicate sweep (2026-08-07)

Eleven-plus arms run unattended through `tools/overnight_sweep.py`, each
training from v21 on a **byte-identical recipe** except for its one change and
gating through unmodified `gate.py`. Arms cost ~15 min, not the 85 estimated,
so the spare capacity went to replicates rather than to more first-attempt
ideas.

### 21.1 The finding that matters: the gate's per-side floor is low-powered

The per-side floor is checked on **half a leg** -- a 40-game leg is 20 games per
colour, **SE 0.112**. Simulating the floor against a known true rate:

| true Black rate | clears 0.40 on one leg | clears **both** v21 legs |
|---|---:|---:|
| 0.425 (parity with v21) | 67.3% | **45.0%** |
| 0.450 (better than v21) | 74.8% | **55.9%** |
| 0.500 | 86.9% | 75.5% |

**A candidate genuinely better than the incumbent on Black fails the gate
roughly half the time on sampling noise alone.** Demonstrated live: the same
recipe at seed 42 scored Black 0.425/0.450 and PASSED; at seed 43 it scored
0.375 and FAILED, with a *higher* overall bar leg (0.5375 vs 0.5125). Only
Black moved, and Black is where the noise is.

This is **not** an argument to lower the floor. The threshold stands. The fix
is sample size -- more games per leg, or pooled replicates -- neither of which
touches the protocol. It does mean a single gate verdict has been carrying more
weight than it can bear.

**It does not mean the archive holds misjudged winners.** `tools/gate_triage.py`
ranks every binding FAIL by how narrowly it missed: 18 of 27 missed by under one
standard error, which sounds like a reservoir of rejected candidates. It is not.
Thirteen of those were judged against v21 and have since been re-tested at 320
and 800 games -- all genuinely null. The remaining near-misses were judged
against **superseded bars** (v19, v19_B, ramp); an arm marginal against v19_B is
not a candidate against v21, which beats v19_B 0.736 to 0.486 in the ladder.
Only two were judged against v20. So the low power is real and matters for
*future* decisions, while the record of past rejections has held up wherever it
could be checked.

### 21.2 Capture-WDL: two passes, no measurable effect

`--aux-wdl-head --wdl-target capture_result --wdl-loss-weight 0.03`, the only
signal that had ever moved Black for a principled reason.

| arm | verdict | bar | confirm | confirm Black |
|---|---|---:|---:|---:|
| capture_wdl_w003 (seed 42) | PASS | 0.5125 | 0.5250 | 0.450 |
| capture_wdl_plus_black_policy | PASS | 0.5500 | 0.5250 | 0.425 |
| capture_wdl_w003_seed43 | FAIL | 0.5375 | -- | -- |
| capture_wdl_w003_seed44 | FAIL | 0.5250 | 0.4875 | 0.375 |
| capture_wdl_w001 | FAIL | -- | 0.4500 | 0.400 |

Pooled over the four capture-WDL arms, 160 bar-leg games
(`tools/replication_summary.py`):

| pool | score | reference | z |
|---|---:|---:|---:|
| overall | 0.5312 | 0.500 | +0.79 |
| White | 0.6312 | 0.600 | +0.56 |
| Black | 0.4313 | 0.425 | +0.11 |

**Final, over all eight capture-WDL arms and 320 bar-leg games:**

| pool | score | reference | SE | z |
|---|---:|---:|---:|---:|
| overall | 0.5234 | 0.500 | 0.0280 | +0.84 |
| Black | 0.4469 | 0.425 | 0.0395 | +0.55 |
| confirmation (6/8 arms) | 0.4917 | 0.500 | 0.0323 | -0.26 |

**The recipe is a null.** Point estimate +0.023 (~16 Elo); effects above ~+0.08
are excluded. The confirmation pool sits *below* parity despite being the
biased-upward statistic. Four replicate seeds of the identical recipe went
PASS, FAIL, FAIL, FAIL -- one in four, which is what 21.1 predicts at parity.

The weight sweep is likewise flat: bar legs 0.5250 / 0.5125 / 0.5375 / 0.5500
at weights 0.01 / 0.03 / 0.06 / 0.10, with only 0.03 passing. **0.03 was not a
peak, it was the lucky draw.**

The seed-42 checkpoint that opened the sweep with a PASS is the one the
high-power 800-game match tests directly; the recipe-level answer above is
already settled.

### 21.3 Two analysis errors worth not repeating

**Per-colour scores must reference the incumbent's self-match, not 0.50.** A
model identical to v21 does not score 0.50 per colour against it -- White is
structurally advantaged, and v21's self-match splits White 0.600 / Black 0.425.
Judged against 0.50 the capture-WDL pool read "White z=+2.45, distinguishable";
against the correct reference it reads z=+0.56, nothing. The wrong baseline
manufactures a White gain for every candidate.

**The bar leg alone is not sufficient either.** It is unbiased with respect to
*selection* (every arm plays it), but the conversion arms pooled to bar 0.6125
(z=+2.01) while their confirmations pooled to 0.4500 -- high on the first
opening set, parity on the second. Both legs together is the estimate.

### 21.4 The v21 self-match baseline was wrong, and it mattered

Every per-colour judgement references the incumbent's self-match, because an
identical model does not score 0.50 per colour (21.3). That reference had been
measured on **80 games**. Re-measured at 800:

| | White | Black | colour gap |
|---|---:|---:|---:|
| 80 games (used at promotion) | 0.600 | 0.425 | 0.175 |
| **800 games (2026-08-07)** | **0.6675** | **0.3513** | **0.316** |

Black was wrong by **0.074**, near two standard errors, and three things follow.

**The v21 promotion's headline supporting claim does not survive.** The manifest
records a gap narrowing from v20's 0.3875 to 0.175 -- called at the time "the
most encouraging signal available". v21's true gap is 0.316, and v20's 0.3875
was also an 80-game estimate. The narrowing was mostly noise. The promotion
stands (gate cleared on disjoint seeds, owner playtested); this one supporting
figure should not be cited. The manifest is annotated in place with the
originals preserved.

**A three-sigma scare evaporated.** The seed-42 checkpoint's high-power Black
score of 0.3500 read z=-3.00 "significantly worse" against the old reference;
against the true 0.3513 it is **z=-0.04**, exactly at parity. Treating a
measured reference as exact can manufacture a three-sigma result in either
direction. `tools/replication_summary.py` now propagates reference error.

**Per-colour comparisons are confounded by opening set.** Candidate bar legs run
on gate seed 20260801; the self-match ran on 27182818. Pooled per-colour reads
flip sign depending on which baseline they reference. **Only the
colour-balanced overall score of a direct match is reference-free** -- colours
alternate within the same games, so equal strength gives exactly 0.50 by
construction. Future comparisons should measure candidate and baseline on
identical opening seeds.

### 21.5 The decisive measurement

`capture_wdl_w003` (seed 42, the arm that passed) vs v21, **800 games**:

| | score | SE | |
|---|---:|---:|---|
| overall | **0.4800** | 0.0177 | 95% CI [0.445, 0.515] |
| White | 0.6100 | 0.0250 | 194-106-100 |
| Black | 0.3500 | 0.0250 | 83-203-114 |

**Not better than v21.** And the cleanest confirmation of 21.1 in the whole
run: this checkpoint read Black **0.425 / 0.450** across its two 40-game gate
legs and **0.3500** over 400 -- same model, same opponent. It passed the gate
on a lucky Black sample.

### 21.6 Black-policy weighting

`--black-policy-weight 1.75` (`BLACK_WEIGHT_BALANCED`, never previously used):
FAIL at 0.4875 on the bar leg. Black stayed at 0.425, exactly v21's self-match
baseline, while White fell 0.600 -> 0.550. It did not help Black; it cost
White. Combined with capture-WDL it was neither additive nor harmful.

---

## 22. The value head is flat in the won region (2026-08-07)

Owner observation, from watching v21 self-play at 6400 sims: "a lot of draws
and super long black wins are shuffling in a black dominant position." Measured
across the fifteen exported games:

| outcome | mean plies with White reduced to a BARE KING |
|---|---:|
| White wins | 1.0 |
| Black wins | **70.8** |
| Draws | **97.8** |

Three of the five draws reach a bare White king by ply 54-78 and then shuffle
**150-171 further plies** with 20-35 points of Black material on the board. The
draws are not fortresses and not close games: they are completely won positions
Black cannot finish. One Black "win" spent 150 plies bare before landing the
capture.

**Why there is no reason to hurry.** v21's value output across the shuffling
stretch of the 171-ply draw:

| plies since White went bare | 0 | 43 | 86 | 129 | 171 |
|---|---:|---:|---:|---:|---:|
| v21 value | -0.679 | -0.678 | -0.763 | -0.688 | **-0.677** |

Standard deviation **0.039** over 172 plies. The 150-ply Black win is flatter
still (sd 0.020) and only jumps to -1.000 when the capture is actually on the
board. **The value head knows Black is winning and cannot tell 171 plies from
one.** MCTS ranks moves by value; when every move returns -0.68 there is
nothing to rank, so the search shuffles.

Note the values sit at -0.68 to -0.92, *not* at the -0.5 ramp floor -- the head
is not merely saturated, it is genuinely confident and genuinely gradient-free.

**This is a different failure from section 20.5.** That one held the `-0.5`
labels to be factually wrong, and was rejected: supplying correctly-labelled
conversions failed at three weights and learning rates. This one says the
labels can be perfectly correct and the *objective* still flat in the winning
region, because `result * gamma^min(plies_to_end, horizon)` with horizon 60
assigns the same target to every position more than 60 plies from the end.
Correct data cannot create a gradient a target does not have -- which is a
coherent explanation for why 20.5 came back null.

### 22.1 Correction: the head is not flat, it is shallow

The reading above was measured along a *shuffling* trajectory and does not
support the causal claim made from it. If Black shuffles, the position barely
changes, so the value barely changes -- flatness there is as easily the
consequence of shuffling as its cause. The missing control was positions where
progress genuinely happens.

`tools/value_gradient_probe.py` supplies it, over 4,000 bare-king positions
from `finished_conversions` whose true distance to capture is known:

| plies to capture | n | mean value |
|---:|---:|---:|
| 0-5 | 156 | -0.6875 |
| 6-15 | 260 | -0.5687 |
| 16-30 | 390 | -0.5663 |
| 31-60 | 743 | -0.5453 |
| 61-120 | 1261 | -0.4652 |
| 121+ | 1190 | -0.3882 |

Spearman(plies_to_end, value) = **+0.42**, monotonic, spread **0.299**. **v21's
value head does track distance to capture.** "The objective is flat in the won
region" is withdrawn.

What survives is narrower and better supported: within-bucket **sd is ~0.20**
while adjacent buckets differ by only 0.02-0.09, so the progress trend is real
but **shallow relative to positional noise**. MCTS compares candidate moves
inside one position; a signal that weak is swamped there even though it is
visible across thousands of positions. The defect is resolution, not absence.

**What follows.** A moves-left head sharpens exactly this -- it was rejected
(section 19) on a single confirmation leg at Black 0.375, a coin flip at 20
games per side by 21.1, and it was masked on 25% of rows, 99.1% of them
Black-leaning. The conversion corpus processes with **zero** masking. That arm
is still worth running, but as "sharpen a weak gradient", not "create a missing
one" -- and the probe above, not a gate, is the sensitive way to read it.

---

## 23. Three gate passes, three nulls (2026-08-07)

Every arm that cleared the gate today was then played 800 games against v21:

| arm | gate | pooled bar legs (80 games) | 800-game match |
|---|---|---:|---:|
| capture_wdl_w003 | PASS | 0.5375 (z=+0.67) | **0.4800** |
| mlh_conversions | PASS | 0.5375 (z=+0.67) | **0.4825** |
| capture_target | PASS | 0.5750 (z=+1.34) | **0.5019** |

**Three for three.** Section 21.1 derived this from the per-side floor being
checked on 20 games (SE 0.112); this is the demonstration. **A gate PASS at the
current leg size means "not clearly worse", not "better."** v21 itself was
promoted on such a pass. The thresholds are sound; the sample behind them is
not, and the fix is games, not thresholds.

### 23.1 The owner's -0.5 hypothesis

Owner, 2026-08-07: "I'm suspicious that knowing the 225 is a -0.5 or whatever
is causing the model to settle."

Two mechanisms were checked. The network **cannot** see the clock -- the 17
input channels carry pieces, turn, half-move, rank and pawn progress, with no
turn-count plane -- and at 3200 sims the cap is far beyond search horizon
mid-game. But the *target* rewards settling:

| what actually happened | rows | mean `game_result` target |
|---|---:|---:|
| Black captured the king | 262,034 | -0.8414 |
| no capture, hit the cap | 156,626 | **-0.4527** |
| White captured the king | 206,796 | +0.9001 |

**Reaching the cap with material intact is worth 54% of a win**, and 99.7% of
capped rows receive a non-zero Black-favourable target. It requires no capture,
only that the heuristic still favours Black -- which repeating a position
preserves perfectly, while playing for a win risks it. `CONTEXT.md` records the
2026-08-03 captures-only ruling and states the +-0.5 *training* label was
deliberately left unchanged; this is that gap, measured.

**Tested and it does not pay off.** `--target capture_result` (capped games
score 0) measurably changed the value function in the predicted direction --
distant Black-dominant positions repriced from -0.3882 to -0.2711 on
`value_gradient_probe` -- and the play effect was Black **+0.0175 +- 0.0354**,
White **-0.0325 +- 0.0354**, netting 0.5019 overall. The predicted trade
appeared and was too small to matter. The interval permits a Black gain up to
~+0.09, so this is "no large effect", not "no effect".

### 23.2 What is now closed

Conversion data (3 variants), capture-WDL (9 arms, 320 games), moves-left (3
independent looks), Black-policy weighting, and the capture-only target. Also
closed: search depth (section 21 depth study, 8x buys nothing) and the
colour gap as a target (0.316 -> 0.345 at 8x search).

## 24. Paired opening books (2026-08-14/15)

### 24.1 The measurement was candidate-dependent

`match.py` diversified NN-vs-NN games by playing the first 16 plies at
temperature 0.5, sampled from the engine's own visit counts. It exists because
two temp-0 engines replay one identical game however the RNG is seeded. But the
opening is drawn from **the candidate's own policy**, so two candidates
measured against the same incumbent at the same seed start from different
positions. Their scores are independent samples and their variances add. That
is why ranking 14 checkpoints costs as many games as generation does.

Stage costs for generation 7, which is what prompted the work:

| stage | time | share |
|---|---:|---:|
| generate (500 games @700) | 44.7m | 31% |
| screen (1,360 games @400) | 44.6m | 31% |
| train (15 epochs) | 28.7m | 20% |
| reanalyze (8k @3200) | 14.7m | 10% |
| gate (460 games @400) | 9.7m | 7% |

### 24.2 What landed

`tools/make_book.py` freezes ply-16 positions from a model's own self-play;
`match.py --book` plays each entry twice with colours reversed and reports a
**paired standard error** over pairs rather than games. The pairing removes
opening-draw variance and makes the per-colour comparison paired, which matters
because White and Black games previously used deliberately disjoint seeds and
so came from unmatched opening sets.

A book entry is FEN **plus** `white_half_pending` (board.turn stays WHITE
across White's pending half) **plus** `turn_count` (rebuilding from FEN
restarts it at 0, which would hand a ten-turn-deep position the full 150 again
and lower its draw rate). Both travel in the entry.

Two traps were caught by tests before any run used them:

- **The gate's confirmation leg would have been a tautology.** Under a book,
  independence lives in the *entry index*, not the seed. The confirmation
  replay uses a fresh seed, which does nothing when the openings are fixed; it
  would have replayed the bar leg's openings and agreed by construction while
  reporting `confirmed: true`. `gate.book_leg_offsets()` now allocates each leg
  a disjoint block, confirmation last, and validates size before playing.
  The full gate needs **220 entries**.
- **The screen had the same trap** between probes and finals. Disjoint blocks
  now keep the high-power confirmation independent of the selection that chose
  the finalists.

Deliberately not copied from TCEC: **balance curation**. Their books select
positions where neither engine is winning. Monster Chess is not balanced, and
selecting for Black-playable openings would make the gate measure a different
game from the one the corpus is drawn from. Only the pairing and the freezing
were taken.

The anchor leg keeps sampled openings: it is the heuristic yardstick and its
value is comparability with every anchor score on record.

### 24.3 Direct diversity measurement and corrected book (2026-08-15)

The failed builds did not reveal a low-entropy arena. The builder was applying
temperature 0.5 at **two** layers: it sharpened network logits before search
and then sampled root visits at 0.5. The arena only does the latter. That
unintentionally made builder walks far more deterministic than match walks.
After matching the arena exactly, the mixed v21b/v21/gen7 book produced 800
entries in 5.6 minutes; its three sources needed 323 / 377 / 335 completed
walks to contribute 267 distinct states apiece.

The new match instrumentation measured the historical sampler directly on an
800-game v21b-v21 re-anchor:

| measure | result |
|---|---:|
| distinct states | 650 / 800 (81.25%) |
| distinct state + candidate-colour games | **684 / 800 (85.5%)** |
| maximum state multiplicity | 34 |
| sampled score | 0.5225 (W 0.6613 / B 0.3837) |

The effective-size correction is `sqrt(800/684) = 1.08`, so historical
standard errors from this sampler are about **8% too small**, not 2x too small.
Some concentration is real—one state occurred 34 times—but the feared
fourfold collapse in sample size did not occur.

The paired-book replay agreed almost exactly: v21b scored **0.5238** (W 0.6212
/ B 0.4263), with paired SE 0.0121 over 400 complete pairs. Every one of its
800 state-plus-colour assignments was unique. The close 0.5225 / 0.5238
agreement validates the new regime for ranking while preserving the important
warning that book and sampled side scores are distribution-specific. Evidence:
`match_v21b_vs_v21_sampled_reanchor_20260815.json`,
`match_v21b_vs_v21_book_reanchor_20260815.json`, and
`books/mixed_v21b_v21_gen7_p16_20260815.json`.

## 25. Policy targets were 99.84% zeros (2026-08-15)

Measured on the generation-8 corpus, 833,418 rows:

| | dense | actual content |
|---|---:|---:|
| `policies.npy` | 13.65 GB | **0.040 GB** |

Density **0.16%** — mean 6.6 non-zero entries per 4096-wide row, median 5, max
35 over a 300-row sample. The dense array was 344x its own content.

**The disk size was not the mechanism.** `epoch_train_idx = train_idx` and
`policies[epoch_train_idx]` re-materialises the full train split **every
epoch**, so each epoch pulled ~11.5 GB out of a memory-mapped 13.65 GB file in
shuffled order. That is a page-cache cliff, not a linear cost:

| generation | policy array | min/epoch |
|---|---:|---:|
| 7 | 11.9 GB | 1.91 |
| 8 | 13.65 GB | **3.58** |

+14.5% rows, +87% time per epoch. Under an accumulating corpus it worsens every
generation.

`src/sparse_policy.py` stores CSR (uint16 indices, float32 values, int64
offsets) in `policies_sparse.npz`. **The storage changed; the interface did
not** — `SparsePolicyTargets` answers `len`, `.shape`, `.ndim`, `.dtype` and row
indexing exactly as the dense array did and returns dense rows, so no consumer
in `train.py` changed. `open_policies` reads either format and prefers sparse,
so a stale `policies.npy` cannot silently win. Nothing already on disk was
rewritten; `compose_processed.py` converts legacy dense sources once, on the
way in.

Parity is asserted with **zero tolerance**, because a silent divergence here
would not crash — it would train on different targets and surface as an
unexplained regression weeks later. Verified on real gen-8 data: 4,000 sampled
rows round-trip bit-identical, and a shuffled batch-256 gather bit-identical.
Corpus **17.3 GB → 3.68 GB**.

**Batchwise loading fixed (2026-08-15).** `_IndexedBatchDataset` now receives
the exact local permutation produced by the existing DataLoader sampler, maps
it to corpus rows, and densifies only the requested policy batch. At batch 256
that allocation is 4 MB instead of ~11.5 GB. Its parity test uses shuffled,
repeated indices plus WDL, moves-left, policy/value weights and legal masks;
every tensor and the complete batch order are bit-identical to the legacy
loader. Dense legacy corpora deliberately retain their established eager path
because random batch reads from a 13 GB dense memmap are slower; newly composed
corpora use CSR and the bounded path. Generation 9 is the first production
timing and strength read.

## 26. Generation 8: paired re-screen and confirmation failure (2026-08-15)

The original sampled-opening screen selected epoch 10. That ranking is retained
as historical evidence, but a corrected mixed-provenance paired book changed
the nominee to **epoch 7**. Every checkpoint received a 40-game paired probe;
five finalists then received 200 games on the same position block. Against the
Gen7 self-calibration (W 0.6900/B 0.3100), the full finalists were:

| finalist | dBlack | dWhite | dAll |
|---|---:|---:|---:|
| epoch 08 | +0.070 | +0.010 | +0.040 |
| **epoch 07** | **+0.095** | **+0.040** | **+0.068** |
| epoch 05 | +0.080 | −0.025 | +0.027 |
| epoch 14 | −0.010 | −0.045 | −0.028 |
| epoch 04 *(offline pick)* | +0.100 | −0.035 | +0.032 |

Epoch 7 won on the binding ranking key: its weaker colour still improved by
0.040. The paired gate used a disjoint opening block:

```
Gen7 initial   200  all 0.5600  W 0.705  B 0.415
v18 ramp        40  all 0.9000  W 1.000  B 0.800
anchor          20  all 1.0000  W 1.000  B 1.000
Gen7 confirm   200  all 0.5175  W 0.685  B 0.350  ← FAIL
```

It provisionally passed, then failed only the fresh-confirmation Black floor.
Across all four legs it scored W 0.7348/B 0.4457 over 230 games per colour,
but the protocol correctly treats independent agreement as binding. Gen8 is
useful replay, not a numbered candidate. Artifact truth is
`screen_gen8_book_20260815.json` and `gate_gen8_book_20260815.json`.

## 27. Generation 9: the bootstrap loop produces a clean successor (2026-08-15)

Generation 9 generated from the unbeaten Gen7 bar. Its 500 games at 700 sims
yielded 51,453 positions; 8,000 positions were reanalysed at 3,200 sims and
4,000 kept. The accumulated from-scratch corpus contains **936,366 rows**.
Sparse policy storage kept its policy artifact to 45.2 MB instead of a
multi-gigabyte dense array, and batchwise densification reduced training to
30.9 minutes. The unchanged 30-epoch/10-patience v20 recipe stopped at epoch
14; offline validation selected epoch 4.

Paired play did not select the offline checkpoint. After probes of all 14
epochs, the four finalists scored against the Gen7 calibration
(W 0.6350/B 0.3650):

| finalist | dBlack | dWhite | dAll |
|---|---:|---:|---:|
| epoch 08 | +0.045 | +0.080 | +0.063 |
| epoch 04 *(offline pick)* | +0.090 | −0.025 | +0.033 |
| **epoch 06** | **+0.085** | **+0.050** | **+0.068** |
| epoch 05 | +0.140 | +0.005 | +0.073 |

Epoch 6 was selected because its weaker colour improved by 0.050. Epoch 5 had
the highest aggregate but was almost entirely a Black trade; the ranking did
what it was designed to do.

The binding gate then passed without changing any threshold:

```
Gen7 initial   200  all 0.5950  W 0.720  B 0.470
v18 ramp        40  all 0.8500  W 0.975  B 0.725
anchor          20  all 1.0000  W 1.000  B 1.000
Gen7 confirm   200  all 0.5575  W 0.705  B 0.410
```

The two independent Gen7 reads both clear aggregate and per-side floors. Pooled
across every gate leg, Gen9 scored **W 0.7478/B 0.4891** over 230 games per
colour. This is the first clean successor produced by the accumulated,
from-scratch bootstrap process and is the leading **v23 candidate**. Numbering
and the owner's playtest remain explicit promotion steps; no release directory
has been created and no historical model has been overwritten.

Two post-gate diagnostics make that result easier to interpret. A 400-game
paired self-match measured the underlying game skew at **W 0.6675/B 0.3325**,
slightly less extreme than v20's old 0.6938/0.3063 sampled baseline. A direct
400-game paired match against v21b scored **0.5763 overall, W 0.6975/B 0.4550**
with paired SE 0.0157. Gen9 is therefore not merely transitive through Gen7;
it directly and significantly outplays the historical gate on both practical
colour assignments.

## 28. Generation 10: accumulation is not monotonic (2026-08-15)

Generation 10 generated from Gen9 and added 55,549 positions from 500 games,
plus the same 8,000/4,000 Black-heavy deep-teacher pass. Its increment contains
111,098 processed rows; the accumulated corpus reached **1,047,422 rows**.
Training retained the exact Gen9 recipe, took 37.1 minutes, and stopped at
epoch 15. Offline validation and the paired arena both selected epoch 5.

None of the four 200-game finalists improved both colours over Gen9:

| finalist | dBlack | dWhite | dAll |
|---|---:|---:|---:|
| epoch 04 | +0.025 | −0.055 | −0.015 |
| epoch 12 | +0.000 | −0.060 | −0.030 |
| epoch 08 | +0.025 | −0.025 | +0.000 |
| **epoch 05** | **+0.005** | **−0.020** | **−0.008** |

The gate agreed with the screen rather than rescuing it:

```
Gen9          200  all 0.5225  W 0.660  B 0.385  ← FAIL, floor 0.40
v18 ramp       40  all 0.8250  W 0.975  B 0.675
anchor         20  all 0.9750  W 1.000  B 0.950
```

Across these legs it scored W 0.7346/B 0.4731, but the binding Gen9 Black leg
failed and no confirmation was earned. This is not evidence that Gen10 data is
bad; it is evidence that blindly appending one more equal-recipe generation is
not guaranteed to improve the model. Gen10 remains useful replay and a clean
negative result. Gen9 remains the successor candidate.

### 28.1 Controlled training-seed replicate

To distinguish a data/recipe ceiling from one unlucky optimization path, the
same Gen10 corpus and recipe were trained again with seed 43. Offline validation
selected epoch 6; paired play selected epoch 11. Two full finalists improved
both colours on the screen block:

```
epoch 11  dBlack +0.035  dWhite +0.015  dAll +0.025
epoch 07  dBlack +0.025  dWhite +0.015  dAll +0.020
```

That promising read did not survive the untouched binding block. Epoch 11
scored **0.4925 overall, W 0.630/B 0.355** against Gen9, failing both aggregate
and Black thresholds. The replicate proves that seed variance is large enough
to change checkpoint rankings, but does not rescue Gen10. A future training
study should treat seed as a nuisance variable and require cross-seed or
independent-book robustness; repeatedly drawing seeds until one passes would
just overfit the gate.

### 28.2 Hyperparameter tuner brought forward

`tools/tune_training.py` is no longer hard-wired to the old v19_B corpus,
v20 bar, and convolutional policy head. It now accepts corpus/bar paths, the
current attention architecture, EMA and memory-map settings, plus a paired book
and reserved offset. Fidelity rungs use disjoint book slices while every trial
within a rung sees identical positions. The objective remains Black-first with
a White-collapse penalty, and the normal untouched binding gate remains
decisive. This makes the next recipe search reproducible without pretending
offline loss is playing strength or manually trying isolated settings.

## 29. Depth redistributes the colour gap without changing who wins (2026-08-15)

Three paired matches at **1600 sims**, 120 games each, on reserved book blocks
500-679:

| | overall | White | Black | paired SE |
|---|---:|---:|---:|---:|
| Gen9 vs v21b @1600 | 0.6000 | 0.7000 | **0.5000** | 0.0280 |
| Gen9 vs v21b @400 | 0.5763 | 0.6975 | 0.4550 | 0.0157 |
| Gen9 vs Gen7 @1600 | 0.5750 | 0.6833 | 0.4667 | 0.0261 |
| Gen9 vs Gen7 @400 | 0.5763 | 0.7125 | 0.4400 | — |

**The ordering survives 4x the search.** Overall scores barely move; what moves
is the split. Black gains in all three (+0.045, +0.027, +0.092) while White is
flat or down. Both engines are deepened equally, so the correct statement is
about the game rather than the model: **deeper search benefits whichever side
plays Black.**

### 29.1 A confound in the first design, and the control that fixed it

The 400-sim references used book entries 0-200; the 1600 runs used 500-560 and
620-680 — different openings. Every comparison confounded simulations with
which positions were drawn. Re-running self-play at 400 sims **on the 1600 run's
own block** decomposed it:

| Gen9 self-play | White | Black | gap |
|---|---:|---:|---:|
| block 0-200 @400 (reference) | 0.6675 | 0.3325 | 0.3350 |
| block 620-680 @400 (control) | 0.6250 | 0.3750 | 0.2500 |
| block 620-680 @1600 | 0.5333 | 0.4667 | **0.0667** |

A third of the apparent effect was the block. The sims effect survives: on
identical positions the gap closes 0.250 -> 0.067. v21b run through the same
control closes 0.133 -> 0.033 — **both narrow by about 74%**.

For self-play under a book the paired SE is 0.0000 *by construction* (identical
engines replay the same game), so it says nothing about between-block variance.
Measuring a model's true skew needs several disjoint blocks, not more games in
one.

### 29.2 The two models close the gap by opposite mechanisms

| | White wins | Black wins | draws |
|---|---:|---:|---:|
| Gen9 @400 | 25 | 10 | 25 |
| Gen9 @1600 | 19 | **15** | 26 |
| v21b @400 | 24 | 16 | 20 |
| v21b @1600 | 19 | 17 | **24** |

Both shed the same White wins. Gen9 turns them into **Black wins**; v21b turns
them into **draws**. The gap number hides this completely.

A caution recorded because it was got wrong first: self-play Black faces its own
White, so a wider self-play gap does **not** imply a weaker Black. Against a
common opponent Gen9's Black is clearly stronger — 0.5000 against v21b's White
versus v21b's 0.3000 against Gen9's.

## 30. The draws are unconverted wins, not fortresses (2026-08-15)

Owner, watching the games: *"all of the draws involve shuffling so anything
that reaches a shuffling state will continue to."* Correct, and it killed a
planned experiment — raising `MAX_GAME_TURNS` would have produced "still drawn"
for the wrong reason, because both engines were in a repetition loop.

Measured instead, on 24 capped games from full-game play at 1600 sims:

- **Every draw is a cap draw.** All 24 ended at exactly 225 plies, min and max
  identical. Decisive games average 57.7 plies and top out at 147: games either
  resolve by ~ply 147 or run the whole clock, with nothing in between.
- **Black is winning all of them.** Ahead on material in 24 of 24, mean edge
  **+26.4**, with White on a **bare king in 21**.
- **Both sides are shuffling.** The last 100 plies contain a mean of **12.6
  distinct positions**.

### 30.1 The exact solver: bounded horizon, not a fortress classifier

`src/forced_capture.py` answers "is this won?" exactly, and neutralises the turn
counter so it asks about the position rather than the clock. Run on the last 3
Black-to-move positions of each capped game (72 positions):

| | forced wins | proven NOT won | budget exhausted |
|---|---:|---:|---:|
| within 3 Black moves | 9 | 63 | 0 |
| within 4 Black moves | **15** | 57 | **0** |

**Zero exhaustion at either depth** — every search ran to completion, so each
negative proves only that there is no forced capture *within that stated
horizon*. It does not prove a draw, fortress, or unchangeable state. Grouped by
game, **6 of 24 capped games contained a forced king capture within four Black
moves that the engine walked past at 1600 sims.**

Depth 5 on the 57 remaining ended without completing and without an error, at
15 of 57: **2 more forced wins, 13 more proven not won, still zero
exhaustion**. Those 15 are the fastest to finish, so they favour positions with
small search trees, and the cause of the termination is unattributed. The
ladder therefore reads 9 forced wins at depth <=3, 15 at <=4, at least 17 at
<=5.

**Depth 4 is the operating point for a finisher.** It completed all 72
positions in 9 minutes inside a 6M-node budget; depth 5 cost roughly 23
CPU-minutes per position and would have needed about 2.8 hours for the 57.
The marginal wins do not justify two orders of magnitude more search inside
real play.

### 30.2 Certainty propagation is a null

`MONSTER_SOLVER=1` (in-tree certainty propagation, implemented and defaulted
off) on the same three matchups and seed:

```
                     solver     baseline
Gen9 vs Gen9         6- 1-17    5- 1-18
Gen7 (W) vs Gen9     5- 4- 2    5- 3- 2
v21b (W) vs Gen9     5- 2- 6    5- 2- 6
```

Identical inside noise. A four-move forced line is far beyond what 1600
simulations will prove through this branching factor.

### 30.3 The finisher, and what it is worth

`benchmark._FinisherEngine` runs the exact search ahead of the network at
Black-to-move positions where White holds at most one non-king piece — the
phase the pathology lives in, and where White's branching is smallest. Opt-in
via `MONSTER_FINISHER`; nothing historical moves. Budget exhaustion falls
through to the network, pinned by test.

Against an **unbiased** 24-game self-play baseline (no early stop in either
arm):

| Gen9 self-play, 24 games | White | Black | draws |
|---|---:|---:|---:|
| baseline | 5 | 1 | 18 |
| finisher | 5 | **3** | 16 |

Black 0.417 -> **0.458**. Two of eighteen draws converted; White unchanged,
which is right since the finisher only fires for Black. That matches the solver
evidence rather than exceeding it: at `max_black_moves=3` it can only catch the
depth-<=3 subset, which was 9 positions across 3 games.

**An earlier read of this was biased and is corrected here.** The first
finisher run stopped after 8 games once its categories filled, showing 3-3-2;
those 8 are the first to *complete*, which skews short, and short games are
decisive because draws take the full 225 plies. The unbiased rerun is the
number above.

**Implication:** the depth-4 solve found wins in 6 of 24 games where the depth-3
finisher converts ~2. Raising the finisher to depth 4 should roughly triple the
effect, and depth 4 completed inside 6M nodes on every position, so it is
affordable.

## 31. Exhibition tooling, and what the opening books contain (2026-08-15)

`tools/matchup_examples.py` records complete games across a curated matchup set,
both colour assignments, keeping a balanced sample of each outcome. Three
lessons are baked into it:

- **Stop when full, don't play the block.** It originally played every game and
  selected afterwards, discarding 26 of 32 at 2-per-category. `--games` is now a
  ceiling; a matchup whose categories fill early stops.
- **Full games from the true start.** A handful of book entries reused across
  matchups makes every game a variation on the same few openings. With no book,
  play starts from the real position and the first 8 plies are sampled at
  temperature 1.0 purely to break determinism.
- **Duplicate rejection must be global and transposition-aware.** A per-matchup
  move-list key let four *different* White models' identical games through:
  White moves twice, so playing its half-moves in either order transposes. The
  key is now the sequence of settled positions, shared across matchups.

### 31.1 The p16 book starts games already decided

Owner: *"too many plies in the opening, some positions are won/lost outright."*
Measured across both books:

| | mean White | mean Black | White down a pawn | both armies intact |
|---|---:|---:|---:|---:|
| p16 (gate/screen book, n=800) | 3.69/5 | 14.25/16 | **89%** | **3%** |
| p8 (n=60) | 4.87/5 | 15.75/16 | 13% | 78% |

At 16 plies White has already lost **1.3 of its 4 pawns** on average, and only
3% of positions still have both armies whole. Given White's entire force is a
king and four pawns, that is a large and highly variable material swing before
either engine plays a move. Screens and gates run on this book: pairing controls
for it, but the measurement starts from lopsided middlegames rather than from
the opening. Changing it is a pinned-artifact decision with a re-anchor cost.

### 31.2 The shallow-book failure was temperature, not a reachability ceiling

An 8-ply book previously failed at 31 of 100 unique positions across 3 models,
which was read as a hard ceiling on reachable shallow positions. It was the
sampling temperature. At **temp 1.0** each of 5 models produced 12 unique
positions in exactly 12 attempts — **zero duplicates**, 60 entries in 34
seconds.

## 32. Moves-left reaches search, and is a clean null (2026-08-15)

The old moves-left experiment was auxiliary training only: inference discarded
the head, so it tested backbone regularisation rather than LC0's search use.
That gap is now implemented end to end. Python and native inference optionally
return a third buffer, tree backup carries remaining-decision estimates with
path distance added, and PUCT can apply a bounded utility: shorten likely wins,
lengthen likely losses. Defaults remain byte- and behavior-compatible; search
use is explicit and rejects checkpoints without the head. The match harness can
enable it independently for either player.

A frozen Gen9 lift trained only four new head tensors (8,321 parameters) on the
Gen10 replay. All 117 inherited tensors are bit-identical to the gated Gen9
nominee. The corpus has 1,047,422 rows, 699,546 trusted moves-left labels
(66.8%). Epoch 8 minimized validation Huber at 24.747; test Huber is 24.908.
On the 71,998 trusted test rows, MAE is 26.54 decisions versus 31.60 for the
constant median, Pearson correlation is 0.449, and mean bias is -9.54. The head
learns useful ordering but underestimates long games.

The game result is neutral. With the same lifted checkpoint on both sides and
only the default bounded utility toggled, 80 paired games at 400 sims scored
**0.5062** for utility-on (W 0.6750 / B 0.3375, paired SE 0.0062). On the more
important conversion probe — the last non-terminal Black state from each of 16
real capped Gen9 self-play draws, clock reset, 1600 sims — it changed **0/16**
selected moves and removed **0/4** reversals. A more aggressive threshold of
0.5 was also checked diagnostically and still changed 0/16, so there is no case
for parameter-mining this null.

Conclusion: retain the general, opt-in mechanism and the exact Gen9 lift, but
do not call it a successor or a conversion fix. The result supports the deeper
diagnosis: the visit leads behind shuffling are not small tie-break errors, and
a modest length utility cannot overcome the value/policy loop that created
them. Evidence: `match_gen9_mlh_on_vs_off_400_block700.json` and
`moves_left_gen9_conversion_probe_1600.json`. Training metadata is
`models/candidates/gen9_mlh_lift/train_run_20260815_214328.json`.

## 33. Deep-teacher recovery: four arms, one trade, no successor (2026-08-16)

### 33.1 The defect

Gen7--Gen10 each generated 4,000 retained one-row teachers at 3,200
simulations. The historical `generation_driver.py` processing call inherited
the four-ply non-human minimum, and a teacher is a deliberate **one-row**
policy-only file, so every one of them was dropped. Gen9's 102,906 rows and
Gen10's 111,098 rows are exactly twice their ordinary record counts, and
neither carries a policy-only value mask. The expensive reanalysis had been
completing and supplying zero training signal. Raw teacher files were intact,
so Gen7--Gen9 were reprocessed into non-overwriting `*_teacher3200_fixed`
increments, each passing an exact census: 4,000 teacher files, 8,000 mirrored
rows, 60/40 Black split, zero teacher value weight, source-linked splits.

### 33.2 Four arms, all rejected

Every arm trained fresh at seed 42 on the same recipe, and every bar is
byte-identical to V22 (SHA-256 `6a59b1f7…`; the 4x arm names it
`gen9_scratch/screen_nominee.pt`, the same weights).

| arm | teacher policy weight | teachers searched by | side split | screen ΔW / ΔB | both? | binding gate |
|---|---|---|---|---|---|---|
| 4x | 4 | historical | 60/40 | +0.050 / +0.050 | yes | e6 FAIL (confirm B 0.3850); e7 FAIL (first leg B 0.3850) |
| 1x | 1 | historical | 60/40 | +0.005 / +0.020 | yes | FAIL (B 0.3850, agg 0.4925) |
| 2x | 2 | historical | 60/40 | −0.080 / +0.125 | no | not gated |
| balanced 4x | 4 | **V22** | **50/50** | −0.030 / +0.105 | no | FAIL (B 0.3800, agg 0.4825) |

The balanced arm re-ran reanalysis with V22 itself at `--black-fraction 0.5`
(4,000 kept teachers, 2,000 per side, action-change rate 0.94) to test whether
the trade was an artifact of stale teachers or the 60/40 Black reservation. It
was neither: the same signature appeared.

### 33.3 The arms agree on a White-for-Black trade

Across weights 1x/2x/4x, both side splits, and both teacher-search models, the
recovered teachers move Black up and White down. On the balanced arm's screen
block all four finalists reproduce it (ΔB +0.070 to +0.105, ΔW −0.030 to
−0.105), and the ordering is monotone: the checkpoint with the largest Black
gain carries the largest White loss. No arm produced a checkpoint that was
positive on both colours at 200 games.

### 33.4 The effect being chased is smaller than the measurement noise

This is the campaign's most useful result. Eight 200-game self-calibrations of
**V22 against itself** on disjoint blocks of the two p8 books:

| block | 0 (v22) | 420 (v22) | 240 | 0 (gen9) | 660 | 420 (gen9) | 540 | 120 |
|---|---|---|---|---|---|---|---|---|
| V22 self-Black | 0.280 | 0.300 | 0.305 | 0.360 | 0.340 | 0.370 | 0.390 | 0.395 |

Mean 0.3425, spread **0.115**, sd **0.043** — consistent with ordinary
sampling noise at 100 games per colour leg (SE ≈ 0.045), not with block
difficulty. A calibrated colour delta subtracts two such measurements, so its
SE is ≈ **0.064**. Every effect in §33.2 is inside 2 SE of zero.

The balanced arm demonstrated this directly and expensively. Its epoch-5
nominee screened at ΔB **+0.105** on block 420 and was projected to reach
raw Black 0.495 on gate block 540. It scored **0.380** — against that block's
own V22 self-calibration of 0.390, a calibrated **−0.010**. The candidate's
raw Black barely moved (0.4050 → 0.380); what moved was the *calibration*
(0.300 → 0.390). The apparent gain was V22 playing unusually poorly as Black
on block 420, harvested by selecting the luckiest of eight checkpoints.
White, whose block-to-block variance is the same but whose effect was near
zero, predicted almost exactly: 0.580 projected, 0.585 measured.

**Consequence for method.** A 200-game screen against a 200-game calibration
cannot resolve a 0.05 colour effect; it will keep nominating the checkpoint
with the luckiest block and that nomination will keep regressing at the gate.
Detecting +0.05 at 2 SE needs roughly 400 games per colour for both candidate
and calibration — about 4x the current screen cost. This also tempers the 4x
arm's reading: its +0.095 and +0.080 calibrated Black legs are ~1.5 SE each,
and only mildly stronger (~2.1 SE) taken together.

### 33.5 The floor is measuring the same noise

Three arms failed a Black leg at exactly 0.3850 with different W/D/L
decompositions (22/45/33, 18/41/41, 19/42/39), so this is coincidence, not a
replayed match. But it sits against an absolute 0.40 floor while V22 itself
scores 0.280--0.395 as Black on these blocks: **an exact copy of V22 submitted
as a candidate would fail its own floor on five of the eight blocks measured.**
The threshold was not moved and is not proposed to move; this is recorded so
raw colour scores are read together with exact-block self-calibration, as
§2 already requires.

The conversion diagnosis is unchanged and visible in every leg. In the
balanced arm's binding Black leg, all **36 of 100** draws were reached with
Black ahead on material at the cap. Converting them would score 0.740 instead
of 0.380. No teacher weighting touched that.

**Conclusion.** The ingestion defect was real and worth fixing — the recovered
increments are correct and registered. But recovering the teachers does not
produce a successor at any weight or side split, and the loop cannot currently
distinguish a 0.05 colour effect from block noise. Evidence:
`screen_gen9_teacher_recovery{1x,2x,4x}_seed42_p8_20260816.json`,
`screen_gen9_v22balanced_teacher4x_seed42_p8_20260816.json`,
`gate_gen9_teacher_recovery{1x,4x}_seed42*_p8_20260816.json`,
`gate_gen9_v22balanced_teacher4x_seed42_p8_20260816.json`, and the four
`match_v22_self_p8_offset*_20260816.json` calibrations.

## 34. The finisher reaches generation, and converts nothing at 700 sims (2026-08-16)

Law 1a says the model plays won positions as lost because the **corpus** says
they are. The obvious attack is to run the exact forced-capture search during
self-play, so a won-but-shuffled ending finishes with a real king capture and
enters the corpus as a true Black win instead of a -0.5 move-limit relabel.
That is now implemented, opt-in via `MONSTER_FINISHER`, running only where the
scripted oracle abstains so its verified class is untouched.

**It does not work at the production generation operating point, and the reason
is not the mechanism.**

### 34.1 The paired pilot

24 games, V22, 700 simulations, seed 4242, no early stop (section 30.3's
sampling lesson), finisher at depth 4 / 2M nodes.

| | true Black win | cap draw | White win | mean records | wall |
|---|---:|---:|---:|---:|---:|
| finisher off | 5 | 8 | 11 | 93.3 | **4m13s** |
| finisher on | 5 | 7 | 11 | 84.3 | **44m30s** |

Zero finisher moves fired with the flag unset, which verifies the opt-in
guarantee in a real run rather than only in unit tests.

The finisher fired **9 times across 3 games** — and all three (games 4, 7, 14)
were **already** Black wins in the baseline. It ended them sooner; it converted
nothing. The `on` arm saved 23 games, not 24: one capped game exceeded its
per-game wall-clock budget and was correctly discarded as aborted. **The
mechanism cost 10.5x runtime and one game of data, for zero conversions.**

### 34.2 Why: there was nothing to convert, and that is proven

The natural suspicion is the gate — that White still held pawns, so the search
never ran. Measured, and it is false. All 8 capped baseline games end with
White on a **bare king**, and each contains **46-55 Black-to-move positions
inside the finisher's gate**. The search was consulted hundreds of times per
game and returned nothing.

The second suspicion is a starved budget. Also false. Reproducing section
30.1's protocol — the last 3 Black-to-move positions of each capped game, 24
positions — at both the pilot's budget and section 30.1's:

| budget | forced wins | proven NOT won | exhausted | cost |
|---|---:|---:|---:|---:|
| 2,000,000 | 0 | 24 | **0** | 33.3 s/pos |
| 6,000,000 | 0 | 24 | **0** | 32.4 s/pos |

**Zero exhaustion at both budgets.** Every search ran to completion, so these
are proofs that no forced capture exists within four Black moves — not
truncations. The pilot's null is real.

### 34.3 The conversion opportunity depends on search depth

This refines section 30 rather than contradicting it. That measurement found
forced wins within four Black moves in 6 of 24 capped games — but those games
were played at **1600** simulations. This pilot ran at **700**, the production
generation setting. At 700 sims Black reaches capped positions that are further
from won, and the depth-4 horizon no longer reaches a capture.

So the two results together say something sharper than either alone: the
unconverted wins section 30 found are a property of *deeply searched* play, and
generation does not produce them. A finisher cannot rescue a position that the
generator never reaches.

**Do not enable `MONSTER_FINISHER` for generation at 700 sims.** It is a 10.5x
tax with a measured zero return and a demonstrated data-loss path. The
mechanism is retained, off by default, because the section 30 evidence says the
opportunity exists at higher search depth; the open question is whether
generating at 1600 sims yields enough convertible endings to pay for itself,
which is a much more expensive experiment than this one.

Two loose ends, recorded rather than resolved. The exact search cost **33
s/position** here against section 30.1's reported 7.5 s/position at the same
depth and budget, a 4.4x discrepancy that is unexplained and may simply be
harder positions. And `_finisher_probe` now returns the exhaustion flag
separately: the first implementation collapsed "budget exhausted" and "proven
not won" into a bare `None`, which is right for *play* (both fall through to
the network) but destroys the very distinction 34.2 turns on.

## 35. The exact solver moves to Rust: 145x, and the finisher becomes free (2026-08-16)

Section 34 rejected the generation finisher for a **10.5x** cost with a
demonstrated data-loss path, and section 30.1 priced depth 5 at roughly 23
CPU-minutes per position. Both numbers were measuring one thing: an exhaustive
AND/OR search written in Python.

### 35.1 Where the time actually went

Profiling the solver on a real capped-game position: **92% of cumulative time
in `_get_white_actions`**, generating White's half-move pairs through
python-chess, with 9,925 `push` calls for a 296-node search.

That explains a null. Adding memoisation, resulting-position dedup at AND
nodes, and stackless clones to the Python solver gave **1.0x at depth 3 and
1.8x at depth 4** — exact (zero disagreements over 56 positions), but unable to
touch a bottleneck that sits below it. The dedup in particular almost never
fired (0.001 skips/node): AND nodes short-circuit on the first refutation and
never enumerate far enough to meet a duplicate.

### 35.2 The port

`native/src/solver.rs` reimplements the search on the bitboard engine, keeping
the semantics that make it a *proof*: king capture is unconditional and
pseudo-legal, the turn cap is neutralised, no-pseudo-legal-White-move is not a
forced capture, and budget exhaustion is reported rather than swallowed.

Because it builds White's successor set before recursing, the transposition
dedup measured at 2.74x now actually fires: **7.7M skips against 2.25M nodes**.

Measured over the 24 last-Black-move positions of the capped games, depth 4,
6M-node budget:

| | time | per position |
|---|---:|---:|
| Python (already optimised) | 501.2s | 20.88s |
| Rust | **3.5s** | **0.144s** |
| | | **145x** |

**Zero disagreements.** The decision, the depth-to-win, and the exhausted flag
match on every position.

### 35.3 The new depth ceiling

| depth | forced wins | proven NOT won | exhausted | cost/position |
|---|---:|---:|---:|---:|
| 3 | 0 | 24 | 0 | 0.01s |
| 4 | 0 | 24 | 0 | 0.14s |
| 5 | 0 | 24 | 0 | **1.78s** |
| 6 | 0 | 21 | 3 | 15.8s |

Depth 5 cost ~23 CPU-minutes per position in section 30.1 and now costs 1.78s —
about **775x** — and depth 6 is reachable for the first time. This also
independently strengthens section 34: those capped positions are proven not won
within **six** Black moves, not merely four.

### 35.4 The finisher is now free

Re-running section 34's exact pilot (24 games, V22, 700 sims, seed 4242) with
the native solver behind the same `MONSTER_FINISHER` flag:

| run | games | Black wins | cap draws | finisher moves | wall |
|---|---:|---:|---:|---:|---:|
| no finisher | 24 | 5 | 8 | 0 | 4m13s |
| finisher, Python solver | **23** | 5 | 7 | 9 / 3 games | **44m30s** |
| finisher, Rust solver | **24** | 5 | 8 | 9 / 3 games | **2m22s** |

**18.8x end to end.** It makes the identical nine decisions in the same three
games, and the game that section 34 lost to its per-game deadline comes back.
The cost objection is retired; the *conversion* result is unchanged, because
at 700 simulations there is still nothing to convert.

`try_forced_capture_move` now uses the native path automatically, so the
finisher and any other caller inherit it. `MONSTER_SOLVER_PYTHON=1` forces the
reference implementation, and a test asserts the two agree — this component
returns proofs, so a fast path that quietly disagreed would corrupt every
conclusion drawn from it.

### 35.5 Depth 6 partially revises the section 34 null

The depth-6 row above had 3 positions exhaust at a 40M budget. Re-run
single-threaded at 400M, they resolve with **zero exhaustion**:

| depth | forced wins | proven NOT won | exhausted |
|---|---:|---:|---:|
| 3–5 | 0 | 24 | 0 |
| **6** | **2** | 22 | 0 |

Both wins sit in the **same game** — 1 of the 8 capped games contains a forced
king capture within six Black moves. So section 34's "nothing to convert at 700
simulations" is right at depth 4 and wrong at depth 6, but the opportunity is
much thinner than deep play offers: 1 of 8 games here against section 30's 6 of
24 at 1600 simulations. Conversion opportunity scales with the *generator's*
search depth, not just the finisher's.

At 15.2s per position a depth-6 finisher cannot run at every gated Black move —
there are 46–55 of them per capped game. It would have to fire selectively,
near the cap, which is where the unconverted win actually matters.

### 35.6 Threaded root search: correct, but it scales poorly

Root Black moves shard cleanly, so `threads` is exposed and defaults to 1
(bit-identical to the sequential path). Depth 6 over 24 positions:

| threads | time | speedup |
|---|---:|---:|
| 1 | 382.4s | 1.00× |
| 4 | 303.0s | 1.26× |
| 8 | 220.6s | 1.73× |
| 16 | 183.7s | **2.08×** |

2.08× on 16 threads is poor, and the cause is structural: each worker keeps its
own memo, so parallelism is bought by discarding the transposition sharing that
makes the solver fast in the first place. A shared table would need a lock on
the hottest path. Threading is therefore a weak lever here and is left opt-in.

The apparent "mismatches" at threads > 1 were investigated rather than assumed:
all 3 were positions where the single-threaded run exhausted its budget and the
threaded run — carrying one budget per worker — completed. **Zero proof
disagreements.** A completed proof is never contradicted; only "no answer"
becomes an answer.

## 36. A repetition draw, and why fourfold beats threefold (2026-08-16)

`is_terminal` fires only on king absence or the 150-turn cap, so shuffling
endings run the full clock. `src/repetition.py` adds an opt-in repetition draw
at the **driver** level — not in `is_terminal`, because repetition is
path-dependent and putting it in the tree reintroduces the graph-history
problem and fights any transposition table.

### 36.1 The threshold sweep

24 games, V22, 700 sims, seed 4242. Arms run **sequentially**: launching three
concurrently put 24 CUDA workers on 16 cores and froze the machine.

| threshold | Black wins | cap draws | rep draws | White wins | score | mean records | wall | **wins destroyed** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off (baseline) | 5 | 8 | 0 | 11 | 0.3750 | 93.3 | 4m13s | — |
| 3 (chess default) | 4 | 3 | 6 | 11 | 0.3542 | 77.0 | **2m17s** | **1** |
| **4** | 5 | 5 | 3 | 11 | **0.3750** | 84.2 | **2m59s** | **0** |
| 5 | 5 | 5 | 3 | 11 | 0.3750 | 84.8 | — | 0 |
| 6 | 5 | 5 | 3 | 11 | 0.3750 | 86.6 | — | 0 |

**Threefold truncates real conversions.** `game_00004` was a king capture at
ply 130 and became a draw at ply 84. The mechanism is specific to this game:
Black converts by *maneuvering*, repeating a position while improving, and
**White can pass** — `Ke1e2` then `Ke2e1` returns the identical position, a
free tempo Black does not have. Chess's threefold convention assumes neither
side has a null move. Here White does, so the third occurrence is not yet
evidence of shuffling; the fourth is.

Four, five and six are identical on outcomes and differ only in records
retained, so the entire risk sits in the single step from 4 to 3.

**The owner set the default to 3** (2026-08-16), taking the 1.85×. The one game
that costs was then inspected rather than assumed, and it is not a conversion
cut short: in `game_00004` Black held **eleven pieces against a bare king from
record 40** and needed until record 129 to capture, repeating positions on the
way. That is floundering, not maneuvering — maneuvering by definition reaches
new positions — so ending it as a draw is a fair verdict on the play. An
earlier version of this section justified 4 by claiming Black repeats while
maneuvering; that reasoning was wrong and is withdrawn.

The scripted oracle was dropped in the same change, which matters here: its
explicit anti-repetition drift kept oracle-driven games at 94–98 distinct
positions per 100 records, blunting this rule. Without it, repetition should
fire more often than the sweep above measured.

### 36.2 What it does and does not change

An earlier reading of this was wrong and is corrected here. Repetition does
**not** collapse Black's match score: a capped ending already scored as a draw
under the owner's captures-only rule, so match scoring moves only by the games
whose *decisive* outcome changes — 0.3750 → 0.3542 at threefold, and
0.3750 → 0.3750 at fourfold. The ±0.5 is a **training label**, not a match
score. What the rule genuinely changes is that label: capped games move from
−0.5 ("leaning Black") to 0.0, removing partial value credit the current proxy
gives Black.

It is off by default (`MONSTER_REPETITION`), because it remains a rules change
and would make earlier numbers incomparable the way the 2026-08-03
captures-only correction did.

## 37. The exact solver at MCTS leaves is a null (2026-08-16)

With the solver 145x faster (section 35), a shallow exact probe at leaves
became affordable in principle: inject proven wins where the network has only
a guess, and let the existing proof machinery propagate them. Implemented in
`run_batched_puct` behind `solver_probe_depth` (0 = off).

Measured on a position the solver proves won in **6 Black moves** — 18 plies,
against an endgame PV pinned at 4:

| configuration | move played | is the solver's winning move | probes | hits | time |
|---|---|---|---:|---:|---:|
| solver at the root | **h4h3** | ground truth | — | — | 65.6s |
| MCTS, no probe | c6b6 | no | 0 | 0 | **0.06s** |
| probe depth 3, limit 200 | c6b6 | no | 200 | 9 | 2.22s |
| probe depth 4, limit 100 | c6b6 | no | 100 | 5 | 28.89s |
| probe depth 4, limit 400 | c6b6 | no | 400 | **46** | **112.94s** |

**Forty-six proven leaf wins changed nothing**, at roughly 1,900x the cost.

The reason is structural and restates section 30.2's certainty-propagation
null. A proof is only useful if it survives an **AND** node — every White reply
proven losing — and proving that layer *is* the whole problem. 1,600
simulations cannot cover a 390-wide AND layer, so proofs accumulate at leaves
and die one level up. Scattering probes pays the solver's cost repeatedly and
buys none of its power.

An earlier version of this measurement asked whether the ROOT becomes proven.
That was the wrong question and was abandoned: reconstructing a root proof from
leaf probes costs hours, while calling the solver at the root answers it
outright. The corrected question — does the probe change what the search
*plays* — is the one above.

**Conclusion: the exact solver belongs at the root, not in the tree.** That is
where the finisher already calls it, at 0.14s for depth 4. The mechanism is
retained at a depth-0 default so the negative stays reproducible, exactly as
the moves-left head was.

## 38. The inference server loses, and the free speedup was already built (2026-08-16)

`src/inference_server.py` (D3 stage 2) has been implemented, benchmarked and
smoke-tested for some time, wired into nothing, with **no recorded result**. Its
premise is sound on its face: per-position forward cost is 0.294 ms at batch 16
against 0.037 ms at 256, so eight workers submitting 16 leaves each ought to
become one 128-leaf forward. Its own bench states the decision rule — *"a
stage-2 number that is not clearly better at N=8 means the server is not worth
its complexity and should not be wired into generation."*

Run at N=8, V22, 700 sims:

| scale | stage 1 (8 models) | stage 2 (one server) | search ratio |
|---|---|---|---|
| 4 games | 2.22s search / 6.5s wall | 3.02s / **3.5s** | 1.36x worse |
| 24 games | 13.34s search / **17.2s wall** | 17.40s / 17.9s | 1.30x worse |

**The server loses.** Its apparent win at 4 games is entirely model-load
amortisation — one load instead of eight — and by 24 games it is already behind
on wall clock. Extrapolated to a 500-game generation the 1.30x search penalty
costs about 85 seconds outright. Eight concurrent workers already keep the GPU
busy enough that the IPC round trip plus the 2 ms linger costs more than
cross-worker batching returns.

Its one real benefit is holding one model in VRAM instead of eight, which binds
only if more workers than memory allows were wanted; 8 is already the measured
throughput optimum. **Do not wire it into generation.** The negative is
recorded here so the question stops being open.

### 38.1 The genuinely free speedup is tree reuse, and it is switched off

Measured 2026-08-04 and never enabled: **strength at equal sims 0.5000 over 200
games — a clean null — for −23% wall clock.** That is the definition of a
quality-neutral speedup, it is already implemented in the native engine with
derived Q-rebasing, and it has sat behind `MONSTER_REUSE=1` ever since. Nothing
needs building.

### 38.2 Reanalysis discards half its deep searches, and pre-filtering is a trade

`--sample 8000 --keep 4000` searches 8,000 positions at 3,200 simulations,
computes priority **from** that search, then discards the lower half. Those
4,000 searches are the largest single piece of pure waste in the pipeline:
roughly 7 minutes of a 14.5-minute stage, every generation.

Priority compares the deep policy/value against the *recorded* ones, so it
cannot be known without searching. But a cheap pass could rank first. Measured
on 300 real positions against a 3,200-simulation reference:

| cheap sims | two-stage cost | index overlap @50% | spearman | **retained priority mass** |
|---|---:|---:|---:|---:|
| 200 | 0.57x | 80% | 0.734 | 87.9% |
| **400** | **0.64x** | 85% | 0.811 | **91.2%** |
| 800 | 0.77x | 85% | 0.852 | 91.9% |

800 is dominated — the same overlap as 400 for more cost. Index overlap
flatters the method, because swaps happen at the 50% cutoff where priorities
are near-identical by construction, so the honest metric is the deep-priority
mass the cheap selection retains: **91.2% at 400 sims**, capturing 78.2% of the
distance between a random half and the true best half.

**So this is a 36% saving for a ~9% degradation in teacher selection, not a
free speedup.** Whether 9% matters is unresolved and may not be worth resolving:
section 33 found the deep-teacher program itself a measured null across 1x, 2x,
4x and balanced arms, so the stage being optimised has no demonstrated value.

### 38.3 What is already exploited

Checked and found correct, so no gain remains: fp16 inference; `pre_nn_clamp`
skipping the forward on decided positions; the eight-worker default (measured
optimum); the native engine; and early stopping, which is **on** for matches
and deliberately **off** for generation because it truncates the visit
distribution that *is* the policy target.

**The one genuinely quality-neutral speedup available is tree reuse** (§38.1):
strength-null at equal simulations, −23% wall clock, implemented, and switched
off since 2026-08-04. Everything else measured today is either a loss (the
inference server, the leaf probe) or a trade (pre-filtering, repetition,
dropping the oracle). The remaining untested free candidate is CUDA graphs or
`torch.compile` on a 1.9M-parameter network at batch 16, where kernel launch
overhead plausibly dominates and the arithmetic would be unchanged.

*Updated 2026-08-16. Suite 711 passing. Predecessor reports
retire to git history per project convention.*
