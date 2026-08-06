# MASTER REPORT — 2026-08-04: the engine rewrite lands, and what it measured

Replaces the 2026-08-03 run report (git history holds it). Covers the owner's
rewrite directive (`DIRECTIVE.md`, 2026-08-03) from its writing through today:
**five of six phases closed inside two days** against the directive's own
1.5–2-week optimistic bound, the first LC0-derived search changes, the data
intake, and the answer to the question the whole campaign was premised on.

Every claim cites its artifact in `benchmarks/` (`benchmarks/INDEX.md` maps the
active evidence set). Every bug below is pinned by a regression test. **Suite:
522 passing plus 3 subtests after the 2026-08-05 successor work.**
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
   Generation uses it; the native search initially couldn't express it.
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
- **`benchmarks/INDEX.md`** — 143 artifacts, 21 families, newest marked,
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
`v21_*capture_wdl*20260805.json` artifacts. Final verification: **532 passed,
10 warnings, 3 subtests passed**. The best Black-leaning lead is
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

`generate -> reanalyze -> process -> compose -> train -> offline_gate ->
binding_gate -> self_skew -> promote`

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
1-point hard threshold rejected it without a game. That result is retained as
an offline warning and the candidate is queued for the binding game protocol.

The production loop now also honors the repository's measured eight-worker
default instead of silently applying the obsolete four-worker NN cap. On this
5060 Ti, prior 400-simulation measurement was 5.39 decisions/s at four workers,
7.11 at eight, and 7.38 at twelve; fourteen exhausted memory. Eight therefore
uses more of the available GPU while keeping useful failure headroom.

A live native smoke run completed generate -> reanalyze -> process: two games
(one win per color), 200 self-play positions, two positions deep-searched, one
teacher retained, 201 raw rows / 402 augmented rows, and no illegal targets.

---

## 16. Next

1. Run generation one with architecture fixed, inspect the offline report, and
   let the full binding gate decide whether the bootstrap champion advances.
2. If it passes, measure self-color skew and start generation two with promoted
   replay. A bootstrap promotion is not automatically V21.
3. After the pipeline has a clean control result, test moves-left as one isolated
   model change. Do not bundle it with data or search changes.
4. Track learning curves, replay composition, policy divergence and per-side
   gates across generations. Use failure positions for general reanalysis, not
   hand-authored tactical rules.

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
loading. The complete suite passes 551 tests plus 3 subtests.

Open owner decisions: §7.4's hand-corrected label; the `combined_v16` copy;
the 23 legacy unreplayable games. The bar is no longer open.

*Updated 2026-08-06. Predecessor reports retire to git history per project
convention.*
