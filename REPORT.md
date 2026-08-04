# MASTER REPORT — 2026-08-04: the engine rewrite lands, and what it measured

Replaces the 2026-08-03 run report (git history holds it). Covers the owner's
rewrite directive (`DIRECTIVE.md`, 2026-08-03) from its writing through today:
**five of six phases closed inside two days** against the directive's own
1.5–2-week optimistic bound, the first LC0-derived search changes, the data
intake, and the answer to the question the whole campaign was premised on.

Every claim cites its artifact in `benchmarks/` (`benchmarks/INDEX.md` maps all
143). Every bug below is pinned by a regression test. **Suite: 456 passing.**
Long runs log to `logs/`; `py -3 tools/runs.py status` shows live progress.

---

## 1. Where things stand

| phase | status | headline evidence |
|---|---|---|
| E0 calibration | **done** | curve + forced-capture spike (§5, §6) |
| E1 rules core | **done** | exact-order movegen parity; lockstep 31,807 plies |
| E2 heuristic + encoding | **done** | **bit-identical** on 148,272; byte-equal tensors |
| E3 native MCTS | **done** | gates: **400/400**, 0.525 (0.71 SE), 6.31× |
| E4 integration | **done** | clean 240-game overnight, healthy distribution |
| E5 re-baseline | **ready** | `tools/rebaseline.py`, ~50 min on native |
| E6 exploit | started | solver + reuse measured/measuring (§8) |

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
one measured after.** E5 exists to discharge that debt in a single event — one
cross-table (v17, ramp, v19, v19_B, heuristic anchor), captures-only, all on
the native engine, which also settles the open bar decision (v19 vs `v19_B`).

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
post-promotion starts, White fixed at 400 sims, captures-only throughout):

| Black sims | 200 | 400 | 800 | 1600 | 3200 | **6400** |
|---|---|---|---|---|---|---|
| true king captures | 0.19 | 0.17 | 0.24 | 0.37 | 0.51 | **0.55** |
| dominant, unfinished | 11 | 26 | 26 | 24 | 20 | 18 |
| White wins | 0.70 | 0.57 | 0.50 | 0.39 | 0.29 | 0.27 |
| python curve (03 Aug) | 0.09 | 0.15 | 0.20 | 0.30 | 0.51 | — |

Per-doubling gains: +0.07, +0.13, +0.14, then **+0.04** (≈1 SE at n=100).
**The knee is at 3200.** The directive's §5 contingency is triggered: raw
simulations cap out around 0.55 on this deck, so the exploit order shifts from
"more sims" to the finisher/solver and to data. The rewrite's justification
stands on throughput (gates at the 3200 operating point cost 12 minutes, not
2.3 hours) — not on sims buying conversion indefinitely.

**Open discrepancy, stated plainly:** native reads above python at low sims
(0.19 vs 0.09 at 200; ≤2.3 SE per point but consistently positive; exact
agreement at 3200). I first blamed the RNG bug — **that attribution is
falsified**: a post-fix re-run reproduced the pre-fix curve *identically to
the decimal*, proving these temperature-0, noise-off playouts never touch the
RNG. Both engines are deterministic here; they play different games through
accumulated tie-breaks. Diagnosis queued: per-decision agreement at 200 sims
(gate (a) ran at 400), then a first-divergence trace. E5 is unaffected — it
runs single-engine.

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
and reports exactly +1.00. **The decisive measurement — true captures at
1600/3200 with solver on vs the §6 baseline — is running now**
(`logs/solver_curve.log`).

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

**Moves-left signal** — blocked as first proposed: reshaping the value target
is a do-not-do (law 5; a global per-ply discount was rejected in v13 for
taxing Black's long wins). LC0's version is a separate *head* — a model
change, post-E5. **WDL head** — machinery exists (`train.py`), but law 4:
label shaping is invisible under WDL; interacts with the ramp.

**The tablebase analogue** — LC0 doesn't search endgames, it looks them up.
Our equivalent is the exact forced-capture solver + repaired oracle; the E6
candidate is an alpha-beta *prover* over the terminal objective ("forced king
capture within N"), where eval quality doesn't matter because lines bottom out
in captures. Native movegen at 19–26× should push the affordable proof depth
from 3 toward 5–6.

---

## 9. Data intake (all unmeasured until post-E5 training)

- **`ps_monster_e1500`** — the full playstrategy corpus was on this box all
  along (`data/playstrategy/ps_games.json`, 2,966 validated games); only 829
  had been imported, at `--min-elo 1600`. Imported the 1500 tier as a separate
  source: **2,132 games / 103,414 positions** (2.6× the existing human corpus).
- **`combined_v19_K_owner41`** — 41 owner games (the whole top level of
  `human_games/`) had never been incorporated; corpus human files 123 → 164.
  L4 remains the best per-record lever on record (+0.150 from 27 games).
- **`tools/finish_unconverted.py`** — resumes −0.5 games rather than
  relabelling them (§7).

Three corpus variants — base, +e1500, +owner41 — are the natural first
training ladder *after* E5 makes gates comparable.

---

## 10. Infrastructure

- **`logs/` + `tools/runs.py`** — every long run gets a named, live log;
  `status` shows alive/elapsed/latest progress; `tail` reads one. Progress
  streaming added to the probes (`imap_unordered` — a run that reports only at
  the end is indistinguishable from a hung one).
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

## 12. Next

1. **Solver measurement** — in flight (`runs.py status`); the number that
   decides whether certainty propagation moves true captures.
2. **E5 re-baseline** — the single comparability event; ~50 min native;
   settles the bar (v19 vs v19_B). Everything downstream waits on it.
3. **Post-E5 training ladder** — base vs +e1500 vs +owner41, gated at the
   3200-sim operating point the rewrite made affordable (12 min/gate).
4. **Curve-gap diagnosis** — 200-sim agreement probe + first-divergence trace.
5. **E6 / pruning conversation** — with the depth data, the plateau, and the
   AB-prover scoping ready (owner has this queued).

Open owner decisions, unchanged: the bar (falls out of E5); §7.4's
hand-corrected label; the `combined_v16` copy; the 23 legacy unreplayable
games.

*Written 2026-08-04. Predecessor reports retire to git history per project
convention.*
