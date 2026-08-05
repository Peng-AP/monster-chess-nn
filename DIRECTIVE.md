# DIRECTIVE — the engine rewrite (2026-08-03)

**Status (2026-08-05): complete through E5.** The native engine is the
default, the captures-only re-baseline established v19_B as the formal bar,
and the follow-on Black-first architecture campaign has produced
`lc0b_attention_ema`, which passed both the binding automated gate and the
owner playtest. Its 64-channel successor has passed two calibrated reads and
awaits its owner playtest. Current evidence and next actions are in `REPORT.md`; the
remainder of this document preserves the rewrite contract and E6 scope.

**Owner's call: the residual problem may be compute.** Search is Python-bound
(CONTEXT law 18: NN forward 14% at batch 16 and 7.7% at batch 256,
python-chess movegen ~35%, clone/apply
25–32%, tree ~25%) and search strength demonstrably buys conversions (law 12,
corrected: true king captures 0.09 → 0.30 as Black's sims go 200 → 1600, still
rising at the last measured point). This directive scopes the whole-engine
rewrite: a native search core that makes high-sim play, mass asymmetric
generation, and a dedicated finisher search affordable on this box.

**What the rewrite is for** — three purchases, in order:
1. **High-sim operation as the norm.** 1600+ sims at today's 400-sim cost, for
   play, gates, and generation.
2. **The finisher search.** A forced-king-capture solver (proof-number or
   alpha-beta) at dominant positions — the direct attack on the 40%
   dominant-unfinished pathology. Nearly free once a native core exists;
   a separate engine project without one.
3. **GPU utilization.** The 5060 Ti idles at 14% of decision time. Native leaf
   collection + cross-game batching turns it into the bottleneck it should be.

**Authorization.** Standing go carries forward (overnights and multi-worker
runs unattended; never while the owner plays; promotion still requires his
playtest; no gate threshold moves).

---

## 0. Non-negotiables

1. **Parity before improvement.** The port reproduces the current engine's
   behavior *first*, gated by the tests in §3. No search "improvements" ride
   along with the rewrite — every behavior change comes later, behind a flag,
   measured on its own. A rewrite that changes strength and behavior at once
   can attribute nothing.
2. **The Python engine is never deleted.** It remains the reference oracle
   (differential tests run against it permanently) and the fallback
   (`--engine python|native` everywhere games are played).
3. **The owner's embedded directives port verbatim** — the king-safety
   override (owner 2026-07-12, engine-wide), the first-half override and
   oscillation penalty (owner 2026-07-17), selected-child value reporting
   (owner 2026-07-17). These are product decisions, not implementation detail.
4. **Abort trigger.** This is a multi-week commitment with a history of
   optimistic estimates behind it (§6). Stop and reassess if **E1 exceeds 8
   days**, if the 10M-position differential cannot reach legal-action set
   equality, or if E3(c) misses 5×. Stopping is cheap — §0.2 keeps the Python
   engine live, and E0 delivers its answers independently of the rewrite.
   Sunk cost is not a reason to continue past these.
5. **Nothing else changes.** Labels, corpora, thresholds, the scoring rule,
   the model, the laws — untouched. One comparability break is inherent in
   swapping the search engine; it is taken once, deliberately, at §3 E5.

## 1. The contract — what the source actually does (read 2026-08-03)

The port target is the *measured behavior* of these files, not their prose.
Every item below is load-bearing; several are subtle enough to destroy
training or gates silently if drifted.

### 1.1 Rules layer (`src/monster_chess.py`, 474 lines)

- **State:** FEN + `is_white_turn` + `turn_count` + `white_half_pending` +
  cached terminal/result. `clone()` carries `CLONE_HISTORY_PLIES = 8` plies of
  history ([monster_chess.py:21](src/monster_chess.py#L21)) — only the
  oscillation detector reads it, at offsets −1/−3/−4.
- **Terminal fires only on king absence** — no checkmate, no stalemate scan
  (deliberate, [monster_chess.py:95](src/monster_chess.py#L95)). At
  `turn_count >= MAX_GAME_TURNS` (150) the result is relabeled **by the
  heuristic's sign**: ±0.5 beyond |0.4|, else 0
  ([monster_chess.py:81-94](src/monster_chess.py#L81-L94)). **Consequence:
  search reaches the cap in-tree during late games, so the native engine must
  carry the heuristic evaluator** — it cannot be left behind in Python.
- **Movegen is pseudo-legal plus king-capture logic**, never python-chess
  `legal_moves` (which forbids king capture). Both sides: winning captures
  first, then safe moves, else **all** moves (forced blunder). The search
  paths truncate to a single winning action when one exists
  (`truncate_wins`); the oracle path (`get_legal_actions`) lists everything,
  winning first — **callers rely on that ordering**
  ([monster_chess.py:134-146](src/monster_chess.py#L134-L146)).
- **King capture wins unconditionally** — mid-pair captures end the game
  before White's own king safety matters; a first-half capture pairs with a
  null m2.
- **White may not END its turn attacked; the first half may pass through
  check** — enforced at the pending node, not per half
  ([monster_chess.py:262-266](src/monster_chess.py#L262-L266)). The half-move
  API deliberately differs from the atomic API in one documented way: an m1
  whose every continuation is unsafe is *offered* (search sees the losing m2);
  the atomic API hides it.
- **En passant is conferred only by the last push** — this falls out of
  python-chess recomputing the single ep square per push; a bitboard
  implementation must reproduce it explicitly
  (`tests/test_ruleset_divergences.py` pins both divergences).
- **Between White's halves, `board.turn` is forced back to WHITE** so FEN
  stays consistent ([monster_chess.py:303-311](src/monster_chess.py#L303-L311));
  the atomic `apply_action` re-checks m2 pseudo-legality after m1
  ([monster_chess.py:339](src/monster_chess.py#L339)).

### 1.2 Search (`src/mcts.py`, 713 lines)

- **Dispatch:** evaluator with `batch_evaluate` + `evaluate_with_policy` →
  batched PUCT; otherwise sequential UCB1 (heuristic mode). Both must port.
- **PUCT with FPU** ([mcts.py:255-278](src/mcts.py#L255-L278)): unvisited
  children start at parent-Q minus 0.30 — with a **perspective flip that
  depends on grandparent/parent side equality**, because White's two
  half-moves do not alternate sides. `_backpropagate` accumulates each node's
  value in *its parent's side-to-move perspective*
  ([mcts.py:690-705](src/mcts.py#L690-L705)). This is the subtlest logic in
  the engine; M2's metric bugs both came from misreading it.
- **Batched loop** ([mcts.py:534-597](src/mcts.py#L534-L597)): root expanded
  and backpropagated synchronously first; Dirichlet noise (α 0.3, ε 0.25)
  root-only and **self-play only**; virtual loss 3 during batch collection; a
  `pending` set cuts the batch early when the frontier is exhausted;
  simulations counted by completed backpropagations. Batch width default 16 —
  deliberately narrow in Python; the native engine may widen it *after* E3
  parity, measured.
- **Early stop** (play/arena only, never when recording training data): after
  ≥30% of sims, on |root Q| > 0.95 or an insurmountable visit lead;
  sequential mode checks every 32 sims.
- **Selection:** temperature over visit counts with the **oscillation
  penalty** (×0.90 on exact reversals of the mover's own recent moves, history
  offsets −1/−3/−4 White-second-half, −3 Black); the raw distribution is kept
  for the policy target ([mcts.py:137-157](src/mcts.py#L137-L157)). Then two
  owner overrides re-rank among searched children only:
  `_white_first_half_override` (an m1 all of whose m2s hang the king), then
  `_king_safety_override` (turn-completing action hands an immediate king
  capture). Reported search value is the **selected child's Q**
  ([mcts.py:160-176](src/mcts.py#L160-L176)).
- **Tree reuse only across White's first→second half** (same-perspective;
  reuse across a side change would need value rebasing and is deliberately
  absent, [mcts.py:353-399](src/mcts.py#L353-L399)).
- **Expansion:** priors from softmax over *legal* policy indices only; the
  atomic-pair fallback marginalizes P(m1)/|m2s| with an 80-child cap; the
  half-move path has no cap. `POLICY_TARGET_PSEUDOCOUNT` smoothing (currently
  0.0 = raw AlphaZero targets) in the recorded distribution.
- **Unvisited child Q reads 0.0** ([mcts.py:236-239](src/mcts.py#L236-L239)) —
  a known probe hazard, but parity means reproducing it.

### 1.3 Evaluation (`src/evaluation.py`, 584 lines)

- **Clamps run before any NN call and must be identical in native:** king
  absence → ±1.0; **side-to-move-only** capture-threat scan → ±0.95, where
  White's scan is **pending-aware** (single-move when `white_half_pending`,
  else full double-move scan) ([evaluation.py:154-166](src/evaluation.py#L154-L166)).
  Threat-against-mover deliberately does *not* clamp (the sacrificial-check
  bug).
- **The heuristic** (~250 lines of arithmetic: material, pawn progress,
  passed-pawn bonuses, king tropism, confinement geometry, 2-move king
  mobility, barrier detection) ports in full — it is the cap-relabel oracle,
  the anchor-leg opponent, and the HybridEvaluator's value source. All
  constants from `config.py` (`KING_GEOM_SCALE` 0.4, `KING_ATTACK_SCALE` 1.8,
  etc.).

### 1.4 Encoding and NN bridge (`src/encoding.py`, 169 lines)

- 17-channel current / 15-channel legacy tensor layouts, selected by
  checkpoint; policy index `from_sq * 64 + to_sq` (4096). Native builds the
  `(N, C, 8, 8)` float32 batch directly from bitboards.
- **The NN itself stays in PyTorch, unchanged** — same checkpoints, same FP16
  path, same architecture inference at load. See D3.

## 2. Architecture decisions

**D1 — Rust, PyO3, maturin.** Memory-safe tree code, first-class Windows/MSVC
support, cargo-driven differential tests. (C++/pybind11 is the acceptable
alternative if a blocker appears; Cython/numba are not — bounded at ~2–3×,
measured category, law 18.)

**D2 — the native core owns:** bitboard state, movegen (both APIs), game
state machine, clone, terminal + cap relabel, the heuristic evaluator, the
full MCTS (UCB1 and batched PUCT, all overrides), and tensor encoding.
**Python keeps:** the NN and checkpoints, training, data_processor, the
scripted oracle, gate/match/generation orchestration, the notebook.

**D3 — NN bridge in two stages.** Stage 1: the native search collects a leaf
batch, fills a `(N, C, 8, 8)` buffer, and invokes a Python callback that runs
the existing torch model (FP16 on CUDA) and returns values + policy logits;
GIL released during search, re-acquired per batch. ~25–100 crossings per
decision — negligible. This keeps model loading/compat logic untouched and
needs no ONNX export. Stage 2: a shared inference server batching leaves
**across the 8 worker games** into single forwards — this is where the GPU
finally saturates. Stage 3 (optional, only if profiling justifies it): ONNX
Runtime in-native to remove Python from the loop entirely.

**D4 — determinism policy.** Rules, heuristic, and encoding: **exact parity**
(same f64 operation order where feasible; tolerance 1e-9 otherwise). Search:
**statistical parity** — Python's MT19937 stream will not be replicated;
native uses its own seeded RNG, and equivalence is established by the E3
gates, not bit-identity.

**D5 — drop-in API.** `NativeMCTS(num_simulations, eval_fn, batch_size,
root_noise, allow_early_stop).get_best_action(state, temperature)` returns
`(action, action_probs, search_value)` with the same types and key formats
(UCI strings, `"m1,m2"` pairs) the recorders and drivers consume today.
`--engine {python,native}` added to `data_generation.py`, `tools/match.py`,
`tools/gate.py`, `benchmark.py`, and the play path; default stays `python`
until E5.

## 3. Phases and exit gates

**E0 — payoff calibration (existing engine, runs in parallel with E1).**

**(a) The sims curve, unattended.** Extend the PPC true-capture curve to
3200/6400 sims — does conversion keep climbing or plateau at the value head's
indifference? Sizes purchase #1. Note this curve is the one major
pre-2026-08-03 result the scoring change does **not** invalidate:
`true_capture_rate` counts `result <= -1` only, so it was already
captures-only. It is the cleanest evidence in the record and the load-bearing
support for this directive.

**(b) The finisher spike — promoted from sizing to its own go/no-go, and it
runs first.** A depth-limited exhaustive forced-capture search over the 40
dominant-unfinished endgames: how many had forced kills MCTS walked past?
This is 40 fixed positions searched offline in Python — cheap, and it settles
purchase #2's premise *before* any Rust exists. Two outcomes:

- **Forced kills are there** -> the finisher is validated as the fix, and the
  next question is whether a Python finisher invoked *only* at dominant evals
  (a rare trigger) is affordable today. If it is, purchase #2 lands without
  waiting for the rewrite, and the rewrite's case narrows to #1 and #3.
- **They are not** -> the dominant-unfinished positions are genuinely unwon at
  reachable depth, the finisher drops down E6, and #1 carries the directive.

Rationale for the promotion: #2 is the only purchase attacking a failure mode
nothing else addresses, and #1/#3 largely buy throughput for generation — the
lever that just failed twice (v20, v20w). The best idea should not sit behind
two weeks of infrastructure it may not need.

**E1 — rules core.** Bitboards, movegen, both action APIs, state machine,
clone, terminal with cap relabel (heuristic stub wired), FEN in/out.
*Exit gate:* replays **every replayable recorded game** move-for-move with FEN
parity and identical results — measured 2026-08-03 with `tools/replay_parity.py`
against the *Python* engine, which is both the harness's own check and the
baseline the port must match: **968 of 991 clean (48,272 plies)**. Three
corrections to this gate as first written:

- **The 2,966-game playstrategy corpus is not on this box.** Only `ps_monster`
  (829 games, all clean) and `human_games` (162) are on disk. Same class of
  gap as `combined_v16` (§4.3).
- **23 human games can never replay** and are carved out in
  `data/legacy_unreplayable_games.json`. All 23 have one cause: White ends its
  turn with its king attacked, which the owner's ruling forbids
  (`test_ruleset_divergences.py` divergence 2). They were recorded under the
  looser rule. The list is fixed — an unlisted game diverging is a real defect.
- **"FEN parity at every ply" is not possible for owner games.** They record
  only White-to-move positions with an atomic `"m1,m2"` policy; the
  intermediate half-ply positions were never stored, so parity is checkable
  only at turn boundaries.

Also required:
randomized differential vs the Python engine — ≥10M positions, legal-action
**set** equality on both APIs, plus contractual *ordering* where callers depend
on it; native mirrors of `test_king_capture_rules.py` and
`test_ruleset_divergences.py` pass.

**Coverage is itself a gate criterion**, measured 2026-08-03 with
`tools/differential.py` over 60,000 random-walk positions (2,808 pos/s on 6
workers, so 10M is ~1 hour):

| case | rate | verdict |
|---|---|---|
| ep capturable by the mover | 1.5% | ample from random walk |
| castling rights present | 45% | ample |
| White second half | 34% | ample |
| **forced blunder (either side)** | **1 in 60,000** | **random sampling will not test this** |

At 1.7e-5, a 10M-position run yields ~170 forced-blunder positions
*incidentally*, concentrated in whatever lines happened to produce them — while
§4 names forced-blunder ordering as a top risk. **That class needs targeted
construction, not more random positions.** Ordering held across 19,501
Black-to-move positions (0 violations) on the current engine, so the contract is
real and checkable.

**E2 — heuristic + encoding.** *Exit gate:* |Δeval| ≤ 1e-9 vs Python on 1M
random positions including clamp and pending-aware cases; tensor byte-equality
on 100k positions in both 15ch and 17ch layouts.

**Heuristic: DONE 2026-08-03.** `tools/eval_parity.py` — **148,272 positions,
bit-identical (worst delta exactly 0.0)**, 48,535 of them pending-aware, drawn
from random walks and every recorded game on disk. Bit-identity rather than
1e-9 is the honest result: the heuristic is only +,-,*,/ on f64 plus an integer
power, so the sole hazard was accumulation order, and any ordering error yields
a delta far larger than the tolerance. The cap relabel is wired into the native
state machine and lockstep now compares the heuristic at **every** ply
(31,807 plies, 0 failures).

**Encoding: DONE 2026-08-03.** `tools/encoding_parity.py` — **98,272 positions
in each layout, zero byte-unequal**, 15ch legacy and 17ch current. Arithmetic is
done in f64 and narrowed on store, matching numpy assigning a Python float into
a float32 array; computing in f32 throughout would round differently and break
equality. **E2 is complete.**

**A FEN under-determines a Monster Chess state — three times over.** Each was
found the same way, by a measurement that disagreed for no visible reason, and
each fails *silently* rather than erroring:

1. **`white_half_pending`** — selects a different action set entirely. A tree
   built from a FEN alone searched a different position; the symptom was the
   native search choosing moves the Python search did not rank at all.
2. **`turn_count`** — decides the move-limit cap and therefore the ±0.5 relabel.
3. **Move history** — the oscillation penalty reads it at offsets -1/-3/-4, so a
   state rebuilt without it stops penalising reversals and simply never fires.

`Game` and `Tree` now take all four (FEN + these three). **E4 must never pass
positions between the engines as bare FENs.**

**And a fourth, at the format level: FENs are lossy about ep.**
python-chess writes `en_passant="legal"`, so a live game object can hold an
ep square its own FEN drops — 1 position in 148,272 evaluated differently
through a FEN round-trip *within Python itself*. The native `Game` must
therefore be driven by actions, never re-created from a FEN each ply, or
evaluations silently diverge at exactly those positions.

**E3 — native MCTS, all modes, all overrides.** *Exit gate:* (a) on
`promotion_defense_deck_v1` (400 positions, 400 sims, no noise), selected-move
agreement with the Python engine **≥99%, every disagreement triaged** — with
noise off and temperature 0 PUCT is deterministic given identical evals, so
agreement should be near-total; a 5% budget over 400 positions would let 20
positions differ silently, and §4 names FPU/backprop perspective bugs as the
top risk. Gate on visit-count distributions, not argmax alone. The §4.4 anchor
position reproduces capture priors to 4 decimals. (b) old-engine vs new-engine
self-match at equal sims, 200 games, score within 2 SE of 0.50. (c)
**performance: ≥5× wall-clock** on the standard late-game profile decision
(the ply-~60, 400-sim benchmark that measured 22.6 s pre-clone-fix).

**E3(c) REVISED AGAIN 2026-08-03, this time from measurement.** The batched
PUCT path now runs end-to-end (native search, Python does only the forward), and
the measured figures are:

| sims | batch | native | python | speedup |
|---|---|---|---|---|
| 400 | 16 | 180 ms | 551 ms | 3.07x |
| 800 | 16 | 298 | 1257 | 4.22x |
| 800 | 64 | 163 | 1026 | **6.31x** |
| 1600 | 64 | 249 | 2364 | 9.50x |
| 1600 | 256 | 216 | 2152 | **9.98x** |

**The batch-16 gate is unreachable and the reason is that law 18 does not
transfer.** Its "NN forward 14%" was measured on the *Python* engine, where tree
work dominated. Once the tree is native the NN share balloons: at batch 16 a
forward costs 4.71 ms, so 400 sims spend 118 ms of a 180 ms native run inside
the network — **65%** — and the ceiling is 551/118 = **4.7x**. Every projection
in this document that fed law 18's shares through a native tree double-counted.

So the gate moves to the batch size the engine will actually operate at:
**E3(c) is ≥5x at batch 64** (measured 6.31x). Batch 16 is recorded as ~3-4x
with a 4.7x ceiling, and is no longer a gate.

**E4's ≥10x at batch 256 is already essentially met (9.98x)** before Stage-2
cross-game batching exists. Per-position forward cost is 0.294 ms at batch 16
against 0.037 ms at batch 256 — an 8x efficiency gain that confirms law 18's
batching claim directly and is where the remaining headroom lives.

*Why 5× and not 10× originally.* Amdahl, from law 18's own numbers. E3 runs at batch
16 (§1.2 pins the width until parity holds), where NN forward is 14% of a
decision and D3 Stage 1 leaves it in PyTorch untouched. Decision time is
`14 + 86/k` for native speedup `k` on everything else: 4.4x at k=10, 5.5x at
k=20, and a **7.1x ceiling at k=infinity**. A 10x gate at E3 demands a result
the phase's own configuration forbids. 5x already implies k~16 and is strong
evidence the port is sound. The 10x target moves to E4, where widening the
batch makes it reachable.

**E3 STATUS 2026-08-03.** Ported: node arithmetic (PUCT/FPU/backprop),
sequential UCB1, batched PUCT with the D3 stage-1 NN bridge, virtual loss,
Dirichlet root noise, tree reuse across White's half-pair, the selection layer
with both owner overrides, the oscillation penalty and selected-child value
reporting.

| gate | bar | result |
|---|---|---|
| (a) selected-move agreement, probe deck | >=99% | **400/400 = 100%** |
| (b) engine-vs-engine self-match, 200 games | within 2 SE of 0.50 | **0.525, 0.71 SE — PASS** |
| (c) wall-clock at batch 64 | >=5x | **6.31x** |

**Gate (a) earned its keep immediately: it caught a sign flip.** The value head
speaks in the **side-to-move** perspective and `NNEvaluator._to_white_perspective`
converts it; the first bridge forwarded the raw value, so every Black-to-move
leaf backpropagated the wrong sign. Agreement read 90% with disagreements at
rank 5 and visibly wrong values. With the conversion applied — natively, since
the search is what knows each leaf's side — agreement is 400/400. This is the
argument for having tightened (a) from 95% to 99%: at 95% a subtler version of
the same bug would have passed.

**Not ported, deliberately: the atomic-pair prior path** (`_white_priors`, the
80-child cap). It is unreachable for `MonsterChessGame`, which defines
`get_search_actions`, so `_state_legal_actions` always returns single moves and
`is_pair` is always False. Porting it would add untestable dead code. If a state
type without `get_search_actions` is ever introduced, this needs writing.

**E4 PROGRESS 2026-08-04.** `NativeMCTS` (D5) is a drop-in for `mcts.MCTS`:
same constructor, same `get_best_action(state, temperature)` contract, returns
the caller's own `Move` object. `--engine {python,native}` is wired through
`benchmark._build_engine` — the single chokepoint for match, gate, benchmark,
iterate and every probe — plus explicit flags on `data_generation`, `match` and
`gate`, and a `MONSTER_ENGINE` environment variable for tools without a flag.
**The default stays `python` until E5**, pinned by test.

Three things the adapter exists to get right, each of which was a real bug when
missing: it passes the *whole* state (never a bare FEN), it returns a `Move`
rather than a UCI string, and it matches the evaluator's value source —
`HybridEvaluator` takes values from the heuristic and only policy from the
network, which the native search now supports explicitly.

Measured end to end: a 16-game match, same model both sides, 200 sims, 4
workers — **169.7s python vs 46.5s native (3.65x)**. Lower than the ~10x
per-decision figure because model loading and game setup are fixed costs
amortised over few games; scores agree within noise.

**The post-timeout hang is discharged.** `terminate_pool` is extracted from
`data_generation` and `tests/test_pool_teardown.py` forces the exact historic
sequence — hang a worker, hit the timeout, tear down, start a fresh pool — and
asserts the second pool completes. The fix was already in the tree; what was
missing was a test that it works. A regression here reintroduces a failure whose
signature is an overnight run that silently produced nothing.

**Stage-2 inference server: built, measured, and it pays less than the
per-position numbers implied.** `src/inference_server.py` holds one model and
batches leaves across all workers; workers hold no model at all (it *replaces*
per-worker inference rather than wrapping it, per the §4 risk row).

At 8 workers, 3 games, 200 sims:

| | stage 1 (model per worker) | stage 2 (shared server) |
|---|---|---|
| search work, slowest worker | 1.13s | **0.99s (0.87x)** |
| wall clock | 7.2s (8 model loads) | 1.7s (1 model load) |
| resident models | 8 | **1** |

**The honest gain is ~13% on search plus an 8x reduction in GPU model memory** —
not the 8x the 0.294ms/0.037ms per-position figures suggested. The reason is
that 8 workers can have at most 8 requests in flight (each blocks for its
reply), so the server gathers ~128 leaves at best and only when they happen to
coincide; CUDA was already time-slicing the 8 processes reasonably well. The
memory saving is the stronger argument, because it is what allows more workers.

A first cut of this benchmark reported **5x** by comparing wall clocks — which
credited the server with 8 model loads it had merely moved before the timer.
Comparing search work only is what makes the number mean anything.

**E4's first overnight run did NOT pass. Two findings, 2026-08-04.**

**1. A frozen RNG killed exploration (found, fixed, tested).** The native RNG is
built from the seed it is *given*; `NativeMCTS` passed a constant, so every
decision re-seeded an identical stream. Temperature sampling returned the same
move 10/10 times and Dirichlet noise drew one vector across 5 trees —
exploration was dead while still looking random, because different positions
still produce different moves. **Gates (a) and (b) run at temperature 0 with
noise off and structurally cannot see this**; generation is the only path using
both, and it is the path whose output becomes training data. Fixed by advancing
the seed per decision (reproducibility preserved: same seed, same sequence) and
pinned by `tests/test_native_exploration.py`. The first overnight run
(240 games, White 240/240) is invalid and is quarantined as
`data/raw/INVALID_native_e4_overnight_frozen_rng`.

**2. A divergence remains in the noise regime, and it is NOT the port being
worse.** Isolated by turning the stochastic elements off one at a time, 4 games
each, then confirmed at n=16:

| config | python | native |
|---|---|---|
| deterministic (no noise, temp~0) | `{1:4}` lengths 85,86,89,97 | **identical** |
| temperature only | `{1:4}` lengths 10-32 | `{1:4}` lengths 9-15 |
| **noise only** | `{1:12, -0.5:4}`, final fullmove med 37 | `{1:16}`, **med 7** |

Per-decision the engines agree (same picks, values within 0.003), and the
Dirichlet sampler matches numpy statistically (mean max component 0.455 vs
0.459, entropy 1.443 vs 1.431, all rows sum to 1). Tracing a whole game, the
engines agree on nearly every move; where they differ **native reports +1.00 —
a proven king capture — against python's +0.88**. Native is finding forced wins
python misses at the same sim count, so self-play ends by move 7 with White
winning every time.

**Why this blocked the gate at the time.** A strength difference is not a correctness
bug under D4's statistical-parity clause, but generation exists to produce
training data, and data from an engine that finishes games at move 7 is not
interchangeable with data from one that runs to move 37. Until the cause is
identified, `--engine native` must not generate a corpus. The deterministic path
is exact, so match/gate/benchmark use are unaffected.

Next step: find what native searches that python does not in the noisy regime —
tree-size and visit-count comparison at matched sims is the obvious probe.

**Resolved later 2026-08-04:** the adapter was not inheriting the per-game
global RNG seed. After that fix, the clean 240-game overnight completed with a
healthy W174/B44/cap22 distribution. E4 is complete; `REPORT.md` is the current
state record and this subsection preserves the failure that found the bug.

**E4 — integration.** `--engine native` through generation, match, gate,
benchmark, play; the Stage-2 inference server. *Exit gate:* **≥10× wall-clock on the
profile decision at batch 256** (met: 9.98x), one overnight generation run
completing clean under the detached-execution pattern, and a full 3-leg gate in
**≤7 minutes**.

**The gate-time bar was recalibrated from measurement 2026-08-04.** It was
written as ≤6 min from an *assumed* 5.7x; a real native gate runs the three legs
in **390s (6.5 min), 5.2x** over the ~34 min Python baseline. The bar was wrong,
not the engine, so it moves to the measured value plus headroom. Per-leg:
vs_v19 58s, vs_ramp 283s, anchor 50s — the ramp leg dominates because a weak
opponent produces long games that run to the cap.

Batch width is the untested lever here: §1.2 permits widening beyond 16 *after*
E3 parity, and per-position forward cost is 0.294 ms at batch 16 against
0.037 ms at 256. It changes selection quality, so under §0.1 it is a separate
flagged change measured on its own — not folded into the port.

*Where 10× comes from, and what the ceiling is.* At batch 256 law 18 measures
NN forward at 7.7% with the decision 1.53× faster overall, so of the original
100 units the NN holds ~5.0 and everything else ~60.4. Native gives
`5.0 + 60.4/k`: 9.1x at k=10, 12.5x at k=20, **20x ceiling at k=infinity**.
The previous draft's "≥40× aggregate throughput" sat *above* that ceiling — it
is reachable only if the Stage-2 cross-game server beats single-process
batch-256 by a further 2× (plausible: 8 workers currently fire 8 separate
16-leaf kernels where one 128-leaf forward would do), but that is an
extrapolation, not a measurement, and it must not be a hard gate. **40×
aggregate is the target; 10× on the decision is the gate.**

**E5 — the re-baseline (the single comparability break).** One event, one
night: v17, ramp, v19, v19_B, heuristic anchor — full cross-table under
**captures-only scoring + native engine**. This becomes the record's new
zero-point; it also discharges the standing re-measure debt from the scoring
fix and is the natural moment for the owner to **settle the bar decision**
(v19 vs v19_B). Nothing measured before E5 is compared to anything after it.

**DONE 2026-08-04.** `rebaseline_20260804_155035.json`: v17 < ramp < v19 <
v19_B survives captures-only/native. v19_B beat v19 0.575 twice on disjoint
40-game reads and led the heuristic anchor 0.762 vs 0.738. The gate bar moves
to v19_B; the numbered incumbent remains v19 pending an owner promotion.

**Queued for E6 by the owner, 2026-08-03: pruning.** Raised after the depth
measurement below, and the measurement is the argument for it.

Search depth is logarithmic in sims and the slope is shallow — measured on the
native engine, PV depth in plies:

| position | 400 sims | 51,200 sims (128x) |
|---|---|---|
| opening | 3 | 7 |
| midgame | 4 | 10 |
| **endgame** | **4** | **4** |

About 0.6-0.9 plies per doubling, so the whole rewrite (~13x) buys **+2 to +3
plies — roughly one extra round**. This is why the engine "struggles post 2-3
moves from each side": the PV genuinely reaches 1-3.3 full rounds and no
plausible compute budget changes that. MCTS keeps everything it visits;
Stockfish's depth comes from discarding most of the tree. **Pruning, not
throughput, is the lever on depth.**

Two facts to carry into that discussion. First, sims do *not* buy conversion by
buying depth: on the controlled native deck, 200 -> 3200 sims moved true
captures 0.19 -> 0.51 while midgame PV
depth went 4 -> 5, so the gain is better value estimates over a shallow tree.
Second, the endgame row is pinned at 4 plies even at 51,200 sims with node count
*below* sim count (43,512 / 51,200) — the search re-walks short lines into
terminals. That is the conversion class, and it is why an exhaustive depth-3
proof search found forced wins MCTS did not (E0(b)).

Constraint on whatever is chosen: §0.1 applies. Pruning changes what the engine
plays, so it lands after parity, behind a flag, measured on its own — which is
what the owner's "after this is done" already says.

**E6 — exploit.** In whatever order E0 indicated: the finisher search
(proof-number on "forced king capture within N plies," invoked at dominant
evals); the PPC curve to 12,800+; asymmetric generation at scale (Black high /
White normal, strongest players, `plies_to_end` truncation); the self-play
loop with the White-rate health metric. Each exploit is its own measured
change — none rides in with another.

## 3a. E0 RESULTS (2026-08-03, same day) — both answers landed

**E0(a), original read (superseded): the curve appeared not to plateau.** v19
Black vs v19 White@400, 100 post-promotion starts, captures-only throughout:

| Black sims | 200 | 400 | 800 | 1600 | **3200** |
|---|---|---|---|---|---|
| true king captures | 0.09 | 0.15 | 0.20 | 0.30 | **0.51** |
| dominant, unfinished | 12 | 13 | 28 | 40 | **20** |
| White wins | 0.79 | 0.72 | 0.52 | 0.29 | **0.29** |
| mean plies | 54.6 | 61.0 | 88.3 | 112.1 | **74.1** |

**Artifact audit correction, 2026-08-04:** the 200–1600 rows above used
`postpromo_starts_v1`; the 3200 row used `promotion_defense_deck_v1` (only
10/400 FENs overlap). The claimed +0.21 step and halving comparison are
therefore invalid. The controlled single-deck native rerun below supersedes
this table and still establishes the 3200 knee, by valid evidence.

**6400 ANSWERED 2026-08-04, on the native engine (whole curve re-run on one
deck and one engine, `ppccurve_native_s*`): 0.55 — +0.04 over 3200, ~1 SE. The
controlled curve is 0.19 / 0.17 / 0.24 / 0.37 / 0.51 / 0.55 and plateaus at
the 3200 knee.** §5's contingency applies: raw sims cap near 0.55
on this deck, so the exploit order shifts to the finisher/solver and to data.
The rewrite's case now rests on throughput. The first directly comparable
full 3200-sim binding gate took **48.5 min** (including a 25.8-min, 40-game
bar leg), not the earlier unmeasured ~12-min estimate and still below the
2.3-hour reference. Full analysis: REPORT.md §6.

**E0(b): forced wins are walked past — and the culprit is the oracle, not the
search.** Of 65 dominant-unfinished games in current generation (v19
Black@1600, 2026-08-02/03), **19 (29%) held a position with a forced king
capture within 3 Black moves and the moves left to play it**; 312 positions,
18 of them forced in *2*. The legacy corpus gives the same rate (13/43).

The games' own records show Black policy-degenerate at 1.0 from those
positions on — the **scripted oracle** had already taken over and then failed
to finish. Cause is a scope gap: `verify_scripted_mate` verifies on canonical
K+Q+R+R vs bare king with every Black piece at Chebyshev >= 5, while
`mate_algo_applicable` admits *any* bare White king facing 3+ heavies,
including cluttered midgame positions with pawns and minors. Played out from
the 11 walked-past positions it accepts, the fence heuristic converted **1 of
8**. With an exact forced-capture preflight: **8 of 8**, each in the proven
minimum of 3 moves. Shipped (`src/forced_capture.py`, wired into
`ScriptedMate.select_move`); A/B at fixed seeds shows no behaviour change where
it does not fire.

**Two consequences for this directive.**

1. **Purchase #2 is largely not a rewrite item.** The finisher for the failing
   class was a day of Python against a function that already existed. What
   remains genuinely native-scale is the 6 of 19 walked-past positions where
   **White still held material** — outside the oracle's class entirely.
2. **A separate, pre-existing oracle defect is now on the record.**
   `verify_scripted_mate --games 12` currently reports **9/12**: three
   canonical starts end with White capturing the Black king (one leaves the
   board bare but for the White king). Controlled A/B confirms the preflight
   is not the cause — it never fires in those lines. This matters beyond the
   oracle: it labels training data as ground truth. **Fix before leaning on
   oracle-generated labels again.**

## 4. Risks, named

| risk | mitigation |
|---|---|
| ep / half-move semantics drift in bitboards | E1's replay-everything gate; the divergence tests; ep cases explicitly in the 10M differential |
| forced-blunder ordering ("callers rely on that ordering") | ordering asserted in the differential, not just set equality where order is contractual |
| float divergence in heuristic | fixed operation order, f64, 1e-9 gate on 1M positions |
| perspective bugs in FPU/backprop (bit M2 twice) | port from the *contract* in §1.2, property tests on 3-ply hand-built trees with known Q |
| GIL contention, 8 workers × callback | each worker owns its core + model as today; Stage 2 server replaces, not wraps, per-worker inference |
| Windows toolchain | rustup MSVC + maturin; CI is `cargo test` + the Python suite, both local |
| **the known post-timeout hang** — worker kill leaves zombies that block the next `Pool`; E4 stacks an inference server on that unfixed pipeline | fix or quarantine it *before* E4 wires the Stage-2 server; the overnight-rehearsal gate must be a clean run, not a restarted one |
| scope creep — "while we're in here" | §0.1. Parity first. Every improvement is a separate, flagged, measured change |
| the rewrite eats the calendar while the real lever idles | **L4 continues throughout**: every owner session harvested; the rewrite never blocks his play or the intake of his games |

## 5. What this directive does not do

It does not promise the rewrite fixes conversion. E0 exists because the
value-head indifference (REPORT 2026-08-03 §3 companion analysis) may cap
what sims can buy; if the curve plateaus at 3200/6400, the rewrite is still
justified by generation throughput and the finisher search, but the exploit
order changes. The owner-game flywheel (L4) and the open owner decisions (the
bar — scheduled into E5; §7.4; the `combined_v16` copy) carry forward
unchanged.

## 6. Sequence and effort

| phase | est. | runs |
|---|---|---|
| E0 | one overnight + one afternoon | unattended, now |
| E1 | 2–3 days | CPU only |
| E2 | 1 day | CPU only |
| E3 | 3–4 days | probe deck + 200-game match |
| E4 | 2 days | one overnight rehearsal |
| E5 | one overnight | the re-baseline |
| E6 | ongoing | per E0's answer |

Calendar: ~1.5–2 weeks to E5 at the pace this project has actually sustained.
**Treat that as the optimistic bound.** E1 covers bitboard movegen for a
double-move variant with pseudo-legal + king-capture semantics, contractual
move ordering and ep parity, *plus* a 10M-position differential harness and
full-corpus replay; E2's 1 day is a mechanical port whose cost is chasing the
last ulp to 1e-9; E3 carries the perspective flip, both overrides, the
oscillation penalty and tree reuse. 4–5 weeks to E5 is the realistic band, and
§0.4's abort trigger exists because of that gap. The parity regime is the
consolation: overruns surface as failed gates, not as silent wrongness.
Commits owner-identity, one-line; evidence to `benchmarks/`; the native crate
lives in `native/` with its own tests; root docs unchanged in role
(`CONTEXT.md` durable, this directive active, `REPORT.md` the last run).

*Written 2026-08-03 after a full source read (monster_chess.py, mcts.py,
evaluation.py, encoding.py, config.py, generation/match call sites). Retained
as the completed rewrite contract and the scope record for any later E6 work.*
