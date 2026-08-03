# DIRECTIVE — the engine rewrite (2026-08-03)

**Owner's call: the residual problem may be compute.** Search is Python-bound
(CONTEXT law 17: NN forward 14%, python-chess movegen ~35%, clone/apply
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
4. **Nothing else changes.** Labels, corpora, thresholds, the scoring rule,
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
measured category, law 17.)

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

**E0 — payoff calibration (existing engine, unattended, blocks nothing).**
(a) Extend the PPC true-capture curve to 3200/6400 sims — does conversion keep
climbing or plateau at the value head's indifference? (b) The forced-capture
probe over the 40 dominant-unfinished endgames — how many had forced kills
MCTS walked past? (a) sizes the high-sim payoff; (b) sizes the finisher
search. Both shape E6, neither blocks E1.

**E1 — rules core.** Bitboards, movegen, both action APIs, state machine,
clone, terminal with cap relabel (heuristic stub wired), FEN in/out.
*Exit gate:* replays **all 2,966 playstrategy games and every human game on
disk** move-for-move with FEN parity at every ply and identical results;
randomized differential vs the Python engine — ≥10M positions, legal-action
**set** equality on both APIs including forced-blunder and ep cases; native
mirrors of `test_king_capture_rules.py` and `test_ruleset_divergences.py`
pass.

**E2 — heuristic + encoding.** *Exit gate:* |Δeval| ≤ 1e-9 vs Python on 1M
random positions including clamp and pending-aware cases; tensor byte-equality
on 100k positions in both 15ch and 17ch layouts.

**E3 — native MCTS, all modes, all overrides.** *Exit gate:* (a) on
`promotion_defense_deck_v1` (400 positions, 400 sims, no noise), selected-move
agreement with the Python engine ≥95%, and the §4.4 anchor position
reproduces capture priors to 4 decimals; (b) old-engine vs new-engine
self-match at equal sims, 200 games, score within 2 SE of 0.50; (c)
**performance: ≥10× wall-clock** on the standard late-game profile decision
(the ply-~60, 400-sim benchmark that measured 22.6 s pre-clone-fix).

**E4 — integration.** `--engine native` through generation, match, gate,
benchmark, play; the Stage-2 inference server. *Exit gate:* a full 3-leg gate
in ≤5 minutes (today: ~34); aggregate throughput ≥40× current 7.1
decisions/s; one overnight generation run completing clean under the
detached-execution pattern.

**E5 — the re-baseline (the single comparability break).** One event, one
night: v17, ramp, v19, v19_B, heuristic anchor — full cross-table under
**captures-only scoring + native engine**. This becomes the record's new
zero-point; it also discharges the standing re-measure debt from the scoring
fix and is the natural moment for the owner to **settle the bar decision**
(v19 vs v19_B). Nothing measured before E5 is compared to anything after it.

**E6 — exploit.** In whatever order E0 indicated: the finisher search
(proof-number on "forced king capture within N plies," invoked at dominant
evals); the PPC curve to 12,800+; asymmetric generation at scale (Black high /
White normal, strongest players, `plies_to_end` truncation); the self-play
loop with the White-rate health metric. Each exploit is its own measured
change — none rides in with another.

## 4. Risks, named

| risk | mitigation |
|---|---|
| ep / half-move semantics drift in bitboards | E1's replay-everything gate; the divergence tests; ep cases explicitly in the 10M differential |
| forced-blunder ordering ("callers rely on that ordering") | ordering asserted in the differential, not just set equality where order is contractual |
| float divergence in heuristic | fixed operation order, f64, 1e-9 gate on 1M positions |
| perspective bugs in FPU/backprop (bit M2 twice) | port from the *contract* in §1.2, property tests on 3-ply hand-built trees with known Q |
| GIL contention, 8 workers × callback | each worker owns its core + model as today; Stage 2 server replaces, not wraps, per-worker inference |
| Windows toolchain | rustup MSVC + maturin; CI is `cargo test` + the Python suite, both local |
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
Commits owner-identity, one-line; evidence to `benchmarks/`; the native crate
lives in `native/` with its own tests; root docs unchanged in role
(`CONTEXT.md` durable, this directive active, `REPORT.md` the last run).

*Written 2026-08-03 after a full source read (monster_chess.py, mcts.py,
evaluation.py, encoding.py, config.py, generation/match call sites). This is
the active campaign document; it retires when the native engine is the
default and the E5 re-baseline is on record.*
