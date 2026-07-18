# Rework Plan: From Random Play to a Working Self-Play Loop

*Drafted 2026-07-02 from a full code review. Supersedes `DIRECTIVE_HYBRID_MCTS.md` sequencing
(the hybrid idea survives, but it runs in Phase 4 on top of a fixed search, not before).*

---

## 1. Executive summary

The engine plays near-randomly because of two compounding pipeline defects, not because the
approach is wrong:

| # | Defect | Location | Effect |
|---|--------|----------|--------|
| D1 | Batched PUCT wastes entire batches on unexpanded nodes. At the sim budgets actually used (arena 80, self-play presets 100–200, `batch_size=128`), **zero root children are ever visited** — move selection is `random.choice`, policy targets are uniform, `mcts_value` is the raw NN output of the root. | `src/mcts.py:290-344` | All NN self-play data ≈ random games with uniform policy labels; arena gates ≈ coin flips |
| D2 | Value head trains on its own output. `VALUE_TARGET="source_aware"` with `SELFPLAY_TARGET_MCTS_LAMBDA=1.00` makes the training target the MCTS root value — which, via D1, is the network's own evaluation fed back to it. Game outcomes (and the forced curriculum tier labels, stored in `game_result`) never reach the value head for self-play-sourced data. | `src/config.py:230,260-263`, `src/train.py:468-481` | Self-distillation fixed point → flat ~0 value calibration (the persistent D2-diagnostic failure) |

Empirical confirmation (probe run 2026-07-02, heuristic value + flat policy through
`_run_batched_puct`, standard opening):

| sims | root children visited | top-5 child visits | outcome |
|------|----------------------|--------------------|---------|
| 80 (arena) | 0 / 80 | 0,0,0,0,0 | uniform random move |
| 120 (preset) | 0 / 80 | 0,0,0,0,0 | uniform random move |
| 200 | 70 / 80 | 2,2,1,1,1 | noise |
| 800 | 80 / 80 | 25,23,18,15,14 | weak signal; first 128 sims wasted |

Everything else in the repo — source quotas, result filters, keep-ratio floors, adaptive gate
scales, black-focus arenas, consolidation presets (~5,000 lines across `iterate.py`,
`data_processor.py`, `train.py`, `config.py`) — is compensation machinery built while chasing
symptoms of D1+D2. Once the core is fixed and validated, most of it is scheduled for deletion
(Phase 5). Current `src/` is ~10,900 lines; target after Phase 5 is ~4,500–5,000.

**Phase order is deliberate: instrument → fix search → fix targets → fix White's action
factorization → validate the loop → then delete.** Deleting first would leave nothing to
validate against; deleting last means every removal is justified by a green benchmark.

---

## 2. Design principles

1. **Outcome-grounded labels.** The value head trains on game results (or provably forced
   labels), never on the search's own value estimates, until the loop demonstrably works.
2. **Outcomes, not assertions — even where the result is known.** Project experimentation
   and consultation with stronger players (2026-07) established that **Black is winning with
   correct play**: freeze or trade the four pawns, barricade the double-move king out, then
   promote at leisure with the Black king kept safe. Observed White dominance (~72 % in
   heuristic self-play) is therefore a search-horizon artifact. This does **not** change the
   labeling policy: a value head fed asserted constants ("this class of position = −0.3")
   receives a position-independent offset with no move-level gradient — it learns *that*
   Black is better, never *which* moves convert. Tier 7–10 forced values and the asymmetric
   move-limit relabeling are still removed; the known result is used as the measuring stick
   instead (Phase 6).
3. **One yardstick.** Heuristic-MCTS (UCB1 sequential path — the only search in the repo that
   currently works) becomes a permanent fixed benchmark anchor. Every candidate is measured
   against it, not only against the incumbent.
4. **Delete after validation, not before.** Each phase has acceptance criteria; Phase 5's
   deletions are gated on Phase 4 passing.

---

## 3. Phase 0 — Instrument before changing (~½ day)

### 0.1 Search-quality contract test

**New file** `tests/test_mcts_search_quality.py`:

- From the standard opening with a stub evaluator (heuristic value, flat policy logits,
  exposing `batch_evaluate_with_policy`), run `MCTS(num_simulations=N)` for N in {80, 200, 800}:
  - assert `sum(child.visit_count for child in root.children) >= 0.8 * N`
  - assert max child visits > 2 × uniform share
  - assert the returned `action_probs` are not uniform (max/min ratio > 2 at N=800)
- This test **fails today** at N=80/200 — that is the point. It pins the Phase 1 fix and
  prevents regression forever.

### 0.2 Benchmark anchor

**New file** `src/benchmark.py` (~150 lines; absorbs the useful part of
`baseline_snapshot.py`):

- Plays `--games N` (default 100, fixed seeds) between a candidate model and the heuristic
  UCB1 engine at `--sims` (default 400), both colors, temperature 0, no Dirichlet noise,
  `record_all_plies` off, no data saved.
- Reports: score, Black-win share, mean game length, mean time/move. Writes one JSON to
  `benchmarks/`.
- Run it once now against `models/best_value_net.pt` to record the "before" number
  (expected: ≈ random, ~0 % as Black).

**Acceptance:** contract test red at 80/200 sims; benchmark JSON for the incumbent exists.

**Complexity impact:** +2 small files now; `baseline_snapshot.py` (303 lines) and
`gate_sweep.py` (407 lines) retire in Phase 5.

---

## 4. Phase 1 — Fix the search (1–2 days)

All changes in `src/mcts.py` unless noted.

### 1.1 Root pre-expansion

In `get_best_action` (`mcts.py:163`): before entering `_run_batched_puct`, evaluate the root
synchronously (`eval_fn.evaluate_with_policy`), expand it with policy priors, apply Dirichlet
noise once (if enabled), and backpropagate the root value. Removes the pathological case where
an entire first batch selects the same unexpanded root.

### 1.2 Pending-set batching

In `_run_batched_puct` (`mcts.py:290-344`):

- Keep a `pending: set[id(node)]` during batch collection. If `_select_puct` returns a node
  already pending, **stop filling the batch** (virtual loss cannot diversify an unexpanded
  frontier; continuing only creates duplicate GPU work).
- Count `sims_done` by completed backpropagations, not by selections attempted.
- Deduplicate states in the eval batch (currently duplicates are forwarded through the NN
  redundantly).

### 1.3 Batch size

Default `batch_size` 128 → **16** (`mcts.py:139`). For an 800-sim tree, 128-wide in-tree
parallelism destroys selection quality. GPU throughput should come from parallelism *across*
games (workers already provide this), not within one tree. Leave the parameter tunable.

### 1.4 Dirichlet noise and temperature become explicit

- `MCTS.__init__` gains `root_noise: bool = True`. Arena/eval/benchmark/play construct with
  `root_noise=False`; only self-play generation keeps it on. Today noise is unconditional
  (`mcts.py:281-288,336-339`) and corrupts arena measurement and human play.
- `data_generation.py` `_run_arena` callers in `iterate.py` pass temperature ≈ 0 and
  `--record-all-plies`; verify noise-off is plumbed through (`--no-root-noise` flag on
  `data_generation.py`, forwarded from `iterate.py:409-505`).

### 1.5 Early stop must not corrupt training targets

`_should_stop_early` (`mcts.py:146-161`) truncates the visit distribution that *is* the policy
training target. Gate it behind `allow_early_stop=True` and set it False whenever records are
kept (self-play generation); keep it for play/arena where only the move matters.

### 1.6 Policy-target smoothing rescale

`POLICY_TARGET_PSEUDOCOUNT=0.25` per child (`config.py:209`, applied `mcts.py:184-191`) adds
20 mass across 80 children — at 80–200 visits, over half the recorded target is smoothing
noise. Change to a *total* smoothing mass (e.g. 0.03 × total_visits spread uniformly) or 0.

### 1.7 FPU generality (prepares Phase 3)

`puct_score` (`mcts.py:67-82`) flips the parent Q assuming strict side alternation. Replace
the unconditional flip with a comparison of `node.parent.state.is_white_turn` at parent vs.
grandparent (no behavior change today; required once White's turn becomes two plies).

**Acceptance:** Phase 0 contract test green at all sim counts; probe shows ≥90 % of sims
reaching children; `benchmark.py` on the *heuristic-value* NN wrapper (hybrid evaluator) at
800 sims scores ≥ the pure-heuristic engine at 400 sims.

**Complexity impact:** compute — removes up to 128 duplicate NN forwards per batch cycle and
the wasted first batch (≈16 % of an 800-sim search, ≈100 % of an 80-sim search). Code — net
+40 lines here, enables large deletions later.

---

## 5. Phase 2 — Ground the value target (~½ day + one retrain)

### 2.1 λ → 0 everywhere

`src/config.py:260-263`: set all four `*_TARGET_MCTS_LAMBDA` to 0.0, and
`VALUE_TARGET = "game_result"` (`config.py:230`). Update the README command examples that
recommend `--target mcts_value` (README §4). The whole `source_aware`/`blend` apparatus in
`train.py:468-501` and `data_processor.py` becomes dead code — **do not delete yet** (Phase 5);
just stop routing through it.

Rationale: the value target must contain information the network does not already have. Game
outcome z does; the root value of a search whose leaves are the network's own value head does
not (and with D1 it was literally the raw network output). AlphaZero trains on z. A λ≤0.3
blend may be revisited *after* the loop works, when root Q genuinely contains search
improvement.

### 2.2 Value head de-regularization

`train.py:221-233` (`_make_value_head`): remove both `Dropout(0.3)` layers. With weak targets,
0.3+0.3 dropout on a 128-dim GAP feature drives predictions to the mean (~0) — it is a direct
contributor to the flat calibration. Weight decay (already configured, `build_optimizer`)
is the regularizer. Optional, recommended: make the existing WDL head
(`--value-head wdl`, `train.py:235-243`) the default — CE over win/draw/loss calibrates better
than tanh regression with a 2.5-power loss, and the code already exists.

### 2.3 Stop labeling aborted games as draws

`data_generation.py:413-418`: a game cut by the wall-clock deadline exits the loop
non-terminal and `get_result()` returns 0 — false "draw" labels. Return a
`(records, aborted: bool)` pair from `play_game` and **discard aborted games** (or label with
the heuristic sign, as move-limit games already do).

### 2.4 Remove belief-encoded labels (principle 2)

- `config.py:193` `CURRICULUM_TIER_VALUES`: keep forced labels only for provably forced tiers
  (Tier 1–2, value −1.0). Tiers 3–10 switch to live results
  (`--curriculum-live-results` becomes the default for those tiers) — with the Phase 1 search
  fixed, live outcomes are meaningful.
- `monster_chess.py:53-59`: the move-limit relabel (−0.5 when heuristic < −0.4) has no White
  counterpart. Make it symmetric (+0.5 when > +0.4) or remove it.

### 2.5 Retrain

Fresh model (`models/fresh_start_v3`) on the existing clean corpus, `--target game_result`,
no dropout, WDL head. Check the D2 diagnostic (avg prediction by game result).

**Acceptance:** D2 shows directional separation for the first time (White-win avg > +0.15,
Black-win avg < −0.15 — the directive's own modest bar). If it does not, the corpus itself is
the next suspect (it was generated by the broken search; Phase 4 regenerates it anyway).

**Complexity impact:** none yet in LOC; conceptually collapses four λ knobs + two blend knobs
+ annealing into a single choice.

---

## 6. Phase 3 — Half-move tree for White (2–4 days)

The current factorization cannot learn White's play:

- Priors: `_white_priors` (`mcts.py:384-408`) gives P(m1,m2) = P(m1)/|m2-group| — the second
  move, where every Monster Chess tactic lives, has a **uniform** prior forever.
- Targets: `policy_dict_to_target` (`data_processor.py:112-136`) marginalizes m2 away, so the
  policy head is never trained on it.
- The top-80 child cap (`mcts.py:109-124,371`) interacts with the group-uniform priors to
  systematically favor first-moves with *few* continuations (per-pair prior = P(m1)/|group|),
  i.e. it prunes toward self-restricting moves.
- `_get_white_actions` (`monster_chess.py:94-162`) enumerates ~900 (m1,m2) pairs with pushes/
  pops per node expansion — the dominant CPU cost of the whole pipeline.

### 3.1 Game layer (`src/monster_chess.py`)

Add half-move support alongside the atomic API:

- New state field `white_half_pending: bool` (False except between White's m1 and m2).
  Include it in `clone()`.
- `get_legal_half_moves()`: when White-to-move and not pending → all pseudo-legal White
  moves; when pending → all pseudo-legal White second moves, filtered to those restoring king
  safety, falling back to all if none are safe (mirrors the existing pair-level forced-blunder
  rule closely; the search sees the losing continuation and avoids the m1 that forces it).
- `apply_half_move(move)`: pushes one move; toggles `white_half_pending`; only flips
  `is_white_turn` and increments `turn_count` when the second half (or Black's move) completes.
- **Keep the atomic pair API for actual game play** (`play.py`, arena result adjudication) so
  the strict pair-level legality semantics of real games are unchanged; the half-move API is
  the search/training decomposition.

### 3.2 Encoding (`src/data_processor.py`)

`fen_to_tensor` (`data_processor.py:44-78`): the reserved `MOVE_COUNT_LAYER` (13) — currently
always 0, a dead input plane — becomes 1.0 on White's second half-move. Signature grows a
`half_pending=False` parameter. FEN alone no longer identifies the state; records must carry
the flag (3.4).

### 3.3 Search (`src/mcts.py`)

- White nodes expand over single moves (~30 children), producing an intermediate
  White-pending node which expands over second moves (~30 children). Branching per White turn
  drops from ~900 enumerated pairs to ~60 across two plies; **delete** `_white_priors`, the
  m1-grouping, and the top-80 cap.
- `_backpropagate` (`mcts.py:440-450`) already uses the actual parent side — correct for
  non-alternating plies. FPU flip fixed in Phase 1.7. Verify perspective with a unit test:
  a forced White king-capture two half-moves deep must back up as +1 at the White root.
- Terminal shortcut: if a half-move captures the Black king, the node is terminal (+1),
  matching current pair semantics.

### 3.4 Data (`src/data_generation.py`, `src/data_processor.py`)

- Records gain `"half": 0|1` and store single-move UCI policies for both White halves (two
  policy training samples per White turn instead of one marginalized sample).
- `policy_dict_to_target` simplifies to direct `move_to_index` for all records — the
  White marginalization branch is deleted.
- Keep reading legacy records (m1 marginal, half=0) for one transition; drop after Phase 4
  regenerates data.

**Acceptance:** benchmark score of the hybrid evaluator at 800 sims strictly improves vs.
Phase 1; node-expansion CPU time per White turn drops (expect ≥5× on `_get_white_actions`
elimination inside the tree); policy top-1 match measured on *both* halves.

**Complexity impact:** compute — removes the O(n²) pair enumeration from every tree node;
code — net negative (~−120 lines: priors/cap/marginalization out, half-move plumbing in);
model — the dead input plane becomes live, 4096 policy space now matches the action space.

---

## 7. Phase 4 — Re-run the loop and validate (compute-bound, ~1 week elapsed)

1. **Hybrid generation** (NN policy + heuristic value — `HybridEvaluator`,
   `evaluation.py:477-509`) at 800 sims, 150 games (opening + midgame starts), per the
   original directive — now running on a working search. Also fix the double capture-scan:
   `HybridEvaluator.batch_evaluate_with_policy` runs `_white_can_capture_king` once in the NN
   wrapper and again inside `evaluate` — pass precomputed scan results or short-circuit.
2. **Retrain** on hybrid + curriculum + human data, `--target game_result`.
3. **Probe pure-NN self-play** (20 games, 800 sims): directive success criteria apply
   (≥1 Black win, mean length > 15).
4. **Gate simplification trial:** promotion requires (a) ≥55 % vs incumbent over ≥100 arena
   games at ≥400 sims, noise off, temp 0, both colors; (b) no regression vs the heuristic
   anchor (`benchmark.py`). That is the whole gate. The side-aware thresholds, floors,
   black-focus arenas and adaptive scales are bypassed (deleted in Phase 5) — they were tuned
   on coin-flip arenas and have never measured anything.

**Acceptance:** a full generate → process → train → gate cycle completes where the promoted
model beats both the incumbent and the previous benchmark anchor score. This unlocks Phase 5.

---

## 8. Phase 5 — Deletion schedule (1–2 days, gated on Phase 4)

Every deletion below is justified by "the simple loop validated without it." Do them as one
reviewed commit series, running the contract tests + one smoke iterate cycle after each group.

### 8.1 Code deletions

| Target | Location | ~Lines | Replacement |
|--------|----------|-------:|-------------|
| Side-aware/black-focus/adaptive gating, consolidation mode, humanseed plumbing, black-survival runs | `iterate.py` (3,045 total) | −2,400 | ~500-line loop: generate → process → train → arena+anchor gate → archive |
| Source quotas, interleaving, capacity estimation, rebalancing, per-source stats/warnings, retention filters beyond age+min-plies, result filters + keep-ratio floors | `data_processor.py:148-330,390-461,683-782,915-964,1287-1336` | −700 | flat conversion + game-level split |
| `source_aware`/`blend` targets, λ arrays, distillation, side-specialized heads, balanced-side sampling, side/result train filters | `train.py:462-626,690-706` + head plumbing | −450 | `game_result` target (+ optional WDL), plain epochs |
| Quota/λ/blend/data-weight/retention constants (~25 knobs) | `config.py:230-263` | −60 | ≤8 constants |
| `gate_sweep.py` | whole file | −407 | benchmark.py JSON history |
| `scripted_endgame.py` + `--scripted-black` path | whole file + `data_generation.py:392-394,430-444` | −280 | live MCTS Black (already default) |
| `baseline_snapshot.py` | whole file | −303 | `benchmark.py` |
| `generate_bruteforce.py` | move to `tools/` (calibration one-off) | −485 from `src/` | — |
| humanseed/blackfocus start-FEN conversion machinery | `data_generation.py:111-273` | −160 | plain `--start-fen-file` |
| Obsolete contract tests for deleted subsystems | `tests/` | −600 | keep + extend core contracts |

Net: `src/` ~10,900 → **~4,700 lines**; `config.py` from ~45 tunables to ~15.

### 8.2 Repository structure

```
monster-chess-nn/
├── README.md              # rewritten: current commands only, no stale doc map
├── REWORK_PLAN.md         # this file (archive to docs/ when done)
├── src/
│   ├── config.py          # ≤120 lines
│   ├── monster_chess.py   # rules (+ half-move API)
│   ├── mcts.py            # fixed batched PUCT + UCB1
│   ├── evaluation.py      # heuristic, NNEvaluator, HybridEvaluator
│   ├── encoding.py        # fen_to_tensor, move_to_index, mirror (split out of data_processor)
│   ├── data_generation.py
│   ├── data_processor.py
│   ├── train.py
│   ├── benchmark.py       # heuristic-anchor evaluation
│   ├── iterate.py         # ~500-line loop
│   └── play.py
├── tools/                 # generate_bruteforce.py, human_eval.py, notebooks
├── tests/                 # core contracts incl. search-quality test
└── benchmarks/            # anchor JSON history
```

Also: README's "Documentation Map" references four files that no longer exist
(`IMPROVEMENT_PLAN.md`, `IMPROVEMENT_EXECUTION_ORDER.md`, `TRANSFER_HANDOFF_2026-02-19.md`,
`context.txt`) — remove; `requirements.txt` is stale by its own admission — regenerate.

### 8.3 Computation cost after Phases 1+3+5

Per-move cost at 800 sims, dominated today by (a) duplicate/wasted NN forwards (Phase 1),
(b) O(n²) White pair enumeration per expansion (Phase 3), (c) double capture-scans in hybrid
mode (Phase 4.1). Expected combined effect: ≥3× more effective sims per second, which can be
spent as either faster generation or deeper search at equal cost. Additional cheap wins,
optional: cache `_white_can_capture_king`/`_black_can_capture_king` per position hash;
transposition-free tree is fine at this scale.

---

## 9. Phase 6 — The balance ladder (Black share vs. depth as the strength metric)

Established conclusion (project owner, confirmed by experimentation and stronger players,
2026-07): **Black is winning with correct play.** Method: freeze or trade the four White
pawns, barricade the double-move king behind heavy-piece barriers, promote at leisure while
keeping the Black king out of 2-move capture range. The mechanisms are the ones the codebase
already gestures at: forced double-move is zugzwang leverage (`_get_white_actions`
forced-blunder branch is the terminal case), K+Q+2R vs. bare king is a forced win, and
White's plans are enumerable (promote, or king raid).

This turns the old "who is better?" experiment into the project's **primary strength
ladder**: since the game-theoretic result is a Black win, the engine's distance from correct
play is measurable as the gap between its Black-win share and 100 %. White dominance at low
depth (72 % heuristic, ~100 % broken-NN) is the horizon-artifact signature — the repo's own
tier data shows Black converting 67 % from Tier 6, 50 % from Tier 7, 28 % from move 1.

Protocol (after Phase 4, engine fixed and non-random):

1. Fixed best engine, standard opening, both sides identical, noise off, temp 0.
2. 200 games each at 400 / 1,600 / 6,400 sims (use the ≥3× speedup from §8.3).
3. Plot Black score vs. sims per promoted generation. **Expected shape: rising with depth
   and rising generation-over-generation.** A plateau far below 100 % at high depth localizes
   what the engine cannot yet find (usually the barricade construction — verify with probe 4).
4. Secondary probe: from Tier 8/9 structural positions, measure how often deep search finds
   the Black conversion (pawn freeze → barricade → promotion) the heuristic already scores.

Implications applied elsewhere in this plan: labels stay outcome-grounded despite the known
result (§2 principle 2 — asserted constants carry no move-level gradient); gate and benchmark
sensitivity concentrates on Black-side performance, since Black-side improvement is where all
remaining headroom lives.

---

## 10. Schedule summary

| Phase | Work | Effort | Gate to next |
|-------|------|--------|--------------|
| 0 | Contract test + benchmark anchor | ½ day | test red at 80/200 sims |
| 1 | Search fix (root pre-expand, pending set, batch 16, noise/temp flags, early-stop, smoothing, FPU) | 1–2 days | test green; hybrid ≥ heuristic anchor |
| 2 | λ=0, dropout out, WDL, abort/label fixes, belief-label removal, retrain | ½ day + retrain | D2 separation ±0.15 |
| 3 | Half-move tree (game API, encoding plane, search, records) | 2–4 days | benchmark improves; both halves trained |
| 4 | Hybrid regen → retrain → NN self-play probe → simple gate | ~1 week elapsed | full cycle promotes a model beating anchor |
| 5 | Deletions + restructure (−6,000 lines, tools/, README) | 1–2 days | smoke cycle green post-deletion |
| 6 | Balance ladder (Black share vs. depth, per generation) | compute only | ongoing strength metric |

Total hands-on effort ≈ 6–10 working days; Phases 4/6 are mostly compute.
