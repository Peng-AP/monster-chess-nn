# Monster Chess NN - Improvement Plan (Living)

**Last updated:** 2026-02-19  
**Scope:** Current repo state on this machine (`main`)

## Status Legend

- `complete`: implemented in code and actively used
- `partial`: implemented in part, missing key behavior
- `pending`: not implemented yet

## Executive Summary

The core reliability foundation is now in place:

- game-level split hygiene
- deterministic seeds and run metadata
- candidate gating with side-aware thresholds
- stability defaults (AdamW, weight decay groups, grad clipping, warmup fine-tune)

The next bottlenecks are search/data quality and model capacity for Black-side play.

## Guiding Rules

1. Keep gating mandatory for promotion.
2. Change one major axis per experiment batch.
3. Keep every tunable exposed via `config.py` or CLI.
4. Track every experiment with run metadata and gate outcome.
5. Roll back by rejection, not by deleting history.

## Phase 0 - Evaluation Hygiene

### 0.1 Split by game, not position: `complete`

- Implemented in `src/data_processor.py` via game-level grouping and split integrity checks.

### 0.2 Deterministic seeds + metadata: `complete`

- Implemented in `src/train.py` and `src/iterate.py` (seed control + JSON run metadata).

### 0.3 Metric correctness (power loss vs true MSE/MAE): `complete`

- Implemented in `src/train.py` logging and test summary.

### 0.4 Arena evaluation harness for gating: `complete`

- Implemented in `src/iterate.py` (`_run_arena` + gate decision logic).

## Phase 1 - Stability and Low-Risk Quality

### 1.1 Gradient clipping: `complete`

- Implemented in `src/train.py` (`clip_grad_norm_`).

### 1.2 AdamW + decoupled weight decay: `complete`

- Implemented with no-decay groups for bias/norm params in `src/train.py`.

### 1.3 Fine-tune from incumbent + warmup: `complete`

- Implemented with `--resume-from`, `--warmup-epochs`, `--warmup-start-factor`.

### 1.4 Position-budget data window (replace fixed generation window): `partial`

- Implemented min-budget selection via `POSITION_BUDGET` and `--position-budget` in
  `src/config.py`, `src/data_processor.py`, and `src/iterate.py`.
- Remaining gap: optional max-cap behavior is not yet implemented.

### 1.5 FPU reduction in PUCT path: `complete`

- Implemented as `FPU_REDUCTION` in `src/config.py`, used in PUCT path in `src/mcts.py`.

### 1.6 Configurable tactical filtering in generation: `complete`

- Implemented via `SKIP_CHECK_POSITIONS` default and `--keep-check-positions` control path
  in `src/data_generation.py` and `src/iterate.py`.

## Phase 2 - Controlled Architecture Upgrades

### 2.1 Policy head widening: `complete`

- Implemented via `POLICY_HEAD_CHANNELS` (currently `16`) in `src/config.py` and wired in
  `src/train.py`.

### 2.2 Backbone expansion (e.g., 8x128): `complete`

- Implemented via configurable `STEM_CHANNELS` and `RESIDUAL_BLOCK_CHANNELS` in
  `src/config.py`, with dynamic tower construction and checkpoint-architecture inference
  in `src/train.py`.

### 2.3 SE blocks: `complete`

- Implemented optional SE modules in residual blocks via:
  - `USE_SE_BLOCKS` / `SE_REDUCTION` in `src/config.py`
  - SE-enabled residual blocks and checkpoint inference support in `src/train.py`
  - iterate passthrough flags in `src/iterate.py` (`--use-se-blocks`, `--se-reduction`)
- Validated with SE-enabled training smoke test and SE-enabled gated iterate runs.

### 2.4 WDL value head experiment: `pending`

## Phase 3 - Search and Self-Play Robustness

### 3.1 Enforced network gating: `complete`

- Candidate promotion is gate-controlled by default in `src/iterate.py`.

### 3.2 Opponent pool: `complete`

- Implemented via archive pool options in `src/iterate.py` + `src/data_generation.py`.

### 3.3 Asymmetric simulation budgets: `complete`

- Implemented with train-side/opponent simulation split.

### 3.4 Playout-cap randomization: `complete`

- Implemented per-game simulation budget sampling via
  `--simulations-min/--simulations-max` in `src/data_generation.py`.
- Wired from iteration loop via `--selfplay-sims-jitter-pct` and
  `SELFPLAY_SIMS_JITTER_PCT` in `src/iterate.py` / `src/config.py`.
- Generation-level simulation stats are persisted in
  `generation_summary.json` and carried into iterate metadata summaries.

### 3.5 Adaptive curriculum allocation: `complete`

- Implemented adaptive game-mix scheduling in `src/iterate.py`:
  - `--adaptive-curriculum`
  - side-specific scaling with bounded range and update factors
  - per-iteration `effective_mix` and `adaptive_update` metadata
- Validated with a smoke gated iteration (`iterate_run_20260219_115558.json`).

### 3.6 Anti-forgetting consolidation pass: `complete`

- Implemented teacher-distilled consolidation in `src/iterate.py`:
  - `--consolidation-epochs`
  - `--consolidation-lr-factor`
  - `--consolidation-batch-size`
  - `--consolidation-balance-sides`
  - `--consolidation-distill-*`
- Implemented training-side controls in `src/train.py`:
  - `--balanced-sides-train`
  - `--distill-from`
  - `--distill-value-weight`
  - `--distill-policy-weight`
  - `--distill-temperature`
- Validated with a full iteration smoke run (`iterate_run_20260219_125036.json`)
  showing primary + consolidation timing and enabled distillation metadata.

## Phase 4 - Advanced Additions

### 4.1 Auxiliary heads: `pending`

### 4.2 Separate White/Black networks: `pending`

### 4.3 Dynamic curriculum generation: `pending`

## Current Priority Queue

1. Continue multi-iteration gated alternating validation with consolidation + SE + adaptive curriculum + playout randomization.
2. Add optional max-cap behavior to position-budget windowing.
3. WDL value head experiment.
4. Separate White/Black network experiment.

## Validation Protocol (Required Per Major Change)

1. Training stability: loss curves and no gradient spikes.
2. Offline metrics: power loss, true MSE, MAE, policy CE.
3. Gate outcome: candidate vs incumbent, color-balanced.
4. Alternating mode check: trained side and non-trained side thresholds both pass.
5. Human eval trend check when relevant (`src/human_eval.py`).

## Exit Criteria

1. Candidate acceptance settles in a healthy non-trivial range.
2. Black-side gate score improves without collapsing White-side play.
3. Improvements persist across multiple generations.
4. No metric inflation from data leakage or split contamination.
