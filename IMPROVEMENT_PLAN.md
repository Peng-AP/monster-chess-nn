# Monster Chess NN - Revised Improvement Plan

**Date:** 2026-02-18
**Status:** Revised after code-level review of `src/` and training pipeline behavior.

---

## Executive Summary

The original plan had many good ideas, but it mixed high-risk architecture changes with missing evaluation hygiene. The revised plan prioritizes **measurement correctness first**, then low-risk stability fixes, then architecture/search upgrades with strict gating.

Core principle:
- If validation is leaky or promotion is ungated, later improvements can look good while actual play gets worse.

---

## Guiding Rules

1. Fix evaluation hygiene before model redesign.
2. Promote models only via arena gating, never by default.
3. Introduce one major change class at a time (data, optimizer, architecture, search).
4. Keep all key knobs in `config.py` or CLI flags for fast ablation.
5. Each stage is a separate git commit and push. No stage is merged silently.

---

## Phase 0 - Evaluation Hygiene (Must Do First)

Goal: make metrics trustworthy and reproducible.

### 0.1 Split by game, not by position

**File:** `src/data_processor.py`

Current split happens after position expansion/augmentation, so near-duplicate positions from the same game can land in train and val/test. This inflates metrics.

Change:
- Split at game-file level first (train/val/test game sets), then convert positions.
- Apply augmentation only within each split.

### 0.2 Add deterministic run metadata

**Files:** `src/train.py`, `src/iterate.py`

Change:
- Seed Python, NumPy, and Torch consistently.
- Save run metadata (seed, git commit hash if available, key hyperparameters, data counts).

### 0.3 Fix metric labeling and add true MSE readout

**File:** `src/train.py`

Current code prints "Value MSE" while optimizing/recording power loss. Keep power loss but report true MSE and MAE explicitly.

### 0.4 Add arena evaluation harness

**File:** `src/iterate.py` (or new `src/arena.py`)

Add head-to-head evaluation between candidate and incumbent models with both color assignments. This is prerequisite for gating.

---

## Phase 1 - Stability and Low-Risk Quality Wins

Goal: improve training stability without changing game semantics.

### 1.1 Gradient clipping

**File:** `src/train.py`

Add `clip_grad_norm_` before optimizer step.

### 1.2 Weight decay with decoupled optimizer

**File:** `src/train.py`

Use AdamW with `weight_decay=1e-4`, excluding BatchNorm and bias terms from decay (parameter groups).

### 1.3 Fine-tune from previous best model

**Files:** `src/train.py`, `src/iterate.py`

Add `--resume-from` and warmup schedule for first few epochs. Keep bootstrap-from-scratch for initial model only.

### 1.4 Replace fixed sliding window with position-budget policy

**Files:** `src/config.py`, `src/data_processor.py`, `src/iterate.py`

Instead of fixed `SLIDING_WINDOW=2`, target a minimum number of recent positions (for example 400k-800k) and include enough generations to hit that budget.

### 1.5 Add FPU reduction in PUCT path only

**File:** `src/mcts.py`

Implement FPU inside `puct_score`/selection logic for unvisited children; do not globally alter `q_value` for all modes.

### 1.6 Make tactical filtering configurable

**File:** `src/data_generation.py`

Current rule drops all check positions. Move this behind a flag so it can be ablated. In this variant, check-like king pressure may carry important tactical signal.

---

## Phase 2 - Controlled Architecture Upgrades

Goal: increase representational capacity without destabilizing training.

### 2.1 Policy head bottleneck fix first

**File:** `src/train.py`

Increase policy head channel width (for example 2 -> 16/32) before deeper backbone changes.

### 2.2 Backbone expansion

**File:** `src/train.py`

Move to 8x128 residual tower only after Phase 1+gating is stable.

### 2.3 Add SE blocks

**File:** `src/train.py`

Add SE after residual conv stack for incremental gain.

### 2.4 Delay WDL head until baseline is stable

**Files:** `src/train.py`, `src/evaluation.py`, `src/mcts.py`

WDL is worthwhile, but touches labels, losses, evaluator plumbing, and MCTS interfaces. Treat as separate experiment after capacity and stability improvements are validated.

---

## Phase 3 - Search and Self-Play Robustness

Goal: improve data quality and avoid model regressions in iterative self-play.

### 3.1 Enforce network gating

**File:** `src/iterate.py`

Require candidate >= threshold (for example 55%) in arena games to replace incumbent.

### 3.2 Opponent pool (3-5 snapshots)

**Files:** `src/iterate.py`, `src/data_generation.py`

Sample opponents from a pool to reduce co-adaptation to a single frozen net.

### 3.3 Asymmetric simulation budgets

**Files:** `src/config.py`, `src/iterate.py`, `src/data_generation.py`

Give Black more simulations while training Black; keep policy configurable and measured.

### 3.4 Playout-cap randomization

**Files:** `src/data_generation.py`, `src/data_processor.py`, `src/train.py`

Randomize sims per move/game and optionally weight losses by search reliability.

### 3.5 Adaptive curriculum allocation

**Files:** `src/iterate.py`, `src/data_generation.py`

Reallocate curriculum games toward tiers with weakest current performance while preserving minimum coverage.

---

## Phase 4 - Advanced/Research Additions

Goal: maximize efficiency after core pipeline is reliable.

### 4.1 Auxiliary heads (KataGo style)

Predict extra targets such as moves-to-terminal and piece survival.

### 4.2 Separate side-specific networks

Two-model setup for White and Black if shared-network limits remain after Phases 1-3.

### 4.3 Dynamic curriculum generation

Generate curriculum positions from real trajectories rather than relying only on fixed FEN bank.

---

## Explicit De-prioritization (for now)

1. Full migration to SGD before stabilizing and gating current pipeline.
2. Large semantic rule changes (handicapped rules) before proving baseline improvements.
3. Multiple major architecture+training changes in one iteration.

---

## Validation Protocol (Required per Major Change)

For each change set, run:

1. Training stability: no exploding loss, smoother val curves.
2. Offline metrics: true value MSE, MAE, policy CE/top-k.
3. Curriculum diagnostics: tier-wise value targets and errors.
4. Arena gating: candidate vs incumbent over fixed game count and color balance.
5. Regression guard: if gate fails, reject candidate and keep incumbent.

---

## Exit Criteria for "Plan Successful"

1. Gated promotion acceptance settles in healthy range (roughly 40-80%).
2. White win rate in self-play decreases materially from current baseline without collapse in overall play quality.
3. Curriculum tier evaluations become monotonic and consistent with intended difficulty.
4. Improvements persist for multiple generations, not just one-run spikes.
