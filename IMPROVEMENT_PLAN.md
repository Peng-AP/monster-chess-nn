# Monster Chess NN - Improvement Plan (Living)

**Last updated:** 2026-02-21  
**Scope:** Current repo state on this machine (`main`)

## Status Legend

- `complete`: implemented in code and actively used
- `partial`: implemented in part, missing key behavior or validation depth
- `pending`: not implemented yet
- `removed`: intentionally removed because it hindered progress

## Executive Summary

The pipeline has a solid reliability base and now enforces gate-controlled promotion.
Recent cleanup removed iteration paths that were producing low-signal regressions.
Per-source processing diagnostics are now first-class artifacts and are surfaced
in iterate logs/metadata for easier skew detection.

The primary bottleneck is no longer basic plumbing. It is now training signal quality and side-balance robustness, especially preventing Black-side strategic collapse while keeping White strength.

Recent code-review fix applied:

- `black_focus_pass` is now enforced in `accepted` gating logic for alternating Black-side runs with black-focus arena enabled.

## Recent Cleanup (2026-02-21)

- Removed `--no-gating` from `src/iterate.py` (`removed`)
- Removed `--explore-from-rejected` branch from `src/iterate.py` (`removed`)
- Removed Black-recovery phase and related CLI from `src/iterate.py` (`removed`)
- Debloated and modularized large sections of `src/iterate.py` and `src/data_processor.py` (`complete`)

Rationale: these paths either bypassed quality control, continued from rejected candidates, or added complexity without measurable acceptance benefit on this repo's run history.

## Guiding Rules

1. Keep promotion gated by arena evidence.
2. Favor fewer, stronger knobs over many interacting knobs.
3. Change one major axis per experiment batch.
4. Persist every run with metadata and explicit gate outcomes.
5. Reject by evidence, not by intuition.

## Phase 0 - Reliability Foundation

### 0.1 Game-level split integrity: `complete`

- Implemented in `src/data_processor.py` with overlap checks.

### 0.2 Deterministic seeds and run metadata: `complete`

- Implemented in `src/train.py` and `src/iterate.py`.

### 0.3 Correct metrics (power loss + true MSE/MAE): `complete`

- Implemented in `src/train.py`.

### 0.4 Mandatory candidate gating path: `complete`

- Enforced in `src/iterate.py` after removal of `--no-gating`.

## Phase 1 - Data and Label Signal Quality

### 1.1 Position-budget windowing with hard cap: `complete`

- Implemented via `POSITION_BUDGET`, `POSITION_BUDGET_MAX`, `PROCESSED_POSITION_CAP`.

### 1.2 Source quotas and source-aware target lambdas: `complete`

- Implemented in `src/data_processor.py` and wired from `src/iterate.py`.

### 1.3 Human/humanseed/blackfocus stream weighting: `complete`

- Implemented with stream-specific train weighting.
- Side-aware data retention now preserves in-check positions for the active
  training side (while still skipping many non-training-side check positions).

### 1.4 Per-source quality audit artifact: `complete`

- Implemented `processing_summary.json` with per-source counts, value stats,
  policy entropy summaries, and warning signals.
- Embedded and printed by iterate per generation via `processed_data_summary`.

### 1.5 Data pruning and retention policy formalization: `pending`

- Codify retention thresholds by generation age and contribution quality.
- Keep processed sample count bounded for throughput stability.

## Phase 2 - Search and Gating Robustness

### 2.1 PUCT FPU reduction and root noise: `complete`

- Implemented in `src/mcts.py` (`FPU_REDUCTION`, Dirichlet mix).

### 2.2 Opponent pool and alternating mode: `complete`

- Implemented in `src/iterate.py` + `src/data_generation.py`.

### 2.3 Black-survival pre-promotion sanity gate: `complete`

- Implemented via `_run_black_survival` and promotion guard checks.

### 2.4 Black-focus gate enforcement wiring: `complete`

- `black_focus_pass` is now included in acceptance when black-focus arena is active.

### 2.5 Gate calibration harness (automated threshold sweeps): `complete`

- `src/gate_sweep.py` now produces recommendation candidates with
  precision/recall/F1/mismatch metrics and sensitivity summaries.
- Sweep artifacts include recommended defaults plus acceptance-top configs.
- Arena and black-survival evaluations now run with deterministic low
  temperature by default to reduce exploratory noise in promotion decisions.

## Phase 3 - Model and Optimization

### 3.1 Backbone scaling + policy head widening: `complete`

- Implemented with configurable stem/tower and wider policy head.

### 3.2 Optional SE blocks and side-specialized heads: `complete`

- Implemented and wired through train/iterate.

### 3.3 Consolidation pass with distillation: `complete`

- Implemented as post-primary optional phase.

### 3.4 WDL-style value head experiment: `partial`

- Implemented optional WDL value-head mode in `src/train.py` and `src/iterate.py`
  with CLI/config wiring, metadata logging, and contract tests.
- Pending: run calibration experiments and tune `wdl_loss_weight` /
  `wdl_draw_epsilon` against gate outcomes.

### 3.5 Stronger side-balance training curriculum: `partial`

- Side-balancing exists, but no automated schedule tied to failure modes.
- Added side-result filtering controls in train/iterate, with automatic
  `nonloss` filtering for alternating Black-side training phases.

## Phase 4 - Code Health and Experiment Velocity

### 4.1 Iteration loop modularization: `partial`

- Significant cleanup done, but `src/iterate.py` is still monolithic.

### 4.2 Regression tests for pipeline contracts: `partial`

- Added `tests/` suite for:
  - data processor split/quota contract checks
  - iterate promotion-guard/path and gate-decision invariant checks
  - CLI schema smoke checks (`iterate`, `data_processor`, `gate_sweep`,
    `iterate_presets`, `train`)
  - processing summary schema and WDL head contracts
- Current local test command:
  `py -3 -m unittest discover -s tests -v`
- Added CI workflow:
  `.github/workflows/contract-tests.yml` (runs contract suite on push/PR)

### 4.3 Reproducible experiment presets: `complete`

- Added `src/iterate_presets.py` with canonical `smoke`/`daily`/`overnight`
  presets.
- Presets enforce bounded data caps and standard gated iterate settings.
- README now documents runtime expectations and produced artifacts.

## Current Priority Queue

1. Execute WDL-mode smoke/daily runs and compare acceptance behavior vs scalar
   value head baselines.
2. Complete data pruning/retention policy implementation (generation-age and
   source-quality aware) while keeping throughput stable under 500k positions.
3. Continue modularization of `src/iterate.py` high-complexity orchestration.

## Validation Protocol (Required Per Major Change)

1. Data checks: split overlap, source counts, quota adherence, cap adherence.
2. Train checks: stable losses, no NaN spikes, consistent convergence shape.
3. Gate checks: overall score + side-specific floors + black-survival result.
4. Human-eval trend check where applicable.
5. Throughput check: end-to-end iteration runtime remains acceptable.

## Exit Criteria

1. Non-trivial candidate acceptance rate under mandatory gating.
2. Black-side arena and black-survival metrics improve without White collapse.
3. Improvements persist across multiple generations and seeds.
4. Data pipeline remains bounded and deterministic under configured caps.
