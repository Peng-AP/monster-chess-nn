# Monster Chess NN - Execution Order

**Date:** 2026-02-18
**Input:** `IMPROVEMENT_PLAN.md` (revised)

This is the concrete implementation order with gates. Do not skip gates.

**Version control rule:** Each stage is a separate git commit and push. Commit message format: `Stage N: <short description>`.

---

## Stage 0 - Baseline Snapshot (No Behavior Changes)

1. Record current baseline metrics from latest model:
- self-play result distribution
- curriculum tier eval summary
- train/val/test metrics currently reported

2. Save baseline artifact bundle:
- metrics json/txt
- key config values
- model hash/checksum

**Gate to proceed:** baseline artifacts saved and reproducible.

---

## Stage 1 - Evaluation Hygiene (Blocking Stage)

### Step 1.1 Game-level split

Files:
- `src/data_processor.py`

Tasks:
- Split by game files before conversion/augmentation.
- Ensure no game appears in more than one split.

Acceptance:
- Script/assertion confirms zero overlap at game id level.

### Step 1.2 Deterministic seeds + run metadata

Files:
- `src/train.py`
- `src/iterate.py`

Tasks:
- Add seed controls (CLI + defaults).
- Persist metadata per training run.

Acceptance:
- Two runs with same seed/data produce near-identical first-epoch metrics.

### Step 1.3 Metric correctness

Files:
- `src/train.py`

Tasks:
- Add true MSE metric.
- Rename power-loss metric to avoid ambiguity.

Acceptance:
- Logs clearly distinguish power loss vs MSE.

**Gate to proceed:** Stage 1 complete and baseline rerun done on corrected split.

---

## Stage 2 - Stability Patch Set (Single PR)

### Step 2.1 Gradient clipping
### Step 2.2 AdamW + weight decay parameter groups
### Step 2.3 Resume-from checkpoint + warmup
### Step 2.4 Position-budget sliding window
### Step 2.5 Configurable tactical filtering
### Step 2.6 FPU in PUCT path only

Files:
- `src/train.py`
- `src/iterate.py`
- `src/config.py`
- `src/data_processor.py`
- `src/data_generation.py`
- `src/mcts.py`

Acceptance:
- Training runs without instability spikes seen in baseline.
- Offline metrics improve or remain neutral.
- No regression in quick manual play sanity checks.

**Gate to proceed:** candidate passes arena test versus incumbent (temporary manual arena acceptable).

---

## Stage 3 - Permanent Gating Infrastructure

### Step 3.1 Arena command/tooling

Files:
- `src/iterate.py` or new `src/arena.py`

Tasks:
- Candidate vs incumbent matches with color swap.
- Return win/draw/loss and score.

### Step 3.2 Promotion policy

Files:
- `src/iterate.py`

Tasks:
- Promote model only if score >= threshold.
- On fail, keep incumbent.

Acceptance:
- Iteration loop can reject a candidate cleanly.

**Gate to proceed:** at least one accepted and one rejected candidate observed in test runs.

---

## Stage 4 - Architecture Track (One Change at a Time)

### Step 4.1 Policy head widening

Files:
- `src/train.py`

Acceptance:
- Arena non-regression and improved policy metrics.

### Step 4.2 Backbone 8x128

Files:
- `src/train.py`

Acceptance:
- Stable training and arena improvement vs Stage 2 model.

### Step 4.3 SE blocks

Files:
- `src/train.py`

Acceptance:
- Arena gain or no-regression with lower variance.

### Step 4.4 WDL head (separate experiment)

Files:
- `src/train.py`
- `src/evaluation.py`
- `src/mcts.py`
- possibly `src/data_processor.py`

Acceptance:
- End-to-end compatibility validated.
- Arena gain over best scalar-value model.

**Gate to proceed:** each sub-step individually gated before stacking next one.

---

## Stage 5 - Self-Play/Data Robustness

### Step 5.1 Opponent pool
### Step 5.2 Asymmetric sims policy
### Step 5.3 Playout-cap randomization
### Step 5.4 Adaptive curriculum allocation

Files:
- `src/iterate.py`
- `src/data_generation.py`
- `src/config.py`
- optionally `src/train.py` and `src/data_processor.py`

Acceptance:
- Improved arena trend over multiple generations.
- Reduced oscillation between generations.

---

## Stage 6 - Advanced Research Additions

### Step 6.1 Auxiliary heads
### Step 6.2 Separate White/Black networks
### Step 6.3 Dynamic curriculum generation

Acceptance:
- Each feature is introduced behind flags.
- Each passes gating before default enablement.

---

## Operational Rules

1. Only one major axis change per experiment batch (data, optimizer, architecture, search).
2. Keep a changelog with experiment id -> code diff -> metrics -> gate outcome.
3. Default to rollback on failed gate.
4. Do not overwrite incumbent model on failed gate.

---

## Minimal First Sprint (Recommended)

1. Stage 1 fully.
2. Stage 2 fully.
3. Stage 3 fully.

That gives a trustworthy and safe training loop before spending compute on larger architecture work.
