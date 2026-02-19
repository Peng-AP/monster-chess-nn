# Monster Chess NN - Execution Order (From Current State)

**Last updated:** 2026-02-19  
**Input:** `IMPROVEMENT_PLAN.md` (living version)

This is the practical next sequence given what is already implemented.

## Stage Status Snapshot

- Stage 0 (baseline snapshot tooling): `complete` tooling, `refresh needed` baseline artifact
- Stage 1 (evaluation hygiene): `complete`
- Stage 2 (stability patch set): `partial` (position-budget max-cap still open)
- Stage 3 (gating infrastructure): `complete`
- Stage 4+ (architecture/research): `partial` (policy widening + backbone scaling implemented)

## Immediate Sprint (Recommended)

### Stage A - Refresh Baseline Artifacts

Files:

- `src/baseline_snapshot.py`
- `baselines/`

Tasks:

1. Run a fresh baseline snapshot on current data/model.
2. Record model hash, split sizes, and gate-relevant metrics.
3. Save artifact with timestamp in `baselines/`.

Gate:

- Baseline JSON exists and matches current repository state.

### Stage B - Position-Budget Windowing (Finish Remaining Gap)

Files:

- `src/config.py`
- `src/data_processor.py`
- `src/iterate.py`

Tasks:

1. Keep current min-budget behavior.
2. Add optional max-cap for deterministic budget banding.
3. Keep existing include/exclude behavior for curriculum/human data.

Gate:

- Processing reports deterministic generation selection and budget hit within configured bounds.

### Stage C - Gated Validation Sweep (Current Priority)

Files:

- `src/iterate.py`
- `models/iterate_run_*.json`
- `models/arena_runs/`

Tasks:

1. Run short alternating gated iterations with current architecture defaults.
2. Compare trained-side and non-trained-side arena scores against prior runs.
3. Confirm whether acceptance rate improves over the previous rejection-only streak.

Gate:

- At least one accepted candidate or a clear metric trend that justifies next change axis.

### Stage D - Baseline Refresh Artifact

Files:

- `src/baseline_snapshot.py`
- `baselines/`

Tasks:

1. Capture fresh baseline JSON with current incumbent and processed split.
2. Store model hash and evaluation metrics for future comparison.

Gate:

- Baseline artifact exists and references current incumbent hash.

## Next Sprint (After Stage A-D)

### Stage E - Playout-Cap Randomization (Implemented)

Files:

- `src/data_generation.py`
- `src/mcts.py`

Tasks:

1. Added bounded randomization over simulation budgets per game.
2. Added generation summary artifacts with sampled simulation statistics.

Gate:

- Implemented in code and validated in a gated run; continue monitoring for acceptance impact.

### Stage F - Adaptive Curriculum Allocation

Files:

- `src/iterate.py`
- `src/data_generation.py`

Tasks:

1. Shift curriculum/normal/black-focus mix based on recent gate outcomes.
2. Keep hard bounds to avoid destabilizing data distribution.

Gate:

- Better trained-side gate scores without collapse on the other side.

## Operational Rules

1. One major change axis per stage.
2. Do not disable gating for normal development runs.
3. Keep each stage as an isolated commit.
4. On failed gate, keep incumbent and continue from known-good state.

## Suggested Commit Titles

1. `Stage A: Refresh baseline snapshot artifact`
2. `Stage B: Add position-budget max-cap windowing`
3. `Stage C: Run gated validation sweep for expanded backbone`
4. `Stage D: Refresh baseline snapshot for new architecture`
5. `Stage E: Add playout-cap randomization`
