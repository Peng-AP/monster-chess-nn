# Monster Chess NN - Execution Order (From Current State)

**Last updated:** 2026-02-21  
**Input:** `IMPROVEMENT_PLAN.md` (living version)

This is the practical next sequence from the current cleaned pipeline.

## Stage Status Snapshot

- Stage 0 (pipeline reliability): `complete`
- Stage 1 (data controls): `complete` core + audit diagnostics
- Stage 2 (search/gating): `complete` core + calibration tooling
- Stage 3 (model work): `complete` core architecture options, `pending` next experiments
- Stage 4 (code quality): `partial`

## Immediate Sprint (Recommended)

### Stage A - Gate Correctness Fix (Black-Focus Enforcement): `complete`

Files:

- `src/iterate.py`

Tasks:

1. Enforce `black_focus_pass` in acceptance when black-focus arena is enabled.
2. Add clear rejection reason line when black-focus threshold is the failing condition.
3. Preserve metadata compatibility for historical run analysis.

Gate:

- Completed: black-focus arena outcome now directly affects acceptance in alternating Black runs.

### Stage B - Data/Label Audit Artifact: `complete`

Files:

- `src/data_processor.py`
- `src/iterate.py`
- `models/iterate_run_*.json`

Tasks:

1. Add per-source diagnostics in processing output (counts, value stats, policy entropy summary).
2. Persist diagnostics into iterate metadata per generation.
3. Add hard warnings for low-diversity or high-skew source streams.

Gate:

- Completed: processing now emits per-source counts/value stats/policy-entropy
  diagnostics plus warnings, and iterate logs these diagnostics from
  `processing_summary.json`.

### Stage C - Gate Calibration Sweep Tool: `complete`

Files:

- `src/iterate.py`
- new script under `src/` (e.g. `gate_sweep.py`)

Tasks:

1. Extend the current sweep utility (`src/gate_sweep.py`) with stronger recommendation logic.
2. Report acceptance sensitivity by side-aware thresholds.
3. Recommend calibrated defaults for `--gate-threshold` and side floors.

Gate:

- Completed: sweep now outputs recommendation candidates, confusion-style
  metrics (precision/recall/F1/mismatch), and sensitivity summaries.

### Stage D - Reproducible Run Presets: `complete`

Files:

- `README.md`
- `src/iterate_presets.py`

Tasks:

1. Define `smoke`, `daily`, and `overnight` presets.
2. Ensure presets use mandatory gating and bounded data caps.
3. Document expected runtime and output artifacts.

Gate:

- Completed: presets can be launched via `src/iterate_presets.py` with
  reproducible bounded settings and documented runtime/artifacts.

## Next Sprint (After Stage A-D)

### Stage E - Contract Tests for Core Invariants: `partial`

Files:

- new `tests/` directory
- `src/data_processor.py`
- `src/iterate.py`

Tasks:

1. Add tests for split non-overlap and source quota enforcement.
2. Add tests for gate decision invariants and promotion guard behavior.
3. Add smoke CLI tests for argument schema stability.

Gate:

- Initial local contract suite added under `tests/` and runnable via
  `py -3 -m unittest discover -s tests -v`.
- GitHub Actions workflow added to run the same suite on push/PR:
  `.github/workflows/contract-tests.yml`.

### Stage F - WDL Value Head Experiment

Files:

- `src/train.py`
- `src/evaluation.py`

Tasks:

1. Implement optional WDL head path behind a flag.
2. Keep scalar value head as default fallback.
3. Compare calibration and gate outcomes against incumbent scalar path.

Gate:

- No regression on side-aware gating with improved value calibration signals.

### Stage G - Iteration Module Decomposition

Files:

- `src/iterate.py`
- new modules under `src/` (runner/gating/metadata helpers)

Tasks:

1. Split generation, processing, training, gating, and reporting into modules.
2. Keep CLI surface backward-compatible.
3. Preserve metadata schema where possible.

Gate:

- `iterate.py` becomes thin orchestration with unchanged behavior.

## Operational Rules

1. Keep gating mandatory.
2. Keep major changes isolated per stage.
3. Always include validation artifacts in run commits.
4. Prefer removal of weak features over adding compensating complexity.

## Suggested Commit Titles

1. `Stage A: enforce black-focus gate acceptance path`
2. `Stage B: add per-source data-label diagnostics`
3. `Stage C: add gate calibration sweep tooling`
4. `Stage D: add reproducible iterate presets`
5. `Stage E: add pipeline contract tests`
6. `Stage F: add optional WDL value head experiment`
7. `Stage G: decompose iterate orchestration modules`
