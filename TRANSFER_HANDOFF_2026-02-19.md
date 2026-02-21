# Transfer Handoff (2026-02-21)

This file is transfer-oriented and grounded in currently verified local state.

## 1) Verified Local Snapshot

Verification timestamp: 2026-02-21

### Data

- Highest normal generation directory: `data/raw/nn_gen178`
- Auxiliary generation directories observed:
  - `*_blackfocus`: 132
  - `*_curriculum`: 163
  - `*_humanseed`: 7
- Human game files in `data/raw/human_games`: 30

Processed dataset (`data/processed`):

- `positions.npy` shape: `(353685, 8, 8, 15)`
- `splits.npz`: `train=246881`, `val=53400`, `test=53404`

### Models

- `models/best_value_net.pt` SHA256:
  `F2BA1CD6DAE7CE59E1EB659979C9674A89B44C2BC1C8B69598F57851A5A83FBB`
- `models/frozen_opponent.pt` SHA256:
  `F2BA1CD6DAE7CE59E1EB659979C9674A89B44C2BC1C8B69598F57851A5A83FBB`

### Runs

- `models/iterate_run_*.json` files: 128
- Most recent run files are on 2026-02-20.

## 2) Important Behavioral/Code State

- Gating is mandatory in current `src/iterate.py` (no `--no-gating` path).
- Rejected candidates no longer seed next iterations (no explore-from-rejected path).
- Black-recovery phase has been removed from iteration flow.
- Consolidation phase remains available.
- Gate-threshold sweep utility exists at `src/gate_sweep.py`.
- Processing now writes per-source diagnostics + warnings to
  `processing_summary.json`, and iterate logs/embeds this metadata.
- Gate sweep now includes recommendation metrics (precision/recall/F1/mismatch)
  and threshold sensitivity summaries.
- Relative human path args (`--human-seed-dir`, `--human-eval-dir`) are
  normalized to project-root absolute paths in iterate.
- Preset launcher added: `src/iterate_presets.py` with `smoke`, `daily`,
  and `overnight` profiles.
- Initial contract test suite added in `tests/` (data contracts, guard/path
  contracts, CLI schema smoke checks).

## 3) What Changed Recently (for Continuity)

Recent cleanup focused on reducing harmful/low-signal complexity:

- Removed non-gated promotion mode.
- Removed rejected-candidate exploration continuation.
- Removed black-recovery sub-phase and related knobs.
- Debloated large orchestration sections in `src/iterate.py` and settings helpers.

## 4) Recommended Resume Procedure

1. Refresh a baseline artifact:

```bash
py -3 src/baseline_snapshot.py
```

2. Run a short gated iterate smoke from preset launcher:

```bash
py -3 src/iterate_presets.py --preset smoke
```

3. Preserve after each run:

- `models/iterate_run_*.json`
- `models/archive/manifest.jsonl` and accepted archive checkpoints
- `models/best_value_net.pt`
- new `data/raw/nn_gen*` directories

4. Run contract tests after code changes:

```bash
py -3 -m unittest discover -s tests -v
```

## 5) Quick Verification Commands

```bash
# highest normal generation
powershell -Command "$gens = Get-ChildItem data/raw -Directory | ? { $_.Name -match '^nn_gen(\d+)$' } | % { [int]([regex]::Match($_.Name,'^nn_gen(\d+)$').Groups[1].Value) }; if($gens){ ($gens | Measure-Object -Maximum).Maximum }"

# processed split sizes
py -3 -c "import numpy as np; s=np.load('data/processed/splits.npz'); print(len(s['train']), len(s['val']), len(s['test']))"

# model hash
powershell -Command "(Get-FileHash models/best_value_net.pt -Algorithm SHA256).Hash"
```

## 6) Current Risk Focus

- Side asymmetry is still the central risk: improving Black without causing White-side collapse.
- Data/label quality by source now has automated diagnostics/warnings; next risk
  is calibrating warning thresholds against acceptance outcomes.
- Black-focus arena threshold is now enforced in acceptance for alternating Black-side runs.
