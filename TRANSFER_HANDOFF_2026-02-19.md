# Transfer Handoff (2026-02-19)

This file is a transfer-oriented summary. It now separates historical notes from currently verified state.

## 1) Historical Snapshot (Read as Context, Not Source of Truth)

The previous handoff captured a run around generation 11 and referenced artifacts such as:

- `models/run_large_20260218_221423.log`
- `models/run_large_20260218_221423.err.log`
- `models/archive/manifest.jsonl`
- `models/candidates/gen_0011/best_value_net.pt`

Those references are useful for historical reasoning but are not guaranteed to exist on this machine now.

## 2) Currently Verified State on This Machine

Verification timestamp: 2026-02-19

### Data

- Raw generation directories exist up through `data/raw/nn_gen24`
- `data/raw/human_games` exists with `25` JSONL files
- Processed dataset currently exists:
  - `data/processed/positions.npy` shape: `(86990, 8, 8, 15)`
  - `splits.npz`: `train=69592`, `val=8699`, `test=8699`

### Models

- Present: `models/best_value_net.pt`
- SHA256: `D54E3117A7A28BD4C0C678FE2D3FAB231AA8943660CFBC1FF6958E9B70BC815B`

### Absent (at verification time)

- `models/archive/manifest.jsonl`
- `models/candidates/`
- `models/run_large_20260218_221423.log`
- `models/run_large_20260218_221423.err.log`

## 3) Recommended Resume Procedure

1. Rebuild a fresh baseline artifact before resuming long runs:

```bash
py -3 src/baseline_snapshot.py
```

2. Run iteration with explicit flags (example):

```bash
py -3 src/iterate.py --iterations 2 --games 180 --curriculum-games 220 --black-focus-games 260 --simulations 120 --curriculum-simulations 50 --black-focus-simulations 100 --epochs 12 --warmup-epochs 3 --warmup-start-factor 0.1 --keep-generations 3 --alternating --opponent-sims 140 --pool-size 6 --arena-games 80 --arena-sims 80 --arena-workers 4 --gate-threshold 0.54 --gate-min-other-side 0.42 --seed 20260219 --human-eval
```

3. After each run, preserve:

- `models/iterate_run_*.json`
- `models/best_value_net.pt`
- any generated `models/archive/` and `models/candidates/` content
- newly generated raw data directories in `data/raw/`

## 4) Quick Verification Commands

Use these before relying on any transfer note:

```bash
# highest normal generation
powershell -Command "$gens = Get-ChildItem data/raw -Directory | ? { $_.Name -match '^nn_gen(\d+)$' } | % { [int]([regex]::Match($_.Name,'^nn_gen(\d+)$').Groups[1].Value) }; if($gens){ ($gens | Measure-Object -Maximum).Maximum }"

# model hash
powershell -Command "(Get-FileHash models/best_value_net.pt -Algorithm SHA256).Hash"

# processed split sizes
py -3 -c "import numpy as np; s=np.load('data/processed/splits.npz'); print(len(s['train']), len(s['val']), len(s['test']))"
```

## 5) Known Ongoing Risk

- Side asymmetry remains the core challenge: candidate models often improve White-side play more than Black-side play, so side-aware gating may reject otherwise strong overall candidates.