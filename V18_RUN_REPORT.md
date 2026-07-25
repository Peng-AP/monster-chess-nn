# v18 candidate run — report

Started 2026-07-25T04:08:58, finished 2026-07-25T17:23:03.
Corpus `data/raw/combined_v17` -> `data/processed/combined_v17_r50h60` (ramp labels: floor 0.5, horizon 60, 15ch).

Bar: beat BOTH v17 and ramp head-to-head, per-side floor 0.4 on every leg, no anchor-Black regression. These were fixed before the run and are not tuned to the result.

## Arms

| arm | vs v17 (W/B) | vs ramp (W/B) | anchor | verdict |
|---|---|---|---|---|
| `gap` | 0.6 (0.7/0.5) | 0.6 (0.9/0.3) | 0.8 (B 0.6) | **FAIL** |
| `spatial` | 0.7 (0.8/0.6) | 0.6 (0.9/0.3) | 0.65 (B 0.3) | **FAIL** |

## Failures

- `gap`: vs_ramp black leg 0.3 < floor 0.4
- `spatial`: vs_ramp black leg 0.3 < floor 0.4

## Stage timings

| stage | exit | minutes |
|---|---|---|
| pretrain_check | 0 | 0.1 |
| process | 0 | 0.4 |
| train:gap | 0 | 265.1 |
| train:spatial | 0 | 264.2 |
| gap:vs_v17 | 0 | 48.7 |
| gap:vs_ramp | 0 | 33.6 |
| gap:anchor | 0 | 25.5 |
| gap:model_diff | 0 | 0.2 |
| spatial:vs_v17 | 0 | 51.3 |
| spatial:vs_ramp | 0 | 60.0 |
| spatial:anchor | 0 | 44.8 |
| spatial:model_diff | 0 | 0.3 |

## Next step

A PASS is automated evidence only. Promotion still requires the owner gate: his playtest decides, and no automated scorecard substitutes for it.
