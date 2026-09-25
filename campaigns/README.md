# Completed campaign drivers (frozen)

These root-level drivers ran the September 15–17 campaigns. They were moved
here on September 25, 2026 to clear the repository root. **They are records,
not runnable in place:** each computes `ROOT = Path(__file__).parent` and pins
its plan/recipe files and all of `tools/*.py` and `tests/*.py` by hash, so a
move or any later source change fails their identity check by design.

To inspect or exactly resume one, use the snapshot where they last ran:

```powershell
git worktree add ..\monster-chess-snapshot b46ce1c
```

| Directory | Campaign | Plan and results |
|---|---|---|
| `gen50/` | Gen50 deep-target generation (Sept 16) | `docs/experiments/gen50/` |
| `gen50_recovery/` | Gen50 checkpoint recovery and policy/value crossover (Sept 16–17) | `docs/experiments/gen50/` |
| `mainline_extended/` | Counterplay study resume and extension (Sept 15) | `docs/experiments/mainline_counterplay/` |
| `value_calibration/` | Frozen-policy value-head calibration (Sept 17), source of v28 | `docs/experiments/value_calibration/` |

Evidence for each lives under `benchmarks/` as before. `value_calibration.py`
holds the reusable frozen-feature value-head fitting code; import it from a new
driver instead of editing it here.
