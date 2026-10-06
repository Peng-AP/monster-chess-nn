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

## `archive_tools/`: retired one-off tools (October 6, 2026)

Moved from `tools/` (and three tests from `tests/`) during the October 6 cleanup
(`docs/history/CLEANUP_20261006.md`). Each was reachable from nothing live:
no current driver, the site, the match/gate/Elo tools or `runs.py` imports or
launches it. No document mentions it, and it was written for one finished
experiment. They resolve `ROOT` from their old location, so they are records,
not runnable in place. Run one from a worktree at `d784471` (the last commit
before the move) or move it back.

| Area | Files |
|---|---|
| August LC0/E-series drivers and probes | `lc0_experiment_driver`, `post_e5_driver`, `curve_runner`, `disagreement_cost`, `gen7_driver`, `inference_server_bench`, `inference_server_smoke`, `promotion_policy_metrics`, `search_agreement_gate`, `moves_left_probe`, `book_depth_sweep`, `book_rescore` |
| Free-gate campaign (Sept 5–6) | `analyze_free_campaign`, `evaluate_free_campaign` |
| B2 bridge and teacher work (Sept 8–10) | `b2_screen_teacher`, `benchmark_bridge_variants`, `profile_bridge_cuda`, `validate_graph_cache`, `validate_resident_input` |
| Search-first / CPU profiling (Sept 11–12) | `profile_cheap_value`, `profile_cpu_workload`, `profile_search_first`, `search_first_search_sweep` |
| One-off chains and records | `overnight_chain_20261002`, `promote_september_releases` |
| Their tests | `test_free_campaign_audit`, `test_lc0_experiment_driver`, `test_post_e5_driver` |

