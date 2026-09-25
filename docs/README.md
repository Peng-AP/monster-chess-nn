# Documentation index

The repository root keeps only `README.md` (usage), `CONTEXT.md` (durable
rules, laws and hazards) and `HANDOFF.md` (current state and next step).
Everything else lives here. Evidence stays in `benchmarks/`; finished root
campaign drivers are frozen in `campaigns/`.

## Plans (forward-looking)

| Document | Status |
|---|---|
| [plans/GEN51_STRENGTH_PLAN.md](plans/GEN51_STRENGTH_PLAN.md) | **Current.** Proposed September 25; not started |
| [plans/SHIPPABLE_ENGINE_PLAN.md](plans/SHIPPABLE_ENGINE_PLAN.md) | Deferred September 25 (hosted-server route preferred when resumed) |

## Protocols

| Document | What it defines |
|---|---|
| [protocols/SAMPLED_GATE_PROTOCOL.md](protocols/SAMPLED_GATE_PROTOCOL.md) | Current sampled normal-start gate (v3) |
| [protocols/FREE_GATE_PROTOCOL.md](protocols/FREE_GATE_PROTOCOL.md) | Older endpoint-uniform free-play gate (v2) |

## Experiments, newest first

| Folder | Dates | Outcome |
|---|---|---|
| [experiments/value_calibration/](experiments/value_calibration/) | Sep 17 | Frozen-policy value-head fit; 58.4% vs gen50 epoch14 @3,200; **became v28** |
| [experiments/gen50/](experiments/gen50/) | Sep 16–17 | Gen50 (75% vs gen49 @3,200) and checkpoint-recovery/crossover study |
| [experiments/mainline_counterplay/](experiments/mainline_counterplay/) | Sep 15 | Conditional e4+d4 ...e5/...d5 study at 3,200/12,800/51,200 |
| [experiments/gen49/](experiments/gen49/) | Sep 14 | Mainline-only gen49 (94% vs gen48) |
| [experiments/gpu48/](experiments/gpu48/) | Sep 13 | GPU gen48 data recipe |
| [experiments/search_targets/](experiments/search_targets/) | Sep 13 | Search-backed CPU evaluator targets; inconclusive |
| [experiments/search_cpu/](experiments/search_cpu/) | Sep 12 | CPU alpha-beta scaling and CPU/GPU cooperation |
| [experiments/search_first/](experiments/search_first/) | Sep 11 | Cheap CPU evaluator + alpha-beta prototype |
| [experiments/b2/](experiments/b2/) | Sep 8–10 | Architecture arms (state CNN, hybrid) and bridge optimization |
| [experiments/gen47/](experiments/gen47/), [gen46/](experiments/gen46/) | Sep 7 | Stateful mixed-opponent data; v25–v27 releases |

## History

| Document | Contents |
|---|---|
| [history/HANDOFF_LOG.md](history/HANDOFF_LOG.md) | Chronological operational log through September 25 |
| [history/REPORT.md](history/REPORT.md) | August experiment ledger, §§1–53 (code comments cite its sections) |
| [history/DIRECTIVE.md](history/DIRECTIVE.md) | Completed August native-rewrite scope record |
| [history/CLEANUP_20260925.md](history/CLEANUP_20260925.md) | Cache deletion and log archival inventory |

`NEXT_STEPS_HANDOFF_20260905.md` and `BOOTSTRAP_FOLLOWUPS.md` were deleted as
superseded on September 25; both are in commit `b46ce1c`.
