# Documentation index

The repository root keeps only `README.md` (usage), `CONTEXT.md` (durable
rules, laws and hazards), `HANDOFF.md` (current state) and `requirements.txt`.
Everything else lives here. Evidence stays in `benchmarks/`; finished root
campaign drivers are frozen in `campaigns/`.

## Plans

| Document | Status |
|---|---|
| [plans/GEN54_PLAN.md](plans/GEN54_PLAN.md) | Done October 7: gen53 teacher with Arm R/LR extra self-play, hole-scan pool, one arm |
| [plans/GEN53_PLAN.md](plans/GEN53_PLAN.md) | Done October 5 (revision section: teacher by selection rule) |
| [plans/TEACHER_SELECTION_PLAN.md](plans/TEACHER_SELECTION_PLAN.md) | Pre-declared teacher-selection rule (used for gen53) |
| [plans/OVERNIGHT_20261002_PLAN.md](plans/OVERNIGHT_20261002_PLAN.md) | Done October 3 (top-group round robin, Arm LR) |
| [plans/GEN52_RAMP_PLAN.md](plans/GEN52_RAMP_PLAN.md), [GEN52_LARGE_PLAN.md](plans/GEN52_LARGE_PLAN.md), [GEN52_POOLCAP_PLAN.md](plans/GEN52_POOLCAP_PLAN.md), [GEN52_PLAN.md](plans/GEN52_PLAN.md) | Done September 27 – October 2 |
| [plans/GEN51_STRENGTH_PLAN.md](plans/GEN51_STRENGTH_PLAN.md) | Done September 27 (gen51 became v29) |
| [plans/SHIPPABLE_ENGINE_PLAN.md](plans/SHIPPABLE_ENGINE_PLAN.md) | Deferred; the owner prefers hosting on this box behind a move API |
| `plans/gen5x_teacher_decision.json` | Machine-read teacher/pool decisions for gen52–gen54 |

## Protocols

| Document | What it defines |
|---|---|
| [protocols/PROMOTION_RULE.md](protocols/PROMOTION_RULE.md) | **Current promotion rule:** gate v4 vs the release, held-out non-regression, owner sign-off |
| [protocols/SAMPLED_GATE_PROTOCOL.md](protocols/SAMPLED_GATE_PROTOCOL.md) | Sampled normal-start gate (v3; gate v4 adds the 12,800 guard, `tools/gate_depth.py`) |
| [protocols/FREE_GATE_PROTOCOL.md](protocols/FREE_GATE_PROTOCOL.md) | Older endpoint-uniform free-play gate (v2) |

## Experiments, newest first

| Folder | Dates | Outcome |
|---|---|---|
| [experiments/gen54/](experiments/gen54/) | Oct 5–7 | gen54: passes gate v4 vs gen53 (79%), fixes the v27 hole, held-out 92.2%; **fails vs v29 on the White floor**; not eligible |
| [experiments/gen53/](experiments/gen53/) | Oct 4–5 | gen53: +105 Elo vs v29, passes gate v4 vs v29; not eligible (v27 endgame hole); hole scan |
| [experiments/gen53_prep/](experiments/gen53_prep/) | Oct 3–4 | Teacher selection: gen52 Arm R |
| [experiments/gen52/](experiments/gen52/) | Sep 27 – Oct 3 | Arms A/B (pool), C (capped deep value), L (wide), R (ramped labels), LR; value audit; top-group round robins |
| [experiments/elo_rr/](experiments/elo_rr/) | Sep 29–30 | Elo round robin (v21 = 1600) and depth ladder; search saturates above 3,200 |
| [experiments/gen51/](experiments/gen51/) | Sep 25–27 | Diversity audit, search constants (null), gen51 arms; **deep-value arm became v29** |
| [experiments/value_calibration/](experiments/value_calibration/) | Sep 17 | Frozen-policy value-head fit; **became v28** |
| [experiments/gen50/](experiments/gen50/) | Sep 16–17 | Gen50 and checkpoint-recovery/crossover study |
| [experiments/mainline_counterplay/](experiments/mainline_counterplay/) | Sep 15 | Conditional e4+d4 ...e5/...d5 study |
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
| [history/HANDOFF_20260925_20261005.md](history/HANDOFF_20260925_20261005.md) | The previous handoff: September 25 state plus every update to October 5 |
| [history/HANDOFF_LOG.md](history/HANDOFF_LOG.md) | Chronological operational log through September 25 |
| [history/CONTEXT_LOG.md](history/CONTEXT_LOG.md) | `CONTEXT.md`'s old status preamble and §2 ledgers (moved October 6) |
| [history/REPORT.md](history/REPORT.md) | August experiment ledger, §§1–53 (code comments cite its sections) |
| [history/DIRECTIVE.md](history/DIRECTIVE.md) | Completed August native-rewrite scope record |
| [history/CLEANUP_20261006.md](history/CLEANUP_20261006.md), [CLEANUP_20260925.md](history/CLEANUP_20260925.md) | Cleanup inventories |

`NEXT_STEPS_HANDOFF_20260905.md` and `BOOTSTRAP_FOLLOWUPS.md` were deleted as
superseded on September 25; both are in commit `b46ce1c`.
