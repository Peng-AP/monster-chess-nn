# CPU search cost and scaling — completed September 12, 2026

The plan → implementation → waiting cycle is complete. The managed campaign
finished at 22:07 Eastern after 7 hours 2 minutes, with **160 real games** and a
successful replay audit. Its separate end-to-end rehearsal covered 14 games.
Nothing was promoted, retrained, deleted, committed or pushed. No jobs remain
queued. Release v27/gen46 and the GPU gen47 reference are unchanged.

## Clear conclusion

We obtained a measured implementation improvement, but not an established
playing-strength improvement. Incremental evaluation saved another 6.04% of
fixed-depth search time beyond the previous optimizations, yet tied the baseline
in timed games. Giving the baseline four times as much search time produced a
small, inconclusive gain concentrated in Black. It did not improve White outcomes.

Do not interpret this as proof that deeper search cannot help. The sample is
small, and four times the time bought less than one additional completed turn
at the shared roots. Nevertheless, it does not justify another throughput-only
campaign or replacing the GPU reference on strength grounds.

## What was implemented

- Optional, disjoint native cost counters; disabled by default.
- Optional incremental first-layer evaluation for the absolute840 evaluator.
  It reuses the last evaluated position's raw sums across arbitrary DFS jumps,
  accounting for pieces, captures/promotions, phase, castling rights, raw en
  passant and remaining-turn budget. It rebuilds when rebuilding is cheaper
  or after 32 updates. Relative6240 evaluators retain the direct fallback.
- A direct-versus-incremental sequence diagnostic, refresh/update counters,
  numerical tests and full-search speed validation. Direct evaluation remains
  the default and reference; the optional path is not claimed bit-identical.
- Separate candidate/opponent clocks and CPU node ceilings in the match tool.
  Both CPU clocks used a 100-million-node ceiling; actual timing, node-ceiling
  interruptions and depth-ceiling completions are logged.
- A source/model-pinned campaign with success-receipt checks, a complete tiny
  rehearsal, single-worker stages, heartbeats, stage timeouts, conditional
  extension and final replay auditing.

Primary files: `native/src/alphabeta.rs`, `native/src/cheap_value.rs`,
`native/src/search_profile.rs`, `src/cpu_search_engine.py`,
`tools/profile_search_costs.py`, `tools/validate_cpu_scaling.py`,
`tools/search_cpu_scaling_campaign.py`, `tools/search_cpu_gpu_match.py`,
`tools/analyze_cpu_gpu.py`, and `tests/test_cpu_scaling.py`.

## Profiling and validation

Costs were measured on 18 replayed roots, six per phase, retaining the original
match repetition history. At 250k nodes per search, evaluation accounted for
61.98% of instrumented time; move generation was 6.13%. The timers themselves
added 5.36%, so speed nomination used uninstrumented searches.

Three interleaved fixed-depth repeats per root gave these sums of per-root median
times. They are workload aggregates, not single-search latencies:

| Configuration | Seconds |
| --- | ---: |
| Baseline | 9.5752 |
| PVS + fresh TT + 131k entries | 8.2855 |
| Same + incremental evaluation | 7.7853 |
| Incremental evaluation alone | 8.9808 |

The nominated incremental configuration saved 6.04% elapsed time versus the
existing optimized configuration, or 18.69% versus baseline. Those are efficiency
results, not strength measurements.

Saved default fixed-node results remained unchanged. On 11,126 evaluations,
maximum direct/incremental error was 1.252e-6; on the 18 fixed-depth roots, maximum
root-value difference was 1.193e-7 and no best moves changed. The declared
acceptance tolerance was 2e-5. Relative-model fallback and special-state tests
also passed. There were **26 Rust tests and 893 Python tests plus 3 subtests**
passing; the full Python suite passed again in both rehearsal and real campaign.

## Timed development: same 32 games per configuration

CPU and GPU gen47 each received 2 seconds per half-move. All three configurations
used the same 16 paired starts, book indices 336–351, and the same CPU leaf-epoch3
weights. This closes the previous pure-CPU 2-second comparison gap.

| CPU configuration | W/D/L | Overall score | As White | As Black |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 12/3/17 | 42.19% | 9.38% | 75.00% |
| Existing optimized | 11/3/18 | 39.06% | 9.38% | 68.75% |
| Optimized + incremental | 12/3/17 | 42.19% | 9.38% | 75.00% |

Paired-start bootstrap change versus baseline:

- Existing optimized: −3.125 percentage points, 95% interval [−10.9375, +3.125].
- Incremental: 0 points, interval [−7.8125, +6.25].

Neither is an established strength gain or regression. The predeclared tie rule
selected the simpler baseline for the fresh scaling comparison. These games had
a strong Black-win skew; cross-book raw color scores are not controlled evidence
of improvement or regression.

## Does four times the CPU clock help?

Fresh book indices 352–367, 32 games per clock, baseline CPU in both arms.
The GPU gen47 opponent stayed at **2 seconds**. This is deliberately an
unequal-clock scaling diagnostic, not a fair-clock promotion gate.

| CPU clock | W/D/L | Overall score | As White | As Black |
| --- | ---: | ---: | ---: | ---: |
| 2 seconds | 13/1/18 | 42.19% | 28.13% | 56.25% |
| 8 seconds | 14/2/16 | 46.88% | 28.13% | 65.63% |

Paired overall gain: **4.6875 percentage points**, 95% bootstrap interval
**[−3.125, +14.0625]**. Black gained 9.375 points; White's result was unchanged
on every one of its 16 games. For Black, two game outcomes improved, one worsened,
and thirteen were unchanged. Thus the gain is encouraging but inconclusive.

The predeclared extension required at least +10 points overall with no negative
Black delta. It did not trigger; indices 368–383 were not used for real games.

At the first shared-history CPU decision in each of the 32 paired games, average
completed depth rose from **6.156 to 6.625 player turns**; four selected moves
changed. Across all 210 shared-prefix CPU roots, depth rose from 5.686 to 6.010,
with 13 move changes. Later roots follow different play and their averages are
not a clean same-position comparison. These depths count completed player turns,
not individual piece moves or conventional chess-engine nominal depth.

No CPU node-limit hits or depth-ceiling completions occurred in either scaling
arm. Consequently neither ceiling explains the modest scaling result. Three
Black decisions reached the 12-turn ceiling in the earlier development baseline;
that is separately logged and not part of the fresh scaling sample.

## Integrity and operation

All 160 real games replayed successfully, with no proof contradictions. Outcomes:
33 White king-capture wins, 115 Black king-capture wins, 11 repetition draws and
one turn-cap draw. A turn-cap draw is an operational outcome, not a solved draw.
The rehearsal's 14 games were separately audited and are not strength evidence.

Actual CPU medians were approximately 2.007 and 8.007 seconds. GPU medians were
roughly 2.08–2.11 seconds because its clock is checked between complete batches;
all recurring timing is logged. One resident match worker was used throughout.
Peak PyTorch-allocated CUDA memory was 167,413,760 bytes, below the 12 GiB ceiling
(this is allocator telemetry, not a claim about total system GPU memory usage).

Frozen runtime SHA256:
`7bc21e9abc7c3207bd468ad7529b9cabc00e6a090dbca6b320937c1f08ed495b`.
All stage source/model hashes and game manifests are preserved.

## Recommended next block — not launched

1. **Diagnose move-ranking errors.** Use these games and existing human lines as
   evaluation diagnostics: where did both clocks select a bad plan, where did
   deeper search change it, and was the failure tactical, evaluative or a draw /
   conversion issue? Do not relabel every long game a fortress or build arbitrary
   pawn-specific rules. Keep benchmark and human positions out of training.
2. **Create stronger targets on training-only roots and actual CPU leaves.**
   Compare the cheap value with GPU search-backed values and alternative-move
   rankings, not just the GPU network's raw output. Preserve complete phase,
   repetition and turn-budget state and existing split isolation. The teacher
   remains imperfect; disagreement alone is not proof that the CPU is wrong.
3. **Run one controlled evaluator-training comparison.** Keep architecture,
   dataset volume, compute and checkpoint-selection procedure fixed while
   comparing the current raw-value recipe with search-backed / ranking targets.
   Include balanced White coverage and explicit Black conversion reporting.
   Preselect a small number of checkpoints and play-test them; do not gate solely
   on validation MSE or play a full match after every epoch.
4. **Retest at the intended CPU deployment clock.** Use fresh, common, paired
   starts against gen47 and another independent reference, followed by common
   selfplay for any credible candidate. Require a repeatable both-color result
   before promotion. Keep the 8-second result as a scaling diagnostic.

Keep the faster evaluator available as an opt-in experiment, but do not make
further major search rewrites, new architecture families or quantization projects
the default next move without evidence that they address the observed failures.
The CPU engine can still be useful for standalone deployment; these results do
not establish that it has surpassed the GPU engine.

## Evidence

All under `benchmarks/search_cpu_scaling_20260912/`:

- `costs.json`, `baseline.json`, `instrumented_default.json`, `validation.json`.
- `rehearsal/summary.json`, `rehearsal/replay_audit.json`.
- `campaign/summary.json`, `campaign/decision.json`.
- `campaign/scaling_difference_initial.json`, `campaign/replay_audit.json`.
- Each stage's `games.jsonl`, `manifest.json`, `analysis.json`, `hardware.json`
  and `complete.json`.

The plan is `SEARCH_CPU_SCALING_PLAN.md`; operational history is in `HANDOFF.md`.
