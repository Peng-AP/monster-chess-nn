# CPU search cost and scaling experiment — September12 afternoon

**Completed22:07 Eastern.** All160 real games and14 rehearsal games passed their
replay audits. No strength promotion. See `SEARCH_CPU_SCALING_RESULTS.md` for
the final findings and recommended next block. No jobs remain queued.

Owner authorized plan -> implement -> wait. Existing CPU/GPU campaign completed
at10:17 Eastern with successful audits. PVS/cache changes saved15.7% fixed-depth
elapsed time but scored51.6% in a32game CPU duel. Guided confirmation46.875% versus
unchanged CPU47.656% on the same64games showed no established strength gain.

## Objective and controls

Determine which CPU costs deserve implementation work and whether substantially
more CPU search improves play. Use the existing absolute512 leafepoch3 evaluator
and frozen GPU gen47epoch17. Keep White/Black scores separate, with Black retained
as a primary acceptance concern. No training or architecture changes this run.

## Ordered work

1. Snapshot runtime/source and fixed-node reference results. Add disabled-by-default
   search cost instrumentation: static cache, exact tactical scan, neural evaluation,
   move generation, move ordering, clone/apply, searched-bound lookup/store. Compare
   instrumented/uninstrumented fixed-node results and quantify timer overhead. Use
   actual replay positions in all phases, original repetition histories, and fixed
   node budgets; do not infer cost from Python/FEN microbenchmarks alone.
2. Select an implementation from measured costs. If neural evaluation is a major
   cost, prototype lazy incremental first-layer evaluation for absolute840 inputs.
   Reuse previous evaluated-position sums only when feature differences cost less
   than rebuilding; refresh periodically to bound floating-point drift. Include
   phase, rights, rawEP and remaining budget; refresh unsupported representations.
   Retain direct evaluation as reference. If another category dominates, target
   that category instead and document the measured reason before implementation.
3. Check numerical parity on broad random walks and special-state transitions,
   fixed-depth values/actions on representative positions, timer-disabled default
   parity, and full-search speed. Float summation reordering requires a declared
   tolerance and actual games; it is not assumed bit-identical. Reject a slower
   implementation from nomination while preserving its evidence.
4. Add explicit candidate/opponent time and node limits to the match driver.
   Current CPU default10Mnodes can confound an8second search, so use100Mnodes for
   this experiment at BOTH clocks and log node-bound interruptions. Keep the GPU
   opponent at2seconds for the scaling comparison; record actual end-to-end times.
5. Rehearse the full chain at tiny budgets and audit complete games. Then:
   -32games unchanged CPU versus GPU gen47 at2seconds, starts336..351.
   -32games PVS/freshTT131k CPU versus the same opponent/starts at2seconds, closing
     the CPU-only confirmation gap. Compare paired-start scores with uncertainty.
   -If the new implementation passes speed/parity nomination, give it32games on
     those same starts at2seconds. This is its development comparison.
   -Nominate the2second arm by development score (ties favor the simpler control).
     Test that frozen arm at2seconds and8seconds versus fixed GPU2seconds on fresh
     starts352..367,32games each. This deliberately unequal-clock diagnostic asks
     whether extra CPU compute buys strength; it is not a fair-clock promotion gate.
   -If the8second arm gains at least10percentage points without a negative observed
     Black delta, extend the same matched-clock experiment to16new starts368..383.
     The trigger is exploratory continuation, not statistical proof or promotion.
   -Replay all games and write a final result/continuation receipt.

## Operation and interpretation

One heavy job at a time, one resident match worker, <=12GBVRAM. Managed jobs check
source/model hashes and required success receipts between stages. Stage heartbeats
and bounded child waits detect failures without conversational game-by-game polling.
Expect several hours; complete paired stages rather than stopping halfway to fit a
soft duration. Preserve all existing evidence, models and user worktree changes.

Completed depth is measured in player turns, not individual piece moves. Faster
evaluation need not complete an extra depth. Stable best moves need not be correct.
Gen47 search targets and evaluations remain imperfect references, not solved truth.
If8seconds helps substantially, prioritize larger CPU throughput improvements. If
it does not, shift the next work block toward better search-backed/ranking training
targets after examining decision errors. This run does not launch that training.

No release promotion, deletion or push is part of this experiment. The official
release remainsv27; GPUgen47 and unmodified CPUleafepoch3 remain reference engines.

## Implementation checkpoint

The disabled-by-default disjoint cost counters passed saved fixed-node parity.
On18 replay roots (six per phase), two search configurations,250k nodes each:
evaluation61.98%, move generation6.13%, static cache5.25%, bound store5.29%,
ordering4.86%, tactical scan4.07%, bound lookup3.12%, clone/apply1.16%; residual
8.14%. Instrumentation added5.36% elapsed time; use uninstrumented speed trials.
Evidence: `benchmarks/search_cpu_scaling_20260912/costs.json`.

Implemented optional `incremental_eval`: cache raw first-layer sums for the last
evaluated position; apply XOR feature changes including phase/rights/rawEP and
continuous remaining-turn-budget delta; rebuild when deltas cost more than a
refresh or after32 updates. Relative6240 representation falls back to direct
evaluation. Default remains direct. This is numerical-tolerance equivalence,
not bit-identical equivalence. Direct evaluation is retained as the reference.

26 Rust tests and18 focused Python tests passed. The broader validation checks
saved default parity,11k+ direct/incremental values,18 replay roots with three
interleaved fixed-depth repeats across four configurations, and matched2/8second
position probes. An incremental configuration enters games only if it is at
least5% faster than the faster existing control on the fixed-depth workload.

`tools/search_cpu_scaling_campaign.py` rehearses ALL chain branches at tiny
budgets (including the conditional extension), then the real chain requires its
successful, source/model-hash-matched receipt. It checks complete games, proof
contradictions, node-ceiling confounds and replay audits; failures stop the chain.
Only the real campaign's strength scores are meaningful. No per-game agent
polling is needed. All dependencies are frozen once the rehearsal begins.

Validation completed:11,126 evaluations, maximum direct/incremental error
1.252e-6; fixed-depth root error<=1.193e-7, no best-move changes on18 roots.
Aggregate per-root median fixed-depth seconds: baseline9.5752, optimized8.2855,
incremental optimized7.7853, incremental-only8.9808. The nominee saves6.04%
elapsed time versus existing optimized search (1.064x throughput),18.69% versus
baseline. Six matched-position probes completed mean depth6.0 at2s and6.667
at8s for BOTH optimized variants: speed alone still did not buy an extra depth
at these sampled clocks. These are complete player-turn depths, not half-moves.

The full tiny chain passed893 Python tests plus3subtests and14 complete games
through all seven branches, followed by a successful replay audit. Rehearsal
scores are NOT strength evidence. Managed `cpu_scaling_campaign` requires that
source/model-identical rehearsal receipt and runs the real matches automatically.
Validation and rehearsal artifacts: `validation.json`, `rehearsal/summary.json`
and `rehearsal/replay_audit.json` under the experiment output directory.

## Development result (96 games complete)

Same16paired starts336..351, CPU2seconds versusGPUgen47@2seconds:

| CPU arm | W/D/L | Overall | As White | As Black |
| --- | --- | --- | --- | --- |
| Baseline |12/3/17|42.1875%|9.375%|75%|
| PVS/freshTT131k |11/3/18|39.0625%|9.375%|68.75%|
| Same + incremental |12/3/17|42.1875%|9.375%|75%|

Paired change versusbaseline: optimized-3.125pp,95%CI[-10.9375,+3.125]pp;
incremental0pp,CI[-7.8125,+6.25]pp. Faster fixed-depth performance did not
establish stronger timed play. The predeclared tie rule selectsBASELINE for
the fresh2s/8s scaling comparison. These score distributions are highly skewed
toward Black wins in played games; raw color scores cannot be compared across
different opening sets as if they were controlled improvements/regressions.
