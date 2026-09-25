# B2 optimization evidence — September8

Full three-arm architecture plan retained in B2_EXPERIMENT_PLAN.md. These
experiments precede architecture/data changes, as requested by owner.

## Measurements completed

16 current-generation source games, early/middle/late snapshots at700/3200,
eight workers,96 decisions. Baseline fixed-state wall13.96s. Evaluator-owned
graph caching12.07s, identical actions, values and policies. Capture count368
->119. See benchmarks/b2_profile_baseline_8.json and b2_profile_graphcache_8.json.
This test overrepresents engine restarts; it is not a campaign-speed forecast.

Whole-game ABBA (32 identical-seed games per run, four runs at700, eight workers)
confirmed exact complete-record hashes across all128 games. Mean wall baseline
75.596s, cached74.904s, speedup1.0092x. **Insufficient measured benefit: graph
reuse remains opt-in/off by default.** See benchmarks/b2_graphcache_games_700.json.
Match workers already keep engines alive, so graph caching cannot be credited
with a steady-state gate improvement.

CUDA warm-bridge trace: benchmarks/b2_bridge_cuda.json/.trace.json. Single-worker
unprofiled callback means: batch1 .491ms,4 .530ms,8 .546ms,16 .651ms. Trace confirms
nontrivial launch/transfer overhead. CPU `to`/`cpu` timings include synchronization
and GPU work; don't label them all avoidable copying or add overlapping times.

Isolated eight-worker transfer ABBA,80 callbacks per mode per worker per pass:
reference mean .31028s; pinned input .26654s; combined output .30843s;
both .27030s. Every output byte matched. See benchmarks/b2_bridge_variants_8.json.
Pinned inputs reduce this microbenchmark's wall time14.1%; packing is not useful.
No full-game saving is inferred from that result.

## Current candidates and validation

Production adapter exposes experimental switches, both initially off:
- MONSTER_CUDA_GRAPH_CACHE=1: evaluator-owned exact-shape capture reuse; bounded
  signature cache and locks through synchronized output copies.
- MONSTER_PINNED_INPUT=1: reusable pinned float32 input and device buffers; same
  float32 H2D -> GPU half conversion as reference; no padding or changed arithmetic.

`b2_pinned_games` completed the same128-game ABBA using the pinned switch.
Exact complete-record parity passed. Mean baseline73.982s versus pinned66.537s:
10.1% less elapsed time / 1.1119x throughput. This is a measured700-sim generation
benefit, not yet a3200-sim gate or entire-campaign forecast.
Report benchmarks/b2_pinned_games_700.json. Follow-up fixed-state700/3200 parity
also passed: all96 actions, policy dictionaries and values exactly matched the
baseline. Pinned wall11.880s versus13.956s; see b2_profile_pinned_8.json. Single-worker
profiling and full tests are sequential jobs b2_profile_single -> b2_suite.
Production default remains off until resident-engine/deeper throughput validation.
No model weights, simulations, policy temperatures, tree reuse, precision, rules,
generation outcomes or gate thresholds changed.28 targeted tests passed.

Next: inspect pinned whole-game results, verify representative3200-sim decisions,
measure worker scaling and resident-engine throughput, run the full suite with
the worker lease free. Only enable a default after end-to-end benefit and parity
are supported. No broad Rust rewrite is justified by the evidence so far.

## Remaining experiments launched

Owner requested experiments begin. b2_pinned_games_3200 runs32 games x4 ABBA,
eight workers,3200 sims, with exact full training-record parity required.
b2_pinned_resident_3200 is queued behind it and explicitly requires a passing
report. It runs32 games x4 through the actual tools/match.py persistent-engine
driver: gen47 versus v27,3200 sims,16 sampled opening plies, fixed seeds across
passes, exact per-game record comparison. Outputs benchmarks/b2_pinned_games_3200.json
and benchmarks/b2_pinned_resident_3200/summary.json. No automatic default switch.
Single-worker profiling and full suite already completed:810 tests passed.
These are the active optimization experiments; architecture arms are not yet
implemented/training and are not silently queued by these two run entries.
