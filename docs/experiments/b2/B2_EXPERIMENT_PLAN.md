# Bootstrap-2: full architecture comparison, optimization first

Owner September8: retain the full plan and all arms; profile production work and
optimize measured bottlenecks before the architecture campaign. No automatic
release promotion, no reduced-arm substitution.

## Order and acceptance criteria

1. Profile gen47's current native/PyTorch workload at700/3200 sims, representative
   early/middle/late states and White half-moves, at1 and8 workers. Separate model
   load, state restoration, engine creation, graph capture, inference callbacks,
   and remaining search wall time. Save settings/model/runtime hashes, batch-size
   histograms and profiler output. Do not infer GPU arithmetic time from CPU timers.
2. Optimize only measured hotspots. No reduced sims, changed precision, padding,
   search rules, policy temperatures, move selection or early-stop thresholds.
   Compare fixed-state policy/value/actions, full trajectories, representative
   throughput and memory before/after. Keep a reference/disable route. Benchmark
   under the same concurrency; reject faster code that changes required semantics.
3. Broadly re-screen existing gen47 shortlisted checkpoints; freeze the reference
   and disjoint selection/confirmation opening seeds and opponent panels.
4. Implement complete basic state inputs consistently across processing, Python
   inference and native leaf encoding: pieces/turn/White phase, coordinates,
   castling, en passant, remaining turn budget. Repetition/history require separate
   validated native-path propagation; omit explicitly rather than fabricate them.
5. Generate/audit one shared state-complete corpus. Proposed12,000 completed games:
   6000 free,3000 mixed-opponent,1500 fresh-prefix,1500 deep-fork continuations.
   Equal teacher contributions from v27 and broadly selected gen47; equal teacher
   colors in league; diverse opponent roster; all outcomes retained. Ordinary700,
   forks3200 sims;60%Black fork roots; one fork/source family; mix uniform and
   disagreement sampling. Reanalyze80k/retain40k at3200. Audit reuse of existing
   stateful data before generating redundant games. Do not fake missing old state.
6. Train ALL THREE arms on identical data/splits, from scratch:
   A=current15-channel CNN control;
   B=same CNN with basic expanded state;
   C=same expanded state with two64-square attention blocks (similar size budget).
   Keep policy ABI, scalar value shaping, optimizer, EMA, seed3173, max30/patience10
   initially. Benchmark hybrid inference cost before full training. No simultaneous
   WDL/MLH/search-utility change and no full-transformer replacement in this round.
7. Save epochs but screen a fixed four-checkpoint shortlist spanning training and
   best validation, duplicates removed. Screen against multiple opponents and both
   free/matched starts; at most two finalists per arm, no implicit expansion.
8. Repeat winning arm and control with a second seed, then untouched multi-opponent
   confirmation at equal sims AND equal thinking time. Preregister counts and
   per-color regression margins before results. Direct parent wins are supporting
   evidence, not the sole criterion. Self-skew is diagnostic, not a balance target.
9. Owner playtest; only promote a broadly supported improvement as next public v28.
   Candidate names b2_001_control, b2_001_state_cnn, b2_001_hybrid; version ladder
   does not reset. No promotion for a name/architecture change alone.

## Estimated cost before profiling

Profile/optimization implementation depends on measurements. Baseline campaign:
state encoding6–12h implementation; hybrid6–12h; generation7–11h; reanalysis2.5–4h;
processing20–60min; three trainings7–12h; checkpoint screens4–8h; second-seed
comparison6–10h; final testing4–7h. Machine work roughly2–4days plus engineering.
These are estimates, not promises. Record measured savings before revising them.

Constraints: at most8 workers and one game job at a time; target <=12GiB GPU use;
preserve historical evidence and all current user edits; never stage play.ipynb
or NEXT_STEPS_HANDOFF_20260905.md; no push or automatic promotion.

## Engineering gates before spending the training budget

- State encoding must agree between replayed training records, Python inference
  and native search leaves, including White's second half-move. Test castling,
  en passant expiry, turn-limit boundaries and transformed policies explicitly.
  Existing horizontal tensor mirroring is NOT automatically valid for new file
  coordinates or castling channels: recompute/swap semantic features and check
  legality, or disable that augmentation consistently across all three arms.
- Preserve the original15-channel checkpoint loader and encoding byte-for-byte.
  Store an explicit encoding/architecture version in each new checkpoint; reject
  mismatches rather than guessing from a tensor that happens to have the same size.
- Hybrid requires forward/backward, checkpoint round-trip, EMA, native bridge and
  graph-capture tests. Measure parameter count, peak memory and batch1..16 latency.
  Equal-simulation and equal-time results must remain separate: a slower model
  may improve each simulation without improving the downloadable engine.
- Split by complete source family BEFORE augmentation, replay mixing or forks.
  Audit teacher/opponent/color/result counts and missing state. Preserve separate
  selection and confirmation seeds; don't mine the confirmation losses into this
  campaign's training or repeatedly choose new checkpoints on confirmation data.
- Campaign stages write immutable input/output hashes and completion receipts.
  A failed prerequisite must stop the chain, not silently launch the next arm on
  partial data. Rehearse all three training/load/search paths on tiny isolated data
  before production. Resume completed stages instead of regenerating them.

## Evaluation budget to freeze before selection starts

Proposed fixed budget (final seed manifest must be written before any games):

- Existing gen47 re-screen and each new-arm checkpoint screen:200 games/checkpoint,
 50 each against v24/v25/v26/v27, balanced candidate colors; half free starts,
 half matched fresh starts. At most four distinct checkpoints per arm.
- At most two finalists/arm:400 games each, same four-opponent balance. Selection
 uses aggregate AND color scores; report opponent cells rather than pooling away
 regressions. Retain a control finalist even if its screen ranks below both others.
- Final winning-arm versus control and reference comparison:800 games/model at
 equal3200 sims and800/model at a calibrated equal-time budget, balanced colors
 and opponents on common, untouched start sets. Include direct arm-versus-control
 games as a separately reported panel, not a replacement for the opponent panel.
- Report paired-opening bootstrap intervals and per-color differences. Require
 positive aggregate improvement with95% interval above zero and lower95% bounds
 on each color above a preregistered-3percentage-point regression margin. A
 candidate that fails is an experiment result, not permission to widen margins.
-200 self-games per finalist at3200 for skew/termination diagnostics. Second-seed
 control and winning arm must support the direction of the result; a seed reversal
 means uncertainty, not a release. Owner playtest remains the last promotion gate.

These game budgets make broad testing a substantial cost. They must be reflected
in the measured campaign ETA; no promise that every arm and seed fits one night.
