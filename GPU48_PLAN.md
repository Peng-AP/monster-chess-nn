# GPU strength-track return: gen48 research campaign

**Completed September14 00:05 Eastern.** See `GPU48_RESULTS.md`. Both fresh
H2H legs favored gen48 modestly; independent B2 transfer was flat, selfplay
White score unchanged. No broad upgrade established, no promotion or follow-up
launched. The plan below is retained as the pre-run protocol.

September 13, 2026. Owner: "same workflow, go" after agreeing to pause CPU
strength development. Plan -> implement -> rehearse -> run -> completion waits.
Keep CPU code/models intact. No architecture change, deletion, promotion, commit
or push. The public release remains v27/gen46. Gen47 is the frozen strength bar.

## Question

Can stronger current-teacher data and broader independent policy supervision
improve the existing GPU engine on both colors, including transfer to an
opponent absent from the new training recipe? This is one data-recipe experiment,
not an architecture search or proof of perfect play.

This bundled data-recipe experiment can test the resulting model, not isolate
the causal contribution of search depth versus coverage versus mixture size.

Teacher / baseline: `bootstrap_main_gen_0047/arena_selected.pt` (epoch17).
Holdout opponent: `b2_seed9053_state_cnn/selected_epoch_008.pt`. B2 is excluded
from generation opponents, prefix teachers and reanalysis. Its evaluation
positions and all human/benchmark games remain outside training.

## Data recipe and cost pilot

Nominal 4,800 new completed games:

| Component | Games | Simulations |
| --- | ---: | ---: |
| Free gen47 selfplay |2,400|1,600|
| Fresh stochastic prefixes, then gen47 continuation |1,200|1,600|
| Gen47 versus v25/v26/v27, equal teacher colors |600|1,600 both players|
| Completed continuations from distinct new parent families |600|6,400|

Prefix teachers: gen44, v27, gen47. Existing stochastic eight-half-move prefixes;
prefix rows are not policy targets. Fork roots retain the current uniform
sampling rule, 60% settled Black / 40% settled White. One fork per parent,
full state and history, all outcomes retained. Opponent policy rows remain
masked; no hand-written tactical motifs, filtering for teacher wins, or solver
timeouts relabeled as draws.

First run an isolated 64-game full-budget cost pilot. If its total elapsed time
projected to 4,800 games exceeds four hours, use the predeclared smaller size:
1,600 free +800 fresh +400 league +400 forks =3,200. Keep search depths and
proportions unchanged. Pilot startup costs make this a conservative sizing rule.
Publish the final size before production; never resize a run mid-flight. Pilot
games are not silently imported into training.

Sizing decision12:29 Eastern: pilot64/64, zero failures,358.05seconds including
startup. Nominal4,800 projection7.46hours, so production is **3,200 games**.
The same crude projection is4.97hours for generation alone; startup amortization
should reduce that, but the final duration remains an estimate, not a deadline.

New policy reanalysis: 24,000 sampled positions /12,000 retained at6,400 sims,
60% Black. Add a deterministic source-family coverage cap of16 sampled roots
per transitive original game family. Preserve full-state replay and existing
policy-only teacher labels: actual completed outcomes remain the value targets.
Report independent-family counts, side/phase counts and cap exhaustion. Refuse
an undersized sample rather than silently repeat positions. Same depth on every
new reanalysis root; no CPU mixed-horizon value targets are used.

## Training

Reuse the canonical stateful iteration. Same gen47 GPU architecture, scratch
initialization seed3173, Adam settings, LR0.002, batch256, EMA0.999, warmup3,
maximum30 replay epochs / patience10, scalar value ramp0.5/60, policy multiplier4,
generation-only replay8, no legacy human anchor. All sources/forks/teachers share
their transitive family split. Preserve old accepted replay and its split IDs.

Save epochs; use the existing bounded representative-checkpoint shortlist and
play-based nomination. Probe40 games, finalists200 games at3,200 simulations,
two requested finalists; existing mandatory offline/Black nominees remain.
No game gate after every epoch and no offline-only rejection. Stop the canonical
iteration after checkpoint selection: this research block does NOT bypass or
rewrite the official binding gate or mark a model accepted.

## Fixed play-testing after nomination, regardless of screen score

Generate a fresh frozen 320-start book from gen47/v27/gen44 at700 sims,
temperature0.8,16 half-moves. Use new RNG namespaces. Candidate never generates
its test book. Reuse starts only where pairing is intentional:

1. Candidate vsgen47:256 paired games at3,200 sims, starts0–127.
2. Fresh confirmation vsgen47:256 games, starts128–255. Always run, even if the
   first leg looks weak; do not choose another epoch based on confirmation.
3. Candidate AND gen47 each play128 games against B2 on common starts256–319.
4. Candidate AND gen47 each play64 self-games on common starts256–319. One
   game per start, not duplicated swapped self-games. These selfplay/transfer
   subsets intentionally overlap and must not be pooled as independent trials.

Total896 post-selection games. Report integer W/D/L and each color; uncertainty
resamples complete opening pairs. Report both confirmation legs separately,
B2 paired deltas and self-skew. Same architectures and fixed simulation budgets,
NOT a claim of strict equal wall time. Record actual timings and runtime hashes.
No model promotion: a credible both-color gain would justify the unchanged full
binding gate and owner playtest next. An inconclusive/negative result is retained,
not followed automatically by another architecture or parameter sweep.

Implementation: `tools/start_gpu48.py`; production recipe
`tools/recipes/gen48.json`; research results under `benchmarks/gpu48_20260913`.
Canonical state intentionally ends `partial` after `checkpoint_screen`, while
the separate research summary may be `complete`. These mean different things.
Every post-selection game is replay-audited for moves, phase, clocks and outcome.
Book evaluation starts with fresh driver history, as in the existing harness;
this differs from the full source history restored in training/reanalysis.
Fixed common-selfplay reporting to count +/-0.5 cap labels as draws, matching
the capture-only match rules. Historical artifacts are retained unchanged.

Rehearsal completed12:42 Eastern:931Python tests+3subtests;28generated games,
one training epoch and checkpoint nomination, all20post-selection games replay
audited. Production launched12:43 as managed `gpu48_campaign` (PID2120).

## Operation and safety

At most8 workers, one GPU game/training job at a time, <=12GiB target allocation.
Opt into the already validated pinned-input path; no new search optimizations.
Full tests and a real tiny end-to-end rehearsal must pass before production,
including coverage sampling, family processing, training, nomination and every
post-selection match branch. Pin source/model/recipe inputs; require successful
stage receipts, preserve failed outputs, and resume only compatible completed work.
Use process completion waits, not live status-file watchers or per-game probes.

Historical gen47 spent ~3.5h generating,2.4h reanalyzing,1.8h training and several
hours selecting/testing. This is therefore plausibly a 9–14h block after pilot
sizing, not a promise of a quick CPU-style fine-tune. Finish coherent prescribed
stages rather than changing the test counts to hit an arbitrary time.
