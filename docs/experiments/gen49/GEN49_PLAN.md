# Gen49: mainline learning, normal-start evaluation

September 14, 2026. Owner authorized plan, implement, then long waits.
Research only: no release promotion, deletion, commit, or push.

## Decision and correction

The previous gen48 independent H2H, B2 and self-skew measurements all used
fixed book positions. They did not establish that its large sampled-free
checkpoint-screen gain fails to transfer from the normal starting position.
The standing owner decision is that normal-start free play is primary; books
are secondary continuity/stress evidence. Restore that distinction explicitly.

Observed repertoire concentration supports investigating mainline learning,
not declaring other openings unsound or Monster Chess solved. Gen47's saved
200-game screen calibration had 24 distinct full opening endpoints, one seen
105 times. Historical gen36 vs gen33 scored 49.75% with books but 72.75% free
at the same 3,200 simulations. Instruments answer different questions.

This is one strength experiment, not a causal proof that variety hurts:
teacher advances from gen47 to gen48, new-data composition changes, and the
eight-generation replay window rolls forward. No architecture or optimizer
change and no arbitrary tactical rule are included.

## Frozen sequence

1. Run regression tests and an isolated real end-to-end rehearsal: all data
   types actually used, processing, scratch training, checkpoint selection,
   both sampled-gate legs, normal-start held-out opponents and actual-color
   self-skew. Tiny search/counts are plumbing evidence, not strength evidence.
2. Independently test frozen gen48 epoch17 against gen47 epoch17, normal
   initial position, 3,200 simulations, 8 workers. Fixed research counts:
   200 gen47 calibration games, 200 first H2H, 200 confirmation. Then always
   run 200 each: gen48 vs B2, gen47 vs B2, gen48 self-play. Total 1,200.
   This is the v3 sampled method with smaller explicitly recorded research
   counts, NOT the standard 400/400/400 binding-gate evidence.
3. Generate and train gen49 using the fixed recipe below. Gen48 is the frozen
   experimental teacher, not an automatically promoted champion. First-test
   scores do not select a different teacher, change counts, or suppress tests.
4. The usual bounded saved-epoch screen nominates one checkpoint. At most
   eight representative checkpoints get 40-game probes; aggregate leaders
   plus Black-best/offline-best safeguards get 200-game screens. Probes and
   full screens use 3,200 simulations. No every-epoch games; offline accuracy
   cannot reject the whole generation without play-testing.
5. Independently test the nominee vs gen48 with the standard sampled counts:
   400 incumbent self-play calibration, 400 H2H, 400 confirmation. Then always
   run 200 each vs untouched B2, vs public v27, and candidate self-play.
   Total 1,800 post-selection games, even if the H2H gate fails. Stop after
   the fixed chain, summarize and retain everything; no automatic next gen.

All matches start normally. Models choose openings with the existing first
16 search-half-ply temperature 0.5 sampler, then temperature zero. No book,
unique-endpoint quota, forced novelty, or excluding repeated positions.
Independent seed blocks remain independent draws even if endpoints repeat;
do not mislabel different RNG seeds as different strategic structures.

Production seed reservations: gen48 research gate 2,180,000,000; its three
extra blocks 2,181,000,000 through 2,183,000,000. Gen49 gate 2,184,000,000;
extra blocks 2,185,000,000 through 2,187,000,000. One million per block; gate
par/first/confirmation use the existing 100,000 spacings. Rehearsal uses
2,190,000,000 through 2,197,000,000 with tiny fixed counts. No seed-paired
book confidence calculation is applicable to these model-dependent starts.

## Data and training recipe

| Component | Gen48 | Gen49 |
|---|---:|---:|
| Normal-start teacher self-play | 1,600 | 2,800 |
| Old-model/teacher prefix starts | 800 | 0 |
| Older-opponent league games | 400 | 0 |
| Completed deep continuations | 400 | 400 |
| Total newly completed games | 3,200 | 3,200 |

Ordinary self-play: 1,600 simulations. Fork continuations: 6,400 simulations,
one per selected parent, 60% settled Black / 40% settled White, uniform
eligible point after at least eight primitive moves. All parents now come
from this teacher's own normal-start games. Full histories, White half-turns,
clocks and repetition survive into forks and reanalysis. All outcomes remain.

Existing root noise and training temperatures remain: 1.0 for the first 15
primitive moves, 0.1 thereafter. Removing externally imposed prefixes is NOT
turning exploration off. Hard deterministic self-play would repeat one line;
this experiment changes the new-data sources, not the sampling temperature.

Deep reanalysis: sample 24,000, keep 12,000 at 6,400 simulations; same
family-balanced sampling cap 16, 60% retained Black, disagreement ranking,
policy multiplier four and zero teacher value weight. Family balance stops
one long game dominating; it does not force novel opening structures.
Descendants/teachers share parent train/validation/test splits.

Same gen47/gen48 15-plane attention-policy CNN. Scratch training seed 3173,
AdamW learning rate .002, weight decay .0001, batch 256, EMA .999, warmup 3,
up to 30 epochs/patience 10, value floor .5/horizon 60. Eight-generation replay,
no human/outside anchor. Older replay still contains mixed data: this is a
mainline-only NEW increment, not a wholesale historical-corpus ablation.

## Evidence and safety

Use existing stateful iteration and sampled gate/journal tools. Add only a
bounded driver, recipes, free-trajectory audit and conditional recipe checks
so unused prefix/opponent lists may be empty. Preserve all legacy defaults.
Stage receipts pin commands, models, sources, native binary, configuration,
replay inputs and artifacts. Separate campaign and worker locks; maximum
eight workers and one GPU job at a time, campaign VRAM target <=12 GiB.
Other applications are untouched; contention can lengthen elapsed time.

Verify legal moves, state/half/clock reconstruction, opening/history/repetition
digests, exact task counts/seeds, terminal causes and captures-only scoring in
every final-test journal. Repetition and move-limit results count as draws;
neither is a proof of a fortress. Report W/D/L per color, true actual-color
self-par, frequency-weighted sampled scores, endpoint concentration and
nominal uncertainty. Keep selection results separate from independent tests.

The gate remains an operational point-estimate filter, not statistical proof
of both-color improvement. Primary success requires both H2H legs above 50%
and neither color below incumbent self-par minus five percentage points.
Held-out B2, public v27 and self-skew qualify the conclusion; no single score
establishes perfect-play progress. B2 never enters new data or selection.

Run under tools/runs.py. Game/generation/reanalysis journals resume missing
tasks with the identical source/recipe. Interrupted training is retained and
stops the automatic chain rather than overwriting a checkpoint. Execution
errors stop safely; measured FAIL does not prevent the prescribed other tests.

## Timing and handoff

Planning/implementation/rehearsal roughly 30-60 minutes. Initial normal-start
checks roughly 1-2 hours; generation 3-4, reanalysis about 1-1.5, processing
minutes, training about 3-4, selection/final tests about 2-3. These overlap no
heavy stages and sum to roughly 11-15 hours, not a promised morning finish.
The prior production run actually took 11h22 including 3h15 training; this
chain has more independent normal-start games and no new book generation.

Launcher: `tools/start_gen49.py`. Recipe: `tools/recipes/gen49.json`.
Managed name: `gen49_mainline`. Production iteration: `iterations/gen_0049`.
Research evidence: `benchmarks/gen49_mainline_20260914/production`.
The canonical iteration intentionally stops after checkpoint selection;
external gate evidence must not silently rewrite it as a canonical PASS.
Wait primarily on process completion, with infrequent milestone/health reads,
not continual file watchers or assistant probing. No user prompts scheduled.

Rehearsal note, 02:27: the first rehearsal correctly stopped because the CLI
match recorded `sims_b=null` (inherit A) but the strict audit expected explicit
`sims_b=8`. Actual search budgets agreed; the manifest representations did not.
The driver now passes both budgets explicitly. Preserve `rehearsal/` unchanged;
rerun all branches under `rehearsal_v2/`. No production data/training started.
