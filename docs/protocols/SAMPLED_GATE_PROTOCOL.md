# Sampled free-play evaluation, v3 (September 6, 2026)

This is a separately versioned successor to the endpoint-uniform v2 instrument.
It does not change an active v2 campaign or retroactively turn its result into
a binding v3 PASS. No numbered model release is authorized by this document.

## Question being measured

The primary score is expected captures-only match score under the declared
free-opening sampler: native search, 3,200 simulations, temperature 0.5 for the
first 16 search plies, then temperature zero. Each color receives equal weight.
This is a model-dependent opening distribution, not a uniform distribution over
all positions and not the distribution of human opponents' moves.

Repeated endpoints stay in the score. Independently drawn random openings can
legitimately repeat, and their frequency is part of this question. Deduplication
changes the question to equal weighting of observed endpoints; it does not
automatically repair statistical dependence. Distinct endpoints are also not a
guarantee of strategically diverse tests. History/tree/repetition state can make
equal FENs different continuations. All these coverage diagnostics remain in
the report, without controlling sample inclusion or the gate verdict.

## Fixed protocol declared before new games

`tools/gate_sampled.py` emits `free_sampled_equal_color_v3`:

- A fresh 400-game incumbent self-play calibration, no cache by default.
- First H2H: 200 games as White and 200 as Black.
- Confirmation: another 200 per color, from a disjoint RNG block. It is not
  conditioned on avoiding previously seen endpoints.
- Each full leg must score above 50%; each candidate color must remain at least
  incumbent same-color self-par minus five percentage points. These score floors
  are unchanged from v2; their inputs are now sampled rather than deduped.
- Self-par uses all 400 actual-color outcomes, regardless of whether that color
  was called model A or B. White and Black estimates are complements from the
  same 400 games, **not 800 independent games**. Self-play aggregate is exactly
  50%. This is more efficient and coherent than throwing half the observations
  away for each color according to an arbitrary model-role label.
- Counts are fixed before play, not extended until a favorable score appears.
  No mid-leg performance or novelty stopping. A soft deadline may prevent the
  next leg starting, but every started leg finishes all its scheduled tasks.
  Incomplete required evidence is INCONCLUSIVE, never a model rejection.
- Workers remain capped at eight with one exclusive campaign at a time.

Reports include per-color W/D/L, sampled and endpoint-uniform scores, overlap,
new-endpoint counts, conflicting continuations, mean game length, ending reasons,
and draw-aware nominal standard errors. Color deltas include calibration error.
Intervals are descriptive normal approximations, not sequential or simultaneous
confidence guarantees. A gate PASS is an operational point-estimate screen,
**not proof that Black improved or that both colors are statistically
non-inferior**. Release decisions still need the color results and uncertainty,
supporting opponents, self-skew, and held-out stress diagnostics to be reviewed.

## Provenance and recovery

The run manifest pins checkpoint hashes, conservative runtime identity, v3
implementation hashes, sampler, thresholds, fixed sample counts and seeds before
games begin. Par/first/confirmation blocks are separated by 100,000 seeds;
each individual match is capped at 2,000 games to keep its two color seed ranges
disjoint. Reserve one million seeds per separate campaign; choose a fresh base,
not an increment of one. The initial v3 base is 40,000,000, outside the ranges
used by the September 5 v2 campaign and its many 100,000-spaced batches. Retries
resume the original task IDs rather than create extra trials.

Each leg has a durable JSONL journal and task manifest. Resume validates them
and runs only missing tasks. Final reports pin hashes of both journals and task
manifests. Completed evidence is not silently extended by a resume. Source or
configuration changes require a new separately declared run.

Example of a separate binding run (not the retrospective v2 rescore):

```powershell
py -3 tools/runs.py start --name gen44_sampled_v3 -- py -3 tools/gate_sampled.py --model models/candidates/bootstrap_main_gen_0044/best_value_net.pt --bar-model models/candidates/bootstrap_main_gen_0042/screen_nominee.pt --target-per-side 200 --par-games 400 --sims 3200 --workers 8 --budget-min 180 --seed 40000000 --run-dir benchmarks/sampled_gate/gen44_gen42_3200_v3
```

The legacy `gate_free.py` remains v2. After the original campaign completed,
the pipeline default was explicitly changed to `--gate-backend sampled`.
The generation-only learning recipe is unchanged. Existing completed generation
states must not silently migrate; gen44's epoch screen uses a separate output.

## Overnight work order and decision boundaries

1. Finish the existing gen44/gen42, gen41, v24 and self-play campaign without
   changing its code or declared coverage policy. Keep its v2 INCONCLUSIVE
   verdict if endpoint coverage falls short; do not describe that as rejection.
2. Verify the new tools on tiny real games, and run the held-out human-line
   probes. A held-out miss is diagnostic, not a handcrafted training target.
3. Screen gen44's preserved epochs before assuming another training cycle is
   needed: at most eight representative checkpoints, 40-game probes at 1,600
   simulations, then 200-game screens at 3,200. Two aggregate leaders advance,
   plus the existing Black-best and offline-best safeguards (at most four
   finalists). Reserve seed 50,000,000 for the probe and 50,100,000 for the
   full screen. The currently measured epoch 9 remains in this comparison.
   Free-play screens also use all actual-color self-par outcomes, preserving
   the old role-split aggregates separately. Their v3 manifest distinguishes
   this calibration from earlier screens; the nomination ranking is unchanged.
4. Use the existing generation-only recipe for the next controlled increment.
   Train **fresh from scratch**, inheriting only architecture from the selected
   generator; do not silently switch back to fine-tuning. Preserve 1,000 free
   and 400 book-seeded games, replay window eight, 700-sim generation and
   20,000/10,000 deep reanalysis with 60% Black teachers. The new generator is
   an explicit experimental teacher, not an automatic champion promotion.
   If the screen gives no credible reason to change it, use gen44 epoch 9.
   Keep initialization/training seed 3173, matching gen44; the generation
   index still advances the data-generation and evaluation seed namespaces.
   Audit against gen44's recorded commit `f93ab12`: the model, optimizer,
   processor, replay composition, native search and rule sources are unchanged.
   Training/generation entry points gained exclusive-worker guards; reanalysis
   gained recovery journaling. Comparing gen44's stored configuration with
   defaults at seed 3173 found only path normalization, evaluation depth and
   the old `through_phase=train` stop before the new v3 evaluation additions.
5. Play-test the resulting bounded checkpoint shortlist, then run the selected
   candidate through the new fixed-sample gate against its explicit gen44
   generating checkpoint. Keep both color results visible. If that generator
   changes from epoch 9, retain epoch 9 as an additional reference opponent.
   On a promising result, also check gen42 directly and measure actual-color self-skew, within the loose
   overnight budget. No numbered-release promotion and no push.

A source audit also found that the old native-binary hash lookup can miss the
repository-local extension before its adapter has been imported. That lookup
was corrected after the v2 campaign finished and before new binding runs.
The existing DLL's filesystem timestamp is August 16; that is supporting
context, not a retroactive cryptographic attestation of the old campaign.

Deep reanalysis currently resets turn count and omits recent move history.
That deserves a faithful-state follow-up, but is not an established explanation
of gen44's weakness: only 166 of its 95,170 ordinary raw rows are at nominal
turn 120 or later. Keep this separate from the teacher/epoch experiment unless
a direct diagnostic establishes a material target error. Do not bundle an
unmeasured learning change into what is described as the same training recipe.

Reanalysis can now use `--journal PATH --resume` to retain each completed deep
search across interruption. The journal lives outside the raw training tree;
its manifest pins the source files, sampled tasks, model, runtime and search
configuration. Resume schedules only missing tasks, and completed teacher
output is hash-checked before reuse. This changes recovery, not the search or
target formula. New sampled-backend generations enable this cache by default.

The lightweight `tools/replay_census.py` reads actual split indices and loss
weights without loading the multi-GB position tensors. Its gen44 snapshot is
`benchmarks/gen44_replay_census_20260906.json`: 687,614 sampled train rows,
537,141 distinct array rows after deliberate stratum resampling. These are
array-row counts, **not unique board positions**. Policy-only teachers supply
33.4% of nominal policy weight mass, with zero value weight.

Of the sampled training rows, 520,747 (75.7%) come from gen36-42 increments
generated by **gen33**; 166,867 (24.3%) come from gen44's new increment generated
by **gen42**. Replay source generation numbers must not be mistaken for teacher
identities. Black accounts for 291,658 (42.4%) sampled rows; White has two search
half-moves. The census is descriptive lineage/weight accounting, not a new
dataset-integrity certification or a claim about literal gradient percentages.

## Gen45 workload revision (September 6 morning)

This supersedes the original unchanged-workload plan in step 4 above. Gen45
launched at 11:58 Eastern September 6. The completed saved-epoch screen selected gen44 epoch 9 again,
so use `models/candidates/bootstrap_main_gen_0044/best_value_net.pt` explicitly.

The owner requested more data; the launched increase doubles the new
increment, using genuinely newly generated games rather than
duplicating existing rows:

- 2,000 free games plus 800 book-seeded games, up from 1,000 + 400.
- Sample 40,000 ordinary positions for deep reanalysis and retain 20,000
  policy-only teachers, up from 20,000 / 10,000.
- Preserve 60% Black teachers: 12,000 Black and 8,000 White before mirroring.
- Preserve 700-sim generation, 3,200-sim reanalysis, replay window eight,
  teacher policy multiplier four, zero teacher value weight, scratch training
  seed 3173, architecture, optimizer and evaluation protocol.

Use explicit gen45 overrides `--games 2000 --book-seed-games 800
--reanalysis-sample 40000 --reanalysis-keep 20000`; do not silently change
global defaults or resume an old generation with different data arguments.
Record the resulting row counts and generating-model shares after composition.
The aim is more fresh outcomes and stronger-teacher coverage, not more epochs
over old data. Teacher identity and data quantity now both change, so any gain
cannot be attributed exclusively to either without a separate control.

Launch (full chain, no phase stop and no promotion):

```powershell
py -3 tools/runs.py start --name gen45_expanded_sampled_v3 -- py -3 src/iterate.py --incumbent models/candidates/bootstrap_main_gen_0044/best_value_net.pt --seed 3173 --games 2000 --book-seed-games 800 --reanalysis-sample 40000 --reanalysis-keep 20000 --self-skew-games 200
```

State: `iterations/gen_0045/state.json`; log: `logs/gen45_expanded_sampled_v3.log`.
Full replay uses generations 37-42, 44 and new 45. Training is up to 30 epochs
with patience 10. The checkpoint shortlist is play-tested after training, not on
every epoch. The selected candidate faces gen44 epoch 9: 400 fresh calibration
games plus two separate 400-game H2H legs, all at 3,200 simulations. PASS chains
into 200 self-skew games and ends without promotion. FAIL/INCONCLUSIVE ends
normally; an execution error records the failed phase and stops safely.

Before launch, the expanded dry run validated the plan and replay inputs; all
782 tests and 3 subtests passed. A separate real tiny iteration completed through
all binding legs and correctly recorded a measured rejection, with no handoff
error. Its data and model are isolated under the rehearsal namespace and are not
production replay. Allow roughly 7-9 hours overall, dependent on game lengths,
epoch count and number of shortlisted finalists. No repeated polling or owner
prompt is scheduled. Freeze runtime source files until this run ends.

## Held-out human line

`tools/probe_human_line.py` tests positions reconstructed from a human game's
FEN log without adding anything to training. It records both White half-move
policies and values, restores the full-turn count and repetition prefix, and
lists ambiguous reconstructed intermediate White moves. Its search starts with
a fresh tree; the original GUI's unrecorded cached tree cannot be recreated.

The September 4 game `black_2026_07/game_00031.jsonl` is the owner's reported gen42
failure. The log itself lacks a checkpoint identity. The principal question is
White's turn 8: Kb3 commits White to a second king move, then Ka2 permits Qxf2
with check. The earlier e6 push has not been established as an error. Generic
conditional-first-move probes compare alternatives; no pawn-saving rule or
handcrafted engine correction is introduced.
