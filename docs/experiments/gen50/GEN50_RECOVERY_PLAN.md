# Gen50 checkpoint recovery — September 16

Owner: PIW following White-regression investigation. Preserve model weights,
search defaults, training corpus, existing studies and selected epoch14. No
promotion, automatic retraining, hand-written move ban, deletion or git action.

## Question and fixed candidates

Epoch14 beatgen49 at3,200 with75%/800 games, especially Black93.25%, but White
vs v27 fell to84.5% and vsB2 to91.5%. After e4+d4 ...d5, c4+c5 scored4W4D59L
vsgen49 and1W1D6L vsv27. Search at12,800 reduces entry into this branch.
Teacher reanalysis policy at the c4 root gives c5 only2.6%, not35% as the new
checkpoint's3,200search does. Also examine the e5+c4 ...f6 king-route split.

Freeze gen49 and gen50 epochs10,13,14,15,23. These cover an earlier checkpoint,
offline peak, current nominee, adjacent finalist and final-epoch control.
Five gen50 alternatives, not a new architecture search or every-epoch test.
Gen49 and epoch14 get exactly the same new instruments as the alternatives.

## Diagnostic roots and interventions

Full-history legal prefixes, all White to move:
1. e4,d4,...d5;
2. e4,d4,...d5,c4 (White's second half);
3. e4,d4,...d5,c4,c5,...Nf6;
4. e4,d4,...d5,e5,c4,...f6;
5. e4,d4,...d5,e5,Ke2,...f6 (comparison);
6. d4,d5,...c5 (alternative opening).

These positions are selected from inspected failures and correlated; not blind
holdouts or independent opening families. Driver restores full repetition
history; native search retains its current recent-history limitation. Fresh
trees do not recreate unrecorded original cached trees.

Pure-search diagnostic crossover: gen49policy/gen49value, epoch14policy/
epoch14value, gen49policy/epoch14value, epoch14policy/gen49value. Mix the two
native bridge output buffers, never change perspective or model tensors.
Search3,200/12,800/51,200, all6roots:72root-turn probes, no early stopping,
no finisher, same deterministic seed per root/budget. Complete both White
halves if applicable. Same-model crossover must exactly match normal bridge
results in rehearsal; fake-buffer tests verify value/policy source routing.
Mixed evaluators are diagnostic interventions, not candidates or proven causal
decompositions of general playing strength. Different policies change sampled
leaves, so interactions matter. Record unmixed raw network legal priors and
values for all6models on every root as well.

Conditional games:6models asWhite, gen49/v27 asBlack,6roots,2samples each,
both3,200/12,800.288games with existing early stopping/finisher and temp.5
through absolute ply16. Same seed per root/opponent/budget/sample across White
models (common random numbers, not guaranteed identical later trajectories).
Do not use these selected-position scores to nominate a checkpoint.

## Normal-start screening and fixed nomination rule

All6models play80games each vsgen49,v27,B2 at3,200:1,440games,40percolor.
Identical seed blocks across candidates within each opponent, separate from
all earlier tests. Score all arms; don't stop on apparent winners/losers.

An alternative is eligible if aggregate vsgen49 >=50%, Black vsgen49 >=
epoch14's same-screen Black score minus5percentage points, and its White
score against EACH ofv27 andB2 is >=epoch14's respective score. Nominate
eligible gen50 checkpoint with highest mean White score vs(v27,B2); ties:
Black vsgen49, aggregate vsgen49, then lowest epoch. Epoch14 is fallback if
none eligible. These small screens only nominate; selection bias is expected.

## Fresh confirmation, even if screen finds no replacement

Do not copy/overwrite arena_selected.pt. Nominee manifest references the exact
existing checkpoint hash. Always run:
- Standard3,200 sampled gate vsgen49:400bar calibration +400H2H +400confirmation.
- 160each vs v27,B2,and nominee selfplay at3,200.
- 160 vsgen49 at12,800.
1,840independent confirmation games; all branches despite a gate FAIL.
Total3,568full games +72probes, excluding rehearsal. Report all arms, actual
color selfplay, uncertainty and endpoint concentration. No automatic release.

## Execution and provenance

New root-level launcher/worker/test/plan; don't edit pinned old tools/tests.
Pin all candidate and opponent hashes, current runtime, tools/tests and new
files. Rehearsal:full tests, bridge parity, every conditional arm at8sims with
one sample,4games per normal block, all confirmation branches and replay audits.
Atomic configs/results/receipts, explicit code-drift rejection, OS campaign
lock, <=8gameworkers, sequential heavy stages, single-process crossover probes
with at most2models resident, target<=12GiB VRAM. Resume completed evidence.
Estimated12–18hours, no hard cutoff or result-driven sample-count changes.
Wait on process completion with long interruptible sleeps and milestone checks.
