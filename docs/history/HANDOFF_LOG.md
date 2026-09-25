# HANDOFF — chronological project record

## September 25 — current consolidated handoff and cleanup

Start with [HANDOFF_20260925.md](../../HANDOFF.md) for the current state,
models, recipes, results, operating constraints and proposed next steps.
The value-calibration campaign completed September 17 at 09:16:27, not still
running: 58.4375%/800 vs unchanged epoch14 at 3,200 simulations, but
49.375%/160 at 12,800. No promotion or follow-up queued. Public release remains
v27; canonical gen50 remains epoch14. See
[VALUE_CALIBRATION_RESULTS.md](../experiments/value_calibration/VALUE_CALIBRATION_RESULTS.md).

September 25 cleanup removed regenerable caches and archived completed logs,
preserving source, models, data and evidence. See `CLEANUP_20260925.md`.
Historical entries below describe the state at their recorded time.

## September17 02:20 — value-calibration launch record (completed 09:16)

Owner authorized another PIW after checkpoint recovery did not fix balance.
`VALUE_CALIBRATION_PLAN.md`, `run_value_calibration.py`, `value_calibration.py`.
Three arms: unchangedepoch14, frozen-backbone/policy value-head fit on replay,
and matching fit with fresh deeper-outcome data. No architecture/search changes.
192newparent games +384full-history cross-model continuations; roots selected
by generic model/search disagreement, balanced Black/White-first/White-second.
Family-linked splits and exact encoded-input leakage filtering. Strict completed
capture outcomes, not root values as labels. Check frozen tensors and rawpolicy
equality. Allarms get normal-play screens; independentgate is vsunchangedepoch14,
plusgen49/v27/B2/self and deepgen49/epoch14checks. Total3,696games.
Newroot-level scripts leave earlier source pins intact. Full rehearsal passed:
983tests+3subtests,120games,two fits,all26receipts,resume and policy equality.
Manual checkpoint comparison confirms only6value-head tensors changed in
each fitted arm; every other parameter/buffer bit-identical. Production launched
02:19Eastern as `value_calibration`,PID27800,log `logs/value_calibration.log`.
Long completion waits and milestone checks; no source changes midrun. Evidence namespace
`benchmarks/value_calibration_20260917`. No old files/models overwritten.

## September17 01:16 — checkpoint recovery COMPLETE, no replacement

See `GEN50_RECOVERY_RESULTS.md`. All3,568games+72probes completed in~6h9m.
Screen selectedepoch15; independentgen49 gatePASS73.9375%/800games,
White50%,Black97.875%. Whitevsv27 88.75%,vsB2 90%. At12,800vsgen49,
White26.25%/Black99.375%: NOT a clean balanced recovery. No model overwritten
or promoted; existinggen50arena_selected staysepoch14. No further run queued.
All25productionreceipts/frozeninputs/nomineehash verified; conditional games
replayed; probe hashes verified. Evidence `benchmarks/gen50_recovery_20260916`.

Crossover points tovalue/search interaction in thec5 branch:epoch14rawprior
only2.25%, full3,200search35.7%; samepolicy/gen49value reduces to4.81%.
But gen49value also favorsc5 atdeepersearch, so no simple transplant remedy.
Next recommended research is generic search-backed outcome VALUE calibration
with frozen-policy controls, not move bans/architecture rewrite. Not queued.

## September16 evening — gen50 complete, checkpoint recovery rehearsing

Gen50 COMPLETE18:08, selectedepoch14. Independentgen49 legs76.375%/73.625%,
combined75% (W56.75%,B93.25%). Deepgen49 54.6875%. GatePASS but White
vsv27 84.5% andvsB2 91.5% are belowgen49's earlier samples. No promotion.
See `GEN50_RESULTS.md` for scores, selfplay and localized White investigation.

Owner authorized PIW on recommendations. `GEN50_RECOVERY_PLAN.md`,
`run_gen50_recovery.py`, `recovery_probe.py`: fixed six-model matched-root
study and normal screens; diagnostic gen49/epoch14 policy-value crossover;
fixed balanced nomination rule followed by1,840fresh confirmation games.
Total3,568games+72probes, no retraining or model overwrite. Full rehearsal
COMPLETE:983tests+3subtests,172games/24probes, same-model native bridge parity,
all26receipts and frozen inputs verified, clean resume passed. A measured tiny
gateFAIL correctly continued through every confirmation branch. Production
launched19:07Eastern as `gen50_checkpoint_recovery`,PID11228,
log `logs/gen50_checkpoint_recovery.log`. Long waits; no source changes midrun.
Evidence `benchmarks/gen50_recovery_20260916`. Old frozen tools/tests unchanged.

## September16 — gen50 deeper-target iteration

September15 corrected mainline study and extension both COMPLETE at19:51;
1,999 games plus108probes, all12production stage receipts checked. At12,800
gen49 scores78.4375%vsgen48 (W98.75/B58.125),99.375%vsB2; deeper vs shallow
gen49 scores65.625%. Gen49 deep self14W/36B/110D. B2White loses64/64additional
...d5 continuations at12,800. This is search-dependent behavior, not proof
of perfect play or a simple first-move fault. No promotion.

Owner authorized tonight's plan/implement/wait. New `GEN50_PLAN.md` and
`run_gen50.py`: gen49teacher, same2800mainline+400fork recipe, forks and
reanalysis doubled to12,800, unchanged scratch training and eight-gen replay.
Full isolated rehearsal then canonicalgen50, standard3,200gate and diagnostics,
plus12,800H2H/B2/self;2,280postselection games, all branches regardless of score.
Rehearsal COMPLETE:982tests+3subtests,28generated games, one-epoch training,
selection and36independent test games. All9receipts verified and rehearsal
resume passed. Production launched00:50Eastern as `gen50_deep_targets`,
PID8276, log `logs/gen50_deep_targets.log`; starts withgen50generation.
Expected12–18hours; no deadline-based truncation, no automatic promotion.
New root-level scripts/recipes leave previous tools/tests source pins intact.
Outputs `benchmarks/gen50_20260916`; no old results deleted or overwritten.

## September15 02:57 — corrected counterplay v2 launched

Stopped ONLY the original `mainline_counterplay` process tree (verified PID69152)
because equal-model conditional games shared a search object across colors,
unlike the normal match harness. No files deleted. Original completed tasks
remain under `benchmarks/mainline_counterplay_20260915`; marked
`stopped_protocol_mismatch`, not a completed strength study. No original
normal-start test blocks had begun; gen49's prior strength campaign is unaffected.

Fixed separate White/Black tree ownership and added a direct equal-checkpoint
regression test.978full tests+3subtests pass. New source/plan uses a fresh namespace
`benchmarks/mainline_counterplay_20260915_v2`; full rehearsal completed169games/
31probes, all10receipts and source hashes verified. An additional117-ply native
game exactly matched the normal benchmark's moves, states and outcome at the
same seed, confirming separate-color tree behavior. Root probes max4workers include
four full204,800-simulation preflight tasks. All production counts/seeds/models
unchanged; rerun cleanly, do not import/relabel the old tasks. Expected about
one-hour delay beyond the original guide. Corrected production launched02:57
as `mainline_counterplay_v2`, log `logs/mainline_counterplay_v2.log`. No model
changes and no promotion. See `MAINLINE_COUNTERPLAY_PLAN.md`.
The01:41 launch below is history, not the currently active production run.

## September15 01:41 — eight-hour counterplay study launched

Gen49 campaign is COMPLETE; see `GEN49_RESULTS.md` for final numbers and the
qualified mainline interpretation. Epoch7 passed independent gen48 legs95.25%/
92.75%, B2 overall83.75%, v27 98.25%. Gen49 self14White/136Black/50draw. No promotion.
The owner's human playtest also feels much stronger. The gen49 status prose
below is now historical, not an active job.

Owner authorized a new roughly eight-hour plan -> implementation -> long waits.
`MAINLINE_COUNTERPLAY_PLAN.md` freezes gen48/gen49/B2 full-history cross-play
after e4+d4 ...e5/...d5, the dominant B2 drawing continuation, deeper root probes,
and four normal-start search-budget checks. No gen50 training, forced-variety
expansion, rule change, promotion, deletion, commit or push.
Implementation: `tools/mainline_study.py`, `tools/start_mainline_study.py`.
Evidence root: `benchmarks/mainline_counterplay_20260915`.
Managed `mainline_counterplay`, PID69152, log `logs/mainline_counterplay.log`.
Production launched01:41Eastern.977tests+3subtests passed; full `rehearsal_v2`
completed169games and31root probes, including four real204,800-simulation
probes (27-43seconds each). All10rehearsal receipts and pinned inputs checked.
The first rehearsal also passed and is preserved; the second added four-worker
root-probe concurrency and the full-budget memory preflight. No production error.
Expected production:1,423games+108probes, all prescribed blocks regardless of
outcomes, then stop. Source frozen; long process-completion waits and infrequent
health reads. The normal free-game auditor now supports explicit unequal simulation
budgets and does not mislabel same weights at unequal search as equal-agent selfplay.

## September14 02:33 — mainline-focused gen49 launched

**04:36 milestone:** all 1,200 gen48 independent normal-start games completed
and replay-audited. H2H vsgen47: 89.25% and92.25% in200game legs, combined90.75%
(White85.5%,Black96%). Gen47 actual-color self-par: White15.5%,Black84.5%/200.
Held-out B2/200 each: gen48 76.25% (White59.5%,Black93%) versus gen47 39.25%
(White3.5%,Black75%). Gen48 self/200:82Whitewins,67Blackwins,51draws,
White score53.75%. This establishes strong normal-start transfer, superseding
the earlier broad-strength concern drawn from BOOK-only evidence. It does not
isolate whether variety helped/hurt training, or establish perfect-play strength.
Evidence: `benchmarks/gen49_mainline_20260914/production/gen48_free_results.json`.
Gen49 generation now underway,650/2800ordinary games at this check; no failure.
Total device VRAM9,546MiB, GPU95%. Code unchanged.

Owner authorized plan -> implement -> long waits, with no further forced-variety
expansion. New frozen plan: `GEN49_PLAN.md`; launcher `tools/start_gen49.py`.
Managed `gen49_mainline` launched; log `logs/gen49_mainline.log`.
954 tests + 3 subtests passed. Full isolated `rehearsal_v2` completed 28 generated
games, training/selection and 48 audited pre/post-test games; all 10 receipts
and pinned inputs validated. First rehearsal's explicit/default simulation
metadata mismatch was fixed; its evidence is preserved separately. No production
failure or restart. First complete independent NORMAL-START gen48/gen47/B2/self
research tests, then 2,800 gen48 normal-start selfplay + 400 deeper continuations,
same architecture/training and rolling eight-generation replay. No new prefix
or league games. Standard sampled gate plus B2/v27/self always run after selection,
including after a measured gate FAIL. No promotion, deletion, commit or push.
Primary evidence: `benchmarks/gen49_mainline_20260914/production`; canonical
`iterations/gen_0049` starts after the 1,200-game gen48 research block. New
candidate then receives 1,800 independent games after checkpoint selection.
Long process-completion waits, infrequent health checks; no file-watcher loop.

Important correction to the paragraph below: ALL 896 independent gen48 tests
used fixed book starts. The flat B2 comparison does not establish flat transfer
from the normal starting position. Free play remains primary by owner decision;
books are secondary. The prior proposed more-diverse teacher/opponent mix is
superseded by this mainline experiment. Prior results/artifacts stay unchanged.

## September13 afternoon — GPU strength track resumes

**FINAL September14 00:05: COMPLETE.** See `GPU48_RESULTS.md` for full evidence
and proposed next steps. Gen48 epoch17 won both fresh256game matches vsgen47:
54.6875% and53.125%; combined53.90625%, paired95%CI[52.1484,55.7617]%. But B2
transfer was flat: gen48 51.9531% vsgen47 52.7344%; White37.5% for both, Black
66.4063% vs67.9688%. Matched overall delta−0.78125pp,95%CI[−6.25,+4.6875]pp.
Common64start selfplay: gen48 11White/33Black/20draw; gen47 13/35/16, identical
White score32.8125%. Modest repeatable H2H gain, NOT a demonstrated broad,
both-color upgrade. No promotion; publicv27/gen46 and referencegen47 unchanged.

All896final games replay-audited, all8stage receipts and pinned provenance
verified after completion.1,160selection games;931Python tests+3subtests.
Managed `gpu48_campaign` exited; no jobs from this campaign or next experiment
remain queued. Production took11h22m without failure/restart. Canonicalgen48
intentionally remains `partial` after checkpoint_screen; research summary is
`complete`, NOT an official binding-gate pass. Candidate epoch17:
`models/candidates/bootstrap_main_gen_0048/arena_selected.pt`, SHA256
`a8c074390c93390ac974b1f58076a86aa0525d9a34f7cff66e12442bb7e07722`.
The dated checkpoints below are history, not current running jobs.

Owner: "same workflow, go" after questioning CPU strength progress. Current
plan is `GPU48_PLAN.md`, superseding the CPU follow-up recommendation below.
CPU code/models remain intact; no new CPU strength experiment is queued.
No deletion, promotion, commit or push. Public release v27/gen46; gen47 arena
epoch17 remains the frozen strength baseline.

**21:06 checkpoint: training completed25epochs (early stopping), checkpoint
screen running.** Training11693seconds (~3h15); reanalysis4536.6seconds;
processing128.8seconds; composition59.6seconds. Probing8/25checkpoints at3200:
epochs9,10,13,14,15,16,17,25. Offline-best epoch15 is not a strength verdict;
nomination still depends on games. No post-selection results yet. All3200games completed
without failures; generation13563seconds. All24000deep searches finished.
Coverage sampler used all2800original source-game families:24000roots from
231679eligible rows, max10roots/family (cap16), exactly14400Black/4800White-first/
4800White-second. Retained12000teachers span2783families, max10/family:
7200Black/3225White-first/1575White-second. New processed corpus487358rows
INCLUDING augmentation and24000augmented teacher rows; not487358independent
positions. Generation auditPASS, all12000teacher split links checked. These
are coverage/integrity results, NOT strength evidence. Training remains
scratch30epochs/patience10, unchanged architecture/optimizer. Managed PID2120
remains active, no source/recipe change or extra experiment added. VRAM was
under10GiB in generation, ~6.5GiB during reanalysis, ~2.7GiB at first training
epoch; do not change batch size to fill VRAM and confound this training recipe.

22:11resource note: device-wide VRAM reached~13GiB after a separate graphics
workload appeared. Windows per-process counters for the eight Python game
workers totaled~7.2GiB (roughly0.9GiB each), within the task's12GiB target.
Do not terminate other applications or silently reduce this run's simulations.
Selection games looked promising, but no independent confirmation result was
available at this checkpoint. Keep final testing unconditional.

Implemented `tools/reanalyze_coverage.py` (opt-in deterministic transitive-family
sample cap16, 60% Black, phase/family census) via `iterate_stateful.py`;
gen48/rehearsal recipes; `tools/start_gpu48.py` automatic guarded chain with
immutable input receipts and full game replay audits. Existing GPU architecture,
scratch30epoch/patience10 training and bounded checkpoint selection retained.
Fixed common-selfplay helper's capture-only accounting: +/-0.5 caps are draws.
Read-only check of all13 JSONL logs in the September10 B2 confirmation found
zero +/-0.5 results, so that historical comparison is NOT affected by this fix.

Full-depth64game pilot finished12:29, 64/64 saved, no failures,358.05seconds.
Per preregistered sizing rule, final production3200 games:1600free+800fresh+
400balanced older-opponent games+400 completed forks. Ordinary1600/forks6400
sims. Reanalysis24k sample/12k retained@6400; no B2 training opponent/teacher.
Focused34 tests pass; both real command plans pass dry-run.402GiB disk free.

At12:42 Eastern the managed `gpu48_rehearsal` completed successfully:931Python
tests plus3subtests passed (172existingPyTorch deprecation warnings);28game tiny
iteration ->all six post-selection branches (20 games), all replay-audited.
The20 coverage roots represented15original families; retained10 teachers from8
families,6Black/4White; all data-family and training/selection checks passed.

At12:43 Eastern launched managed **`gpu48_campaign`, PID2120**, log
`logs/gpu48_campaign.log`, same launcher without `--rehearsal-only`;
matching receipts skip completed rehearsal work. It chains
the canonical gen48 through checkpoint_screen, then896 fixed games regardless
of score:256+256 separate H2H legs vsgen47,128/model vsB2,64/model selfplay.
Canonical status intentionally `partial`; separate research summary `complete`
does not mean accepted/promoted. Output `benchmarks/gpu48_20260913`.

Waiting: use OS process completion, not filesystem watchers or per-game probes.
One heavy GPU job,8workers, existing pinned-input optimization. Source/model/
recipe changes fail closed; partial training is retained, never restarted
automatically. An inconclusive result does not launch another parameter sweep.

## September13 overnight — search-backed evaluator training

**FINAL06:17: COMPLETE.** See `SEARCH_TARGETS_RESULTS.md`. All208 final real games
and two separate16game rehearsals passed replay audits. Final managed run was
`search_targets_recovered_v2`; no jobs remain active/queued. Ranked CPU versus
gen47 at2s:43.75% vs unchanged39.0625%, White25% unchanged, Black62.5% vs53.125%.
Paired+4.6875pp,95%CI[-6.25,+17.1875]pp: inconclusive, not a both-color upgrade.
RankedvsB2:35.4167%,White16.667%,Black54.167%. Same12start selfplay@1s: ranked
7White/3Black/2draw; gen473White/4Black/5draw. No promotion/default switch.
Releasev27/gen46 and GPUgen47 remain unchanged. No deletions/commits/pushes.

Implemented training-only native minimax targets / exact leaf lineage plus
matched raw/backed/ranked CPU training and safe recovery. All12epochs perarm
finished; epoch1 selected in each. An epoch already resampled the new22,742
TRAIN states about32times because it is one728,450-row original-replay pass.
Next recommendation: more independent actual CPU NN leaves, consistent target
horizon, controlled new-state exposure, then fresh both-color play checks.
The88.8% one-turn-successor mixture is a calibration hypothesis, not a proven
cause. No follow-up experiment was launched.909Python+3subtests/26Rust passed;
final default search snapshot parity also passed. The checkpoints below are
chronological history, not current active jobs.

Owner authorized overnight plan → implement → long waits / completion events.
Current plan: `SEARCH_TARGETS_OVERNIGHT_PLAN.md`; output root
`benchmarks/search_targets_20260913`. No promotion/deletion/commit/push.
Playing architecture and CPU search remain absolute840→512→32→1, baseline.
Three matched fine-tuning arms from CPUleafepoch3: raw targets, search-backed
targets, and backed targets plus sibling ranking. Same data,12epochs,optimizer,
replay anchor and validation nomination; all nominees must receive actual games.

Important discovery: existing GPU PUCT does not use full CPU repetition history
inside search and its cap labels differ. Implemented training-only full-width
two-completed-turn `LabelTree` with exact CPU-rule cap/repetition/capture handling
and batched gen47 GPU frontier values, rather than silently mixing those rules.
Teacher estimates remain imperfect. Actual CPU sample paths are retained with
optional `collect_leaf_paths`; old sampling return tuples remain compatible.

Saved old native runtime/source and baseline under the new benchmark root.
Default fixed-node parity passed.26Rust/12focusedPython tests passed. A first
pilot safely stopped on Windows source-path spelling; its evidence is retained.
`pilot_v2` passed24roots/66trees/586records in12.33s; no capped trees. Mean absolute
raw/backed disagreement0.214 (not proof of correction). Full planned size is
1024TRAIN +256VAL roots, up to2 actual CPU leaves perroot, up to8sibling states
per teacher root, followed by exact-input dedup/opposite-split exclusions.

Managed `search_targets_rehearsal` passed the complete tiny chain, including
all16 replay-audited games. Real `search_targets_campaign` launched01:19 Eastern,
PID62476; output `benchmarks/search_targets_20260913/campaign`, log
`logs/search_targets_campaign.log`. Its source-identical success guard passed.
The real chain conducts full
generation/preparation, three12epoch training runs, a fresh randomly sampled
mixed-gen47/B2 opening book,24games per unchanged/raw/backed/ranked arm@300ms,
32fresh matched games per unchanged/best-trained@2s regardless of development,
24B2games@2s and12selfgames each for trained/gen47@1s, then replay audit.
One heavy stage / one resident match worker, <=12GiB allocated VRAM. Do not edit
pinned native/runtime/Python dependencies once rehearsal/campaign is running.
Use completion waits, not per-game probes. Older sections below are history.

01:27 checkpoint: full generation complete1280roots/3770trees/33652records,
5,264,384GPU frontier evaluations in338.69s; zero capped trees. Prepared
22742TRAIN/5712VAL unique states,12890/3274 ranking pairs; four TRAIN inputs
have repeated backed-target spans>0.25 (history/horizon ambiguity is reported).
All three full12epoch runs completed and nominated epoch1 under the shared
validation criterion. Native parity max<3e-7; peak allocated VRAM3.10GiB.
Candidates: `models/candidates/search_backed_{raw,backed,ranked}_001/epoch_001.bin`.
Opening-book/game stages are now running. Rehearsal and real full suites each
passed905 tests plus3subtests. No strength conclusion yet.

01:49 recovery: original campaign stopped during development_raw on Windows
PermissionError replacing status.json. The observer may have triggered the
reader-lock race. Data, all weights, frozen book and old games are preserved.
atomic_json now retries PermissionError for at most1s; four focused tests pass,
including a real Windows reader lock. search_targets_campaign --reuse-from
validates prepared data, training recipes, model/book hashes and restricts source
differences to receipt I/O and orchestration. All games restart in a new folder,
with old interrupted-run games excluded. `search_targets_recovery_rehearsal`
PID20056 is exercising this path. `search_targets_recovered` PID60588 is queued
behind it and requires its source-identical successful receipt. Its new output
is `benchmarks/search_targets_20260913/campaign_recovered`; log is
`logs/search_targets_recovered.log`. Use process waits, not the status watcher.
No playing changes and no discarded data/models.

01:50 recovery-final checkpoint: the first recovery rehearsal safely caught a
relative/absolute book-path lookup mismatch; normalized recorded paths. Its
dependent queue could not pass the missing-success guard. The second recovery
rehearsal (`recovery_rehearsal_v2`) PASSED909 tests +3subtests and16 replay-audited
games. Current managed real run is `search_targets_recovered_v2`, PID24548,
queued behind that completed rehearsal. Authoritative real game output will be
`benchmarks/search_targets_20260913/campaign_recovered_v2`; log
`logs/search_targets_recovered_v2.log`. Original data/weights/book unchanged.
Earlier run names above are retained as failure/recovery history, not active
strength evidence. Wait on PID24548 completion; no live status-file observer.

02:35 completed milestone: all96 real development games finished in recovered_v2.
Overall unchanged27.083%, raw20.833%, backed22.917%, ranked25.000% againstGPUgen47
at300ms on common fresh starts. White/Black respectively: unchanged29.167/25.000,
raw8.333/33.333, backed25.000/20.833, ranked25.000/25.000. No improvement established.
Best trained is ranked; unchanged wins overall. As predeclared, BOTH unchanged
and ranked still receive32 fresh common2s games, followed by rankedvsB2 and
common selfplay. Real chain remains active; process-completion wait is cell416
(terminal session79256, waiting on managed PID24548). No recipe/gate changes.

Read-only target inspection is recorded in `SEARCH_TARGETS_DATA_DIAGNOSTIC.md`.
Before dedup,88.8% of records are one-turn successors; source / actual NN-leaf
roots have two-turn targets. Only8.54% are exact±1, so the data is not mostly
trivial solved captures. Successor means shift toward the side to move by about
0.15, unlike the two-turn root means. Mixed-horizon calibration is a hypothesis
to test next if games fail, not a proven explanation or a mid-run recipe change.

05:57 completed milestones: fresh common2s confirmation finished32games perarm.
Unchanged11W/3D/18L39.0625%, White25%,Black53.125%; ranked10W/8D/14L43.75%,
White25%,Black62.5%. Paired+4.6875pp,95%CI[-6.25,+17.1875]pp: inconclusive.
White score equal but outcomes differ (unchanged3W/2D/11L vsranked1W/6D/9L),
not the identical-White-outcomes result from the previous scaling campaign.
RankedvsB2 completed24games:8W/1D/15L35.4167%,White16.667%,Black54.167%.
Selfplay and final audit remain; wait on the same managed PID24548/cell416.
No promotion, new experiment, threshold change or fresh training was launched.

## September12 afternoon — CPU cost and search scaling

**FINAL22:08: COMPLETE.** `SEARCH_CPU_SCALING_RESULTS.md` is the final report.
160 real games plus14 rehearsal games passed replay audits; no proof
contradictions. Baseline/optimized/incremental development scores42.1875% /
39.0625% /42.1875%; baseline selected by tie rule. Fresh matchedCPU2s vsCPU8s
againstGPUgen47fixed2s:42.1875% ->46.875%,White28.125% unchanged,Black56.25%
->65.625%. Paired overall+4.6875pp,95%CI[-3.125,+14.0625]pp: inconclusive.
Extension did not trigger. On first shared CPU roots, depth6.156 ->6.625turns;
no node/depth ceilings in scaling. No queued/running jobs, promotion, training,
deletions, commits or pushes. Next recommendation is training-only search-backed
and move-ranking targets plus failure analysis, not another throughput-only
campaign. The dated checkpoints below are historical.

Owner: "Same process: plan -> implement -> wait." Current plan is
`SEARCH_CPU_SCALING_PLAN.md`; outputs `benchmarks/search_cpu_scaling_20260912`.
The earlier CPU/GPU campaign COMPLETED successfully at10:17 Eastern:
guided46.875% versus unchanged CPU47.656% over64 common games at2s againstgen47.
Guided White26.5625%,Black67.1875%; unchanged White31.25%,Black64.0625%.
No established strength gain; PVS/cache fixed-depth speed gain15.7% did not yet
receive its own pure-CPU2s confirmation. Release remainsv27/gen46.

New measured cost profile: NN evaluation61.98% of instrumented search time;
move generation6.13%. Instrumentation overhead5.36%, default fixed-node parity
unchanged. Implemented opt-in lazy first-layer feature deltas,32-update refresh,
relative-model direct fallback, numerical diagnostics and per-player clocks /
CPU node ceilings.26Rust and18focused Python tests pass. Managed validation
`cpu_scaling_validation` is active; final `validation.json` nominates an optional
incremental arm only after drift and full-search speed checks.

Next chain: complete tiny rehearsal -> baseline / existing optimized / optional
incremental32games each against GPUgen47 at2s on336..351 -> freeze best2s arm ->
matched32games each atCPU2s andCPU8s versusfixedGPU2s on352..367. Conditional
extension to368..383 only if overall gain>=10pp and Black delta>=0.100M CPU
node limits both clocks; node-bound evidence stops the chain. Final replay audit.
Unequal-clock scaling is diagnostic, not a promotion gate. One heavy job at a time,
one resident match worker,12GBVRAM ceiling, source hashes and heartbeat receipts.
Do not edit pinned dependencies once rehearsal starts. No training, model
promotion, deletions, commits or pushes in this block. Prior sections are history.

UPDATE15:05: validation and complete tiny rehearsal passed.893Python+3subtests,
26Rust. Incremental max eval error1.252e-6; fixed-depth error<=1.193e-7 with no
changed moves on18replay roots. Incremental optimized saved6.04% elapsed versus
existing optimized search,18.69% versus default. Six timed roots averaged depth
6.0 at2s and6.667 at8s forbothoptimizedvariants. Three arms qualify for games:
baseline, optimizedPVS/freshTT131k, same+incremental. Managed
`cpu_scaling_campaign` is queued after the successful rehearsal; source/model
hashes are frozen.14 tiny rehearsal games passed full replay audit and must not
be interpreted as strength evidence. The real campaign is160games minimum,
224if the matched8s extension triggers. Follow milestone receipts, not each game.

UPDATE16:29: baseline development completed32games in~83minutes:12W/3D/17L,
42.1875% overall,White9.375%,Black75%. Total outcomes were4White wins/25Black
wins/3draws on16paired starts. No proof contradictions or node-limit hits; three
Black decisions reached the default12turn depth ceiling. The optimized arm
started automatically on the identical starts. Do not compare these raw color
scores to different opening sets; paired-arm differences are the target evidence.

UPDATE17:48: existing optimized search completed32games:11W/3D/18L,39.0625%,
White9.375%,Black68.75%. Paired delta versusbaseline-3.125pp,95%bootstrapCI
[-10.9375,+3.125]pp; White unchanged,Black-6.25pp (one win). Inconclusive, not
established improvement or regression. No proof contradictions / node-cap hits.
`development_incremental` started automatically; frozen2s/8s scaling follows
the best development arm. Do not equate game-average NPS/depth on different
reached positions with a controlled same-position performance measurement.

UPDATE19:09: all96 development games completed. Incremental12W/3D/17L,
42.1875%,White9.375%,Black75%: exactly tied baseline by color and overall.
Paired incremental-baseline delta0pp,95%CI[-7.8125,+6.25]pp. No demonstrated
strength gain from either optimization. Tie rule selectedBASELINE. `scaling_2s`
has begun onfresh352..367, then the same baseline at8s versusGPUgen47fixed2s.
The conditional extra32games perclock on368..383 remains governed by the
predeclared>=10pp overall / nonnegative Black delta trigger. Do not promote.

UPDATE20:08: fresh2s scaling control completed32games in~58.6minutes:
13W/1D/18L,42.1875%,White28.125%,Black56.25%. No proof contradictions,
node-limit hits or depth-ceiling completions. The matchingCPU8s run against
fixedGPU2s has started. Its32games, conditional extension and final audit remain.

## September12 overnight — CPU efficiency and GPU cooperation

Owner authorized roughly eight hours from02:26 Eastern. Current concrete plan:
`SEARCH_CPU_GPU_PLAN.md`. Previous leaf campaign AND extended campaign completed
and passed replay audits. Leaf epoch3 scored43.75% vs GPU gen47 at2s over64games,
original32.81% on identical starts; CPU-only leaf scored71.875% over64games at300ms.
Self2s leaf3White/11Black/2draws, gen47 4White/9Black/3draws on common16starts.
No engine has been promoted; v27 remains the release, gen47 the GPU reference.

Implemented optional PVS, fresh-per-iteration searched bounds, bounded TT capacity,
root policy ordering and phase/saturation telemetry in native alpha-beta. Defaults
preserve saved fixed-node decisions/nodes.26Rust tests and focused Python tests
pass, including PVS versus exhaustive minimax in all phases. Snapshot at
`benchmarks/search_cpu_gpu_20260912/runtime_before`. The first shallow profile was
~18% faster; a broader profile was~8% faster, so those are preliminary. A deeper
final profile is running under `cpu_workload_final` before freezing nomination.

`src/cpu_search_engine.py` is the reusable adapter: recurring GPU policy time is
deducted from the per-move clock; depleted clocks return legal unvalued fallbacks.
`tools/search_cpu_gpu_match.py` supports CPU baseline/nominee, GPU root ordering,
GPU-White/CPU-Black routing, direct CPU duels and normal GPU PUCT. Sources/models,
actual clocks, per-player backend/depth and outcomes are recorded.

`tools/search_cpu_gpu_campaign.py` is prepared: profile receipt -> full suite ->
tiny complete-game rehearsals and audit -> direct CPU screen ->32games/mode
againstgen47 on288..303 -> fresh64games/control at2s on304..335 ->B2 and common
selfplay ->replay audit. Stage heartbeat every60s, immediate child-exit checking,
4h stage safety timeout, failure receipts. Managed launch after final profile;
do not change pinned code/runtime once that campaign starts. No deletions,
retraining, promotion, commits or pushes in this session.

UPDATE02:43: final profile completed, selectedPVS+freshTT+131072entries. Equal-
depth time7.8665s ->6.6346s (1.186x), but nine timed probes' average completed
depths unchanged.26Rust/886Python+3subtests and four complete-game rehearsals
passed. `search_cpu_gpu_campaign` is active in cpu_vs_cpu, then automatically
chains development, fresh confirmation, B2, selfplay and replay audit. Runtime
is now pinned; do not rebuild or edit its dependencies mid-campaign. The plan
has the detailed results. This is measured efficiency, not yet measured strength.

## September11 extended session — representation rejected, leaf audit active

UPDATE: diagnostic completed successfully; detailed findings and next protocol
are in `SEARCH_FIRST_LEAVES.md`. Actual-leaf validation compression MSE is
0.224White/0.099Black vs0.050/0.013 at stored roots. Unseen input columns affect
only29/15,812 absolute leaves, so they are not the leading explanation.
Current managed run is `search_leaf_corpus` (2048TRAIN/512VAL roots); queued
`search_leaf_campaign` requires its success receipt, then tests -> matched
replay/leaf fine-tuning -> original/replay/leaf games -> conditional confirmation
-> matched2second games -> common selfplay -> human diagnostic. New models are
`search_leaf_replay_001` and `search_leaf_leaf_001`; no production promotion.
Do not edit pinned runtime/training/match scripts after campaign starts.

Further UPDATE: corpus completed156,377 leaves; after input-identity dedup and
opposite-split exclusions124,515TRAIN/30,834VAL leaves remain. Full suite875tests
+3subtests passed. Replay nomineeepoch2, leaf nomineeepoch3; export max error
<3e-7 and peak VRAM3.736GB. Leaf validation MSE0.1965 ->0.1075 while original
root MSE0.0313 ->0.0371: promising compression tradeoff, not yet a strength result.
Campaign is in development games. `search_leaf_extended` is queued after its
success and replay audit: fresh matched64games/control at2seconds, B2, CPU-only,
and matched2second selfplay. See SEARCH_FIRST_LEAVES.md for exact protocol.
Both chains persist independently of this conversation; no per-game agent polling
is necessary. Failures write failure.json and stop downstream work via receipts.

Owner granted roughly eight more hours from13:18 Eastern, with low-frequency
milestone monitoring. The frozen representation campaign is complete: absolute
6/32 points (18.75%), king-relative5/32 (15.625%), delta-3.125pp. Neither is a
challenger. Conditional confirmation correctly did not trigger. Keep absolute512
as the experimental control; gen47 is still the reference and v27 the release.

Optional native leaf reservoir now implemented and built (default disabled).
It samples uncached, non-proven NN evaluations with independent RNG, preserves
raw EP and reports raw unclipped values. Fixed-node decisions/nodes/cache hits
match sampling-disabled search in all three phases.24Rust and18focused Python
tests pass. Pre-change runtime saved in runtime_before_leaf under the existing
search_first benchmark root. Active managed diagnostic: search_leaf_audit,
output benchmarks/search_leaf_audit_20260911. It uses only original TRAIN/VAL
roots with full recorded history; no test/human/gate positions become training.

Next: inspect actual-leaf versus stored-root compression error and feature
coverage (tools/search_feature_coverage.py), then choose a bounded training
control if evidence warrants. Potential EP/unseen-feature mismatch is only a
hypothesis, not an established explanation. Unlinked search_window.rs prepares
exact PVS helpers, not yet enabled or tested. Any search optimization must pass
fixed-depth parity and measured time-to-depth before play-testing. No runtime
changes while a diagnostic/match is active. No promotions/deletions/commits.

## September11 afternoon — king-relative control started

Owner authorized the next focused evaluator experiment. Current plan and exact
controls are in `SEARCH_FIRST_RELATIVE.md`. Overnight validation and label-control
chains completed around04:11; no stronger engine emerged. The full completed-game
replay audit now passes. No promotions, deletions, commits or pushes.

Active job: `search_relative_campaign`, `tools/search_relative_campaign.py`.
CPU-only baseline finished16games at50% (White62.5%,Black37.5%); same-start GPU
baseline was37.5%. Small deployment screen, not established strength parity.
`tools/search_value_features.py` implements the6240-input king-relative schema;
`tools/train_search_value.py` now supports sparse input storage/dense minibatch
training and MCSV002 export, while preserving absolute defaults. Native MCSV002
integration is built;22Rust/20focused Python tests pass. Old128/512 predictions
remain exactly equal on122 saved states. Runtime snapshot is preserved under
`benchmarks/search_first_20260911/runtime_before_relative`. No runtime edits while
the new campaign runs. Chain: full suite -> absolute512 and relative512 training
with same sparse input path -> trained-model parity/profile ->32games/arm on64..79.
A relative development gain>=10percentage points triggers fresh128+ gen47/B2,
common selfplay and human diagnostics. Trigger is NOT a promotion threshold.
Gen47 remains the reference; v27 remains the release. Do not spend usage on
minute-by-minute conversational polling; use managed chains and milestone checks.

## Earlier overnight checkpoint (historical)

Owner authorized an unattended search-first / cheap-evaluation experiment,
without a hard time limit. Detailed current plan, boundaries and evidence:
`SEARCH_FIRST_EXPERIMENT.md`. Gen47 remains the strength reference; v27 remains
the official release. No promotion, deletion, commit or push in this session.

New opt-in native alpha-beta, small CPU value evaluator, training/export tools,
timed match driver and history-safe search caches are implemented. Existing
simulation-limited MCTS behavior remains unchanged; timed calls additionally
use a parity-tested ownership-transfer reroot to reduce clock overhead.
Model binaries are separate `search_first_*` candidates, deliberately not CNN
notebook choices. Preserve pre-session dirty worktree changes.

Current run: `search_first_validation`, `tools/search_first_validation.py`.
Locked width512/extension0 nominee, epoch7, same gen47-distilled value family.
Full suite853 tests+3subtests passed. Then64games gen47@300ms,16@2s,32againstB2,
16common-start selfgames per engine, one free-play color pair. New-for-this-
experiment indices32+ from the existing confirmation book. No runtime edits
while this chain runs. Same-family128/512 x extension0/2 development screen
finished:21.875%,25%,34.375%,28.125%. No arm beat gen47; no promotion claim.
CPU evaluator scratch/dot optimization verified prediction delta<3.3e-7;
width512 fixed-state search throughput267k->367k nodes/sec.21 Rust tests pass.

Earlier results: raw outcome student0/8 at2s; distilled student1W/1D/6L at2s.
Optimized distilled128 no-extension300ms:2W/3D/11L (White31.25%,Black12.5%).
Small samples, not general strength estimates. More efficient search has NOT
yet produced a gen47-beating engine. Optimized fixed-depth values matched the
uncached implementation; initial-position300ms depth3 ->4. Exact capture scans
matched the old implementation on10,000 random-walk states. Latest Rust tests
cover full-turn threat extension versus exhaustive minimax. See artifacts in
`benchmarks/search_first_20260911/` and managed logs for current progress.

Next after this validation: inspect results/timings/proof consistency, known
human-line diagnostics, then decide whether further search-first work is useful. Keep
nominal search budgets distinct from measured elapsed time; early prototype
MCTS overran2s to~2.2s, mostly subtree-copy overhead. Timed fast-reroot reduces
that; full end-to-end timings are logged on every move. No scores are silently
relabelled as strict equal-time evidence.

## September10 locked challenger tests

Owner requested necessary tests after simplifying to one challenger versus gen47.
b2_challenger_confirmation runs tools/b2_challenger_test.py. Challenger is fixed:
b2_seed9053_state_cnn/selected_epoch_008.pt; reference gen47arena_selected(epoch17).
No training/checkpoint reselection/architecture change/promotion.8workers sequential.
First known human games black_2026_07/game00031/32,rows0,2,..18,3200/6400,
seed101,five-ply continuations againstgen47 (diagnostic,not new heldout truth).
Then400unique opening states excluded from all previous B2 book files, sampled
fromv24/v25/v26/gen44/gen47 at16plies,temp.5,700sims,seeds2060000000+;
bounded3proposals,8x attempts,no shortened count. Exact exclusion state key
includes FEN,White phase,and turn_count.
2800games:800paired direct3200,400free direct3200,400paired direct6400,
400/model common four-opponent panel,200/model one-game-per-opening selfplay.
Direct/depth/broad subsets overlap by design; don't pool as independent trials.
Per-color WDL and paired bootstrap intervals generated. Existing harness has no
per-move clock; these are NOT equal-time tests. Timing is recorded,strict
equal-time remains an explicit limitation. No speculative clock/search rewrite.
Artifacts:benchmarks/b2_challenger_confirmation_20260910.2new exclusion tests pass.

## September10 new nine-hour run

b2_nine_hour_20260910 started at owner request. See B2_NINE_HOUR_RUN.md.
Two fresh seed9053 control/state trainings on unchanged data; up to4800games:
9model/checkpoint400game matched panels,600one-game-per-common-opening selfplay,
600deeper6400sim games. Paired uncertainty/Black regression examples generated
automatically at end. No promotion. No runtime edits during active chain.

## September9 finalist benchmark continuation

Hybrid fresh retrain completed18epochs,best8; all-arm smoke passed. The initial
2800-game screen finished. Finalist book creation then stopped:96/100 unique
positions at the default1.6x attempt budget, so no finalist book was published.
Owner requested continuation. b2_benchmark_resume runs tools/b2_resume_benchmark.py:
validates original runtime/models/driver, records a separate recovery manifest,
builds the missing100-position book with4x attempts and unchanged seed1990000000,
models v24/v25/v26,depth16,temp.5,700sims. Rejects short/duplicate books. Then runs
the unchanged driver; original screen journals skip completed games. No old
manifest, model or screen protocol modified.6 targeted recovery/ranking tests pass.
Remaining:3200 finalist games (6candidate checkpoints plus2references,400each)
then600self-games (one selected checkpoint/arm). No automatic promotion.
Recovery evidence:benchmarks/b2_001_comparison_20260909/opening_recovery_20260909.json.

## September9 reboot recovery: hybrid retrain and all-arm benchmarks

Windows Update triggered a planned restart03:29–03:32. All12k generation games,
80k reanalysis/40k retained, and913,960-row15/24-channel preparations completed.
Control/state-CNN both finished22epochs,best12, receipts verified. Hybrid was
interrupted after18 saved epochs; there was no completed hybrid receipt.

Owner requested deleting interrupted hybrid and retraining then benchmarking.
Permanent deletion was blocked by execution policy. Instead preserved all19.pt
files in models/archive/b2_001_hybrid_interrupted_20260909 (recoverable), leaving
the candidate destination empty. No other models/data changed.
b2_hybrid_retrain runs tools/b2_retrain_hybrid.py: validates original runtime,
all processed artifacts and completed-arm hashes; reconstructs exact original
command from control receipt changing only input24/attention2/hybrid directory.
Fresh seed3173, no weight/optimizer resume. Writes hybrid receipt, training-complete
receipt and runs all-arm inference smoke. Source runtime frozen during work.

b2_001_benchmark queued behind it, requires successful hash-matching smoke.
tools/b2_benchmark.py: up to4 epoch checkpoints/arm (thirds/last/best epoch,
deduplicated),200games each againstv24..27, split24free/26matched per opponent;
top2/arm400games on fresh starts; selected1/arm200self-games. v27/gen47epoch17
also receive matched panels as references. Maximum6600games,8workers,3200sims.
Exact integer W/D/L fractions drive rankings (3new benchmark tests passed).
These are selection tests, NOT equal-time/second-seed/untouched confirmation;
no automatic promotion. Benchmark output benchmarks/b2_001_comparison_20260909.

Teacher screen rounding bug remains documented: actual Wscore epoch11=epoch17
=.63;B.78 versus.815. Floating averages incorrectly chose11. Existing data
uses11 and is preserved; new benchmark code fixes counting without rewriting
historical receipts or switching the teacher halfway through the experiment.

## September8 B2 implementation update

See B2_IMPLEMENTATION.md for the current implementation and exact evidence.
24-channel Python/Rust state input and two-block hybrid implemented; old models
preserved. All three arms completed a tiny generation/reanalysis/training/CUDA/
native-search rehearsal. Full suite825passed; latest B2 targeted16passed.
Streaming sparse preparation avoids allocating a whole dense policy corpus.
b2_teacher_screen is active, selecting among fixed gen47 checkpoints against four
opponents before production. Do not edit src/native runtime while it is running.
Production generation/training entry:tools/b2_campaign.py. Later full B2 strength
screens/second-seed/confirmation remain separate work, not completed by rehearsal.
b2_production is queued behind b2_teacher_screen; b2_production_smoke follows.
Both require successful predecessor receipts. No automatic strength promotion.
Pinned-input full-game validation finished:11.6%generation and10.6%resident-match
time reduction at3200sims with exact record parity. Campaign opts in; global off.
Native prior DLL preserved at native/monster_native.pre_b2_20260908.pyd; old loaded
notebook kernels need restart to see expanded encoding. No user process killed.

## September8 Bootstrap-2 optimization-first work

Owner retained ALL THREE architecture arms and directed production profiling and
measured optimization before architecture trials. Full plan: B2_EXPERIMENT_PLAN.md.
Do not substitute a two-arm plan. Candidate naming b2_001_*; public next releasev28.

No old campaigns remained active at profiling start. Added fixed-state profiler
tools/profile_search_workload.py, graph-cache ABBA whole-game validator, CUDA
bridge tracer and isolated pinned/packed-transfer benchmarks. Eight workers max,
sequential worker lease. Graph-cache, CUDA trace, transfer microbenchmarks and
pinned-input whole-game validation completed. Follow-up b2_profile_pinned ->
b2_profile_single runs sequentially. These are finite experiments, not an
assistant polling loop. Details and benchmark paths: OPTIMIZATION_B2.md.

Initial evidence: benchmarks/b2_profile_baseline_8.json versus
b2_profile_graphcache_8.json:96 fixed-state decisions, exact actions/policies/values;
wall13.96s ->12.07s. Capture count368 ->119. This is NOT a full-game speedup.
CPU callback time includes GPU waits and synchronization, not pure CPU overhead.
GPU snapshot ~10.7GiB,91% utilization.

src/native_mcts.py has opt-in MONSTER_CUDA_GRAPH_CACHE=1 evaluator-owned graph
reuse, model/storage/signature invalidation and locks through output copies.
DEFAULT IS OFF: whole-game ABBA measured only0.9% speedup, insufficient. No precision,
batch shape, sims, search tree or rules change.28 targeted tests passed.
Match workers already retain engines, so this candidate principally targets
generation's repeated engine construction; don't claim it accelerates long gates.
MONSTER_PINNED_INPUT=1 is a second opt-in production candidate. Its128-game ABBA
passed exact full-record parity and reduced wall time10.1% (73.982s ->66.537s).
All96 fixed-state700/3200 actions/policies/values also matched baseline exactly.
It remains off pending resident-engine/deeper throughput validation. Single-worker
profile -> full pytest tests are queued as b2_profile_single -> b2_suite.
Output packing produced no useful microbenchmark benefit and was not adopted.
No architecture training/data generation or release promotion has started.

## September 8 morning extension

Owner requested queued useful work through roughly 11am Eastern. Run
`gen47_morning_checks` waits for `gen47_stateful_mixed_v3`, requires its transfer
state complete, then runs `tools/gen47_morning_checks.py` until
2026-09-08T11:00:00-04:00. First: 200 gen47 self-games at6400. Then fresh
60-opening matched baseline(v27)/candidate(gen47) comparisons, rotating
gen42/v25, gen44, gen45/v26 at alternating3200/6400, plus120 free games each
round. Eight workers, one sequential job. Starts a new round only before11am;
finishes already-started rounds (can extend past11). No training/promotion,
threshold change, data ingestion or assistant polling. Errors stop safely.
Reports: `benchmarks/generalization/gen47_morning_20260908/`. Resume hashes
protect completed stages. Round-plan test passed.

Gen47 binding confirmed PASS: first400 93.75% (W92.25 B95.25); confirm400
91.625% (W87.5 B95.75). Self200:21 White captures,150 Black captures,29 draws,
actual White score17.75%. Existing transfer checks began September8 at01:52.

## September 7 release / gen47 update (supersedes current-run rows below)

Later owner cleanup: 36 retired candidate directories archived recoverably to
`models/archive/cleanup_gen47_20260907/candidates` (3.12 GiB; no deletion).
Active gen42/44/45/46/47 and the gen47 rehearsal remain. See the archive README
and `benchmarks/cleanup_candidates_gen47_20260907.json` for restore instructions.
The notebook now has a permanent **Latest candidate** button: rerun Setup once,
then click it to refresh discovery and load the newest `arena_selected.pt`.
No generation-specific path is baked into the button. Twenty notebook/catalog
tests passed. Notebook outputs preserved; never stage the notebook. Active
gen47 search/runtime sources and data were not changed by this cleanup.

Owner authorized three releases: **gen42 -> v25, gen45 -> v26, gen46 -> v27**.
Immutable release copies and manifests are under `models/bootstrap_v25..v27`.
The bootstrap champion pointer and legacy gate bar now select v27. Historical
generation states/verdicts are untouched. Gen46's nonbinding transfer checks
are still running; do not change `src/*.py` or their pinned tools mid-run.

**Gen47 revised recipe and queue:** [GEN47_RUN.md](../experiments/gen47/GEN47_RUN.md).
Run name `gen47_stateful_mixed_v3`, queued after `gen46_transfer_checks`:
full tests with the lease free -> real 28-game end-to-end rehearsal -> gen47
production -> nonbinding transfer checks. An execution error stops the chain.
No automatic release promotion. No assistant polling loop.

5,600 new games: 2,080 free, 1,600 fresh mixed-teacher prefixes, 1,120 league
(560 each teacher color), 800 full-state deep continuations (60% Black starts).
Full-state reanalysis and transitive source-family splitting are opt-in tools
adapters, leaving the active gen46 runtime unchanged. Training stays scratch
seed3173, 30/patience10, replay8, 80k sampled/40k retained deep targets.
Use `tools/start_gen47.py` for the queue chain and resume; do not bypass the
adapter by invoking plain `src/iterate.py` to resume gen47.

`src/config.py` and `src/iterate.py` retain their v24 fallback strings while
gen46's runtime is frozen. Actual gen47 uses explicit v27; bootstrap default
resolution uses the v27 champion pointer. Update legacy CLI fallbacks only
at a safe boundary. Never stage the notebook or the local next-steps handoff.

**For the next agent or developer.** Read this first, then `CONTEXT.md` §2
(durable reference) and §5 (the laws that will bite you). `REPORT.md` is the
evidence log; `DIRECTIVE.md` is a completed scope record, not a plan.

---

## 1. Where things stand

**Current run: gen46**, `gen46_expanded_sampled_v3`, full iteration with explicit
gen45 epoch-11 teacher (`models/candidates/bootstrap_main_gen_0045/arena_selected.pt`).
`gen46_transfer_checks` is queued behind it for nonbinding matched-start comparisons,
direct gen42/gen44 matches, replay census, a reproducible half-data game manifest,
and both known human-game diagnostics. See `GEN46_RUN.md` for exact commands,
counts, seeds and boundaries. No repeated assistant polling, no automatic promotion.

Gen45 finished September 6 at 19:45, `passed_not_promoted`. Its two 400-game
gen44 legs scored 79.25% / 82.125% overall, White 92.25% / 91.75%, Black
66.25% / 72.5%. The later 200-game gen42 match scored 79.5% overall, White
66%, Black 93%. Self-play: 78 White wins, 29 Black wins, 93 draws (actual-color
White score 62.25%). These sampled scores are not perfect-play measurements;
gen44 H2H endpoint-uniform diagnostic was 58.97%, versus 80.6875% sampled.

| | |
|---|---|
| **Release** | `models/bootstrap_v24/best_value_net.pt` — generation 30, promoted 2026-08-22 |
| **Prior tournament leader** | `gen42` (`models/candidates/bootstrap_main_gen_0042/screen_nominee.pt`) — leads both September 4 ladders; gen44 now has strong direct results below |
| **Newest model** | `gen45` epoch 11 (`models/candidates/bootstrap_main_gen_0045/arena_selected.pt`); passed both gen44 confirmations; gen46 generating |
| **Gate instrument** | **sampled free play v3** — `tools/gate_sampled.py`, fixed counts with separate dedup diagnostics. `gate_free.py` retains v2; book gate remains legacy |
| **`gate.BAR`** | `vs_v24`. Read it, never infer it |
| **Replay window** | **8** (`--replay-generations 8`) so replay spans gen36+ only |
| **Anchor corpus** | **dropped** (`--anchor-data none`) |
| **Suite** | 794 passing, 3 subtests; 172 existing PyTorch deprecation warnings |
| **Branch** | `main`. Commits in the owner's name, **no `Co-Authored-By` trailer**. Never push unless asked |

Never stage `src/play.ipynb` — its outputs are session noise.
Never stage `NEXT_STEPS_HANDOFF_20260905.md` — intentionally local handoff.

Implementation update: [FREE_GATE_PROTOCOL.md](../protocols/FREE_GATE_PROTOCOL.md) describes
the repaired v2 gate, durable logs, production recipe and pipeline integration.
Original gen44 evidence is preserved; a separate rescore gives 61.59% on the
combined endpoint union but inadequate unseen White confirmation coverage.

September 6 measurement clarification: [SAMPLED_GATE_PROTOCOL.md](../protocols/SAMPLED_GATE_PROTOCOL.md)
defines the new, separately versioned fixed-sample instrument. Duplicate opening
draws do not by themselves establish dependence; deduplication changes the
estimand. The completed v2 campaign is preserved unchanged. Its gen42 H2H
legs scored 67.39% and 67.23% sampled, with Black 64.23% and 62.43%. Gen41:
77.22% overall (77.30% White / 77.13% Black), 586 games. V24: 71.69% overall
(68.01% White / 75.37% Black), 544 games. Self-play: 260 White wins, 188 Black
wins, 114 draws; actual-color scores 56.41% / 43.59%, 562 games.

The original binding verdict stays INCONCLUSIVE because v2 endpoint quotas
were not met. This is not a model rejection. Against gen42's actual Black
self-par of 66.91%, pooled gen44 Black is 3.58 points lower, nominal 95% delta
interval [-7.97, +0.81] points. Overall/White strength is clear; Black improvement
against gen42 is not established. No pointer has changed.

Completed campaign: `gen44_depth3200_v2`, launched through `tools/runs.py` after the
implementation/rehearsals. It runs gen42 par + two H2H legs (180-minute budget),
then gen41, v24 and self-play diagnostics (50 minutes each), all sequentially
at 3,200 simulations with eight workers. Output:
`benchmarks/free_gate/gen44_depth3200_v2/`; log `logs/gen44_depth3200_v2.log`.
It finished around 02:57 Eastern September 6 and did not promote anything.
The final retrospective audit is `benchmarks/gen44_depth3200_v2_final_audit_20260906.json`.
Real-tool rehearsals and completed-work resumes passed. The 72 held-out human
probes finished: gen44 avoids the turn-8 queenside retreat, ending on e4 at
both 3,200 and 6,400 simulations; this is diagnostic, not proof of a winning
position. See `REPORT.md` section 53.

Completed at 04:29 September 6: `gen44_checkpoint_3200_v3`, log `logs/gen44_checkpoint_3200_v3.log`,
report `benchmarks/gen44_checkpoint_3200_v3.json`, separate output
`models/candidates/bootstrap_main_gen_0044/screen_nominee_v3.pt`. Epoch 9 was
selected again: 73.5% against gen42 in its 200-game full screen, versus 71.0%
for epoch 10 and 66.25% for epoch 11. These are selection results, not binding
confirmation.

**Gen45 launched at 11:58 Eastern September 6**, run `gen45_expanded_sampled_v3`,
log `logs/gen45_expanded_sampled_v3.log`, state `iterations/gen_0045/state.json`.
Owner requested more data; the launched increase is 2,000 free + 800 book-seeded
games, and 40,000 sampled / 20,000 retained deep teachers (60% Black). Keep
scratch training, seed 3173 and the other learning settings unchanged. Use
explicit run overrides; global defaults remain the gen44 workload. See the
gen45 revision in `SAMPLED_GATE_PROTOCOL.md`. The earlier dry run validated
replay sources 37-42 plus 44, with 45 new; the expanded production dry run
validated the same sources before launch.
No numbered release or working pointer changed. Avoid repeated status polling.

The full unattended chain is generation -> resumable deep reanalysis -> processing
and audit -> replay composition -> scratch training -> bounded checkpoint screen
-> advisory offline comparison -> sampled binding gate against gen44 epoch 9.
A PASS adds 200 self-skew games at 3,200 sims and ends `passed_not_promoted`.
A measured FAIL/INCONCLUSIVE ends normally without promotion; an execution error
records the failed phase and stops safely, without asking the owner or silently
changing the experiment. There is no assistant polling loop or automatic repair.
Source files are frozen while the run is active to preserve runtime provenance.

Preflight: 782 tests and 3 subtests passed (172 existing warnings). An isolated
real rehearsal, `iterations/rehearsal_sampled_gen45_20260906/gen_0001`, completed
generation, reanalysis, audit, composition, one-epoch training, checkpoint screen,
offline advisory and all three sampled-gate legs in about 98 seconds. Its weak
8-sim/one-epoch model was correctly rejected; this was a plumbing check, not a
strength result. Production estimate is roughly 7-9 hours: about 2.5 hours for new
data and reanalysis, 1-2 hours training, and 3-4 hours selection/gating/diagnostics.
Training allows up to 30 epochs with patience 10; gen44 stopped after 19.

---

## 2. The result that reframes everything: the 2026-09-04 round robin

Ten models, 45 pairings, 90 legs, **36,000 games**, 46 hours. Every pairing
played **twice**: a book leg on one identical opening block, and a free leg
deduped on opening state. Ratings by weighted least squares on the logit scale,
verified against synthetic data before use (recovers known ratings to 0.85 Elo).

Anchored so **v21 = 1000** (v21→v22 is +106 book, +88 free, both measured):

| model | book | free |
|---|---:|---:|
| gen42 | **1353** | **1781** |
| gen41 | 1342 | 1750 |
| gen40 | 1325 | 1741 |
| gen36 | 1333 | 1713 |
| gen38 | 1343 | 1699 |
| v24 | 1325 | 1564 |
| gen33 | 1335 | 1558 |
| gen26 | 1314 | 1535 |
| v23 | 1141 | 1254 |
| v22 | 1106 | 1088 |
| v21 | 1000 | 1000 |

Human play is **below v21** by the owner's account, with no measured match, so
it cannot be placed on either scale — only bounded.

**The tier split is the finding.** On free, gen36/38/40/41/42 sit **135–246 Elo
above** v24/gen33/gen26 — more than 6 SE. Book compresses that same structure
into 8–28 Elo, inside its own noise. **Five consecutive generations were
recorded as failures by an instrument that could not see what they improved.**

gen36 is the sharpest case: it **failed** its book gate at 400 sims against
gen33, is **level** with gen33 at 3200 on a book, and beats it by **+164 Elo**
on free.

**Within a tier nothing is separated.** The top five span 20 book Elo against
8.7 SEs. "The top five are tied" is the conclusive answer, not a failed one.

---

## 3. Owner decisions now in force

1. **Free play decides.** Book is retained for continuity, not for verdicts.
2. **Gates run on free play.** Existing v2 evidence uses endpoint-uniform scoring;
   the September 6 successor uses fixed sampled counts with dedup diagnostics.
   Never silently mix the two instruments or rewrite old verdicts.
3. **Corpus is generation self-play only.** No human games, no v19-era anchor,
   no outside data. Replay reaches back only to gen36.
4. Promotion remains the owner's call and wants a playtest.

---

## 4. Measurement rules that will bite you

These are not style preferences. Each cost real time to learn.

| rule | evidence |
|---|---|
| **Never compare a per-colour score to 0.50** | Block colour-bias is ±0.056. Read against a measured par |
| **Free-play par is not 0.50 and is model-specific** | v24 scores White **0.8717** against itself; gen33 0.7933; gen38 0.5833 |
| **A book match carries ~20 Elo of block noise the SE hides** | Same pairing, different blocks: 19.3 and 23.7 Elo apart |
| **Disjoint blocks for independent samples; MATCHED blocks for comparisons** | Giving every cell its own block breaks the comparison you built the run for. Cost this project two runs |
| **Report sampling and coverage separately** | Repeated independent opening draws carry probability mass; dedup estimates a different quantity. Endpoint collisions are common, but histories can differ. V2 is endpoint-uniform; v3 is sampled |
| **One book line is n=1** | ~45% of per-line verdicts flip on resampling. Use `--book-temp-plies`; quote aggregates, not cells |
| **Depth changes values** | Line values reproduce at r≈0.83–0.90 within a depth, r≈0.36–0.67 across. A score at one sim count is a statement about that sim count |
| **Free play is non-transitive** | Round-robin RMS residual **73.5 Elo** free vs 12.6 book. Good tier detector, poor ordering device |
| **Never chain Elo** | v24 and gen26 sit 55 Elo apart via v22 and are **level** head-to-head. Three anchored claims were overturned by direct play |
| **Existence is not completion** | Killed runs leave truncated artifacts that resume logic accepted as done. Validate, don't `stat` |

---

## 5. Tooling added since the last handoff

| tool / flag | what it does |
|---|---|
| `tools/gate_free.py` | free-play gate: par leg (bar vs itself, cached), bar leg, confirmation replay. Stops on unique games or budget |
| `tools/match.py --book-temp-plies N` | samples N plies after each book position so a repeated entry yields *different* games — the only way to error-bar a single line. Also splits the pair seed when sampling |
| `tools/match.py --game-log` | per-game JSONL incl. the opening record, which is the exact dedup key for free play |
| `src/iterate.py --anchor-data none` | drops the v19-era anchor so the corpus is generation self-play only |
| `tools/model_report.py` | self/anchor × book/free × named predecessors × sim levels |
| `tools/export_lines.py` | replays named lines to scrubbable HTML. **Two engines, one per colour** — native MCTS reuses its tree, and sharing one leaks White's tree into Black's search |

---

## 6. Open questions

- **gen44 checkpoint choice** — the 3,200-sim epoch-9 campaign is complete;
  independent gen44 saved-epoch nomination is next. The v2 coverage verdict
  remains inconclusive; do not rewrite it as a binding sampled PASS.
- **Why gen44 improved** — not isolated: teacher identity, replay volume,
  generation/teacher counts and seed changed along with anchor removal.
  Gen42 also overfit. Restoring the old human-containing anchor violates the
  current generation-only policy and is not the planned control.
- **Top-five ordering** unresolved and probably unresolvable at practical
  sample sizes. Deciding among them may need a criterion other than strength
- **The book gate's blind spot** — gen36 was rejected by it and belongs to the
  stronger tier. Any candidate rejected by a 400-sim book gate since gen33
  deserves re-examination
- **`iterations/gen_0043`** is an abandoned stub with state/lock metadata from a killed run; `iterate.py`
  treats directory existence as "generation taken", which is why gen43 was
  skipped and the model is gen44

---

## 7. Operating rules

- Long jobs via `py -3 tools/runs.py start --name X -- ...`, never blocking
- **Never run concurrent worker jobs.** 3×8 workers froze this box; a
  DPC_WATCHDOG_VIOLATION (0x133) hit on 2026-08-31 during a 12-worker match.
  Match workers are now **8**
- Never build Python scripts in bash heredocs — `\n` mangling has cost time
  twice
- Never weaken a gate threshold to let a recipe through
- Estimate from the running job; never extrapolate a rate across workloads
