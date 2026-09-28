# Monster Chess NN — consolidated handoff, September 25, 2026

This is the current entry point for another developer or agent. It consolidates
the inspected working tree, saved evidence, and the owner's requirements.
Research last completed September 17; September 25 work was cleanup,
verification, documentation and the v28 promotion, **not another training run**.
The chronological log is `docs/history/HANDOFF_LOG.md`; it and `CONTEXT.md`
remain useful, but their old “running,” “next,” and “current release”
statements are not current status.

**Later on September 25:**

- **v28 promoted** (owner: “promote a new v from either 49 or 50”; chose
  gen50 + calibrated value). `models/bootstrap_v28/best_value_net.pt`, SHA256
  `b651e740…e35a` = the calibration continuation candidate below. Pointer
  `models/bootstrap/champion.json` and `tools/gate.py` `BAR` now name v28;
  previous pointer saved as `champion_before_v28_20260925.json`. Promotion
  tool `tools/promote_v28.py`; manifest records evidence hashes and caveats.
  The notebook catalog now lists the hash-verified champion first.
- **Snapshot commit `b46ce1c`** captured all September source, tests, native
  code, docs, books and small benchmark evidence (no weights, arrays, per-game
  task records or JSONL journals). **Root reorganized** afterwards: plans and
  results in `docs/experiments/<campaign>/`, protocols in `docs/protocols/`,
  old ledgers in `docs/history/`, finished root drivers in `campaigns/`
  (frozen; resume via a worktree at `b46ce1c`). Two superseded docs deleted
  (`NEXT_STEPS_HANDOFF_20260905.md`, `BOOTSTRAP_FOLLOWUPS.md`); see `docs/README.md`.
- **Next step: better play.** `docs/plans/GEN51_STRENGTH_PLAN.md` (proposed,
  not queued): audit and gate v4 with a binding 12,800 guard, a v28
  search-constant check, then gen51 with teacher v28 and a shared two-arm data
  pool testing deep disagreement-continuation value targets. The shippable
  engine plan is deferred; when it resumes, the owner prefers hosting on this
  box behind a move API.
- Measured Sept 25: distinct value-row inputs fell from 62.4% (gen49) to 53.0%
  (gen50), and the 100 most repeated positions are 14.5% of gen50 value rows.

**September 28, 18:03: gen52 COMPLETE, no gen53 launched. Nothing running.**
See `docs/experiments/gen52/GEN52_RESULTS.md`.

- **Neither gen52 arm passed gate v4** against its teacher (gen51
  deep-value). Both got weaker as White (Arm A −26 pp, Arm B −13 pp).
- Held-out means (B2, v27): Arm A 80.9%, Arm B (pool) 91.4%, teacher 92.7%.
  The pool helped on held-out opponents but did not beat the teacher.
- Arm A drew most games against B2 (65.9%). Hypothesis, untested: the
  deep-value share of value training nearly doubled (13% → 22%) because
  gen51's source rolled forward next to gen52's.
- By the pre-declared rule 1 in `docs/plans/GEN53_PLAN.md`, **gen53 was not
  launched**. The driver `tools/gen53_campaign.py` is ready; the teacher
  decision is unwritten.
- **Best model on the evidence: gen51 deep-value nominee**
  (`models/candidates/bootstrap_main_gen_0051_deepvalue/arena_selected.pt`),
  unpromoted and awaiting the owner's playtest.
- Owner options in GEN52_RESULTS.md: (1) cheap retrain of Arm B with the
  deep-value share capped, about 10–11 h; (2) gen53 with lessons applied,
  about 36 h; (3) release decision on gen51 deep-value first.

**September 27, 03:08: gen52 production RUNNING** *(historical; completed
September 28 18:03)* (managed run
`gen52_production`; log `logs/gen52_production.log`; evidence
`benchmarks/gen52_program/gen52_20260927/production/`; plan
`docs/plans/GEN52_PLAN.md`).

- **gen51 finished 03:01 (32.7 h): both arms passed gate v4 against v28.**
  See `docs/experiments/gen51/GEN51_RESULTS.md`.
  - Control arm: 79.3% at 3,200, 95.6% at 12,800.
  - Deep-value arm: 74.9% at 3,200, 85.0% at 12,800.
  - Both about 99% against the held-out B2, where v28 scored 73.75%.
  - The control arm has a White hole (e4+d4 …d5 c4+Ke2) that gen49 exploits:
    47.2% vs gen49, against v28's 83.1%.
  - The deep-value arm holds that line (75.6% vs gen49) and beat the control
    67.8% at 12,800 (50.1% at 3,200).
  - **Nothing promoted.** Both nominees are playtest candidates:
    `models/candidates/bootstrap_main_gen_0051{,_deepvalue}/arena_selected.pt`.
- **gen52 (overnight authority from the owner):**
  - Teacher = gen51's deep-value nominee, by the rule declared before the
    arm-vs-arm result (`docs/plans/gen52_teacher_decision.json`).
  - Shared generation with 30-ply exploration and the deep-value source.
  - **Arm B adds 1,200 games against the owner's pool (v28, gen49, gen48,
    v26); B2 and v27 are held out.**
  - Full rehearsal passed in 4.9 minutes.
  - Guide: about 35 hours (gen51 took 32.7). Pause and resume exactly as for
    gen51 below, with name `gen52_production` and `tools/gen52_campaign.py`.

**Evening of September 25: gen51 production RUNNING** *(historical; completed
September 27 03:01)* (managed run
`gen51_production`, launched 18:20:40 Eastern; log `logs/gen51_production.log`;
evidence `benchmarks/gen51_program/gen51_20260925/production/`).

- Before it: Stage 0 done (gate v4 `tools/gate_depth.py`; diversity audit
  `docs/experiments/gen51/DIVERSITY_AUDIT.md`: narrowing concentrated and
  accelerating). Stage 1 done, **null** (no c_puct/FPU variant beat the
  defaults; `docs/experiments/gen51/SEARCH_CONSTANTS_RESULTS.md`). Owner
  approved gen51 self-play exploration of 30 plies. Full rehearsal passed in
  4.1 minutes, including zero parent/continuation split mismatches.
- The chain (`tools/gen51_campaign.py`; details in the plan's §7):
  1. canonical gen51 iteration through `train` (control arm);
  2. 768 disagreement roots × 2 continuations at 6,400;
  3. strict-label value-only increment (weight 4) and deep-value training;
  4. selection for both arms, with 12,800 probes of the top three epochs;
  5. gate v4 and diagnostics per arm, and an arm-vs-arm match if both pass.
  Guide: roughly 30–40 hours. Nothing is promoted automatically.
- **Pausing:** `py -3 -B tools/runs.py stop --name gen51_production`, then
  relaunch the same command later (`py -3 tools/gen51_campaign.py` via
  `tools/runs.py start`). Completed stages are receipt-checked and skipped. An
  interrupted *training* stage is retained and refused rather than silently
  restarted: inspect it before resuming. **Do not edit** any file in
  `tools/gen51_campaign.py`'s `PINNED` list or any `src/*.py` while it runs;
  the identity check between stages will stop the chain.

## 1. Executive state

- **Public release: v28** (September 25), gen50 epoch14 with the calibrated
  value head. v27 (gen46 epoch 7, September 7) is the previous release; v25
  came from gen42 and v26 from gen45. Other generation numbers are research
  candidates, not numbered releases.
- **Canonical gen50 checkpoint: epoch 14**, still at
  `models/candidates/bootstrap_main_gen_0050/arena_selected.pt`.
- **Latest experiment: frozen-policy value calibration, complete.** Its
  continuation-trained value head scores **58.4375% over 800 games against
  unchanged gen50 epoch14 at 3,200 simulations**, but **49.375% over 160 games
  at 12,800 simulations**. This is an ordinary-budget improvement, not an
  established general or all-budget upgrade. Nothing was promoted.
- Gen50 epoch15 was a previous checkpoint-recovery nominee, not a replacement
  for epoch14: strong Black results concealed a serious deeper-search White
  regression.
- No active managed training/benchmark job or queued follow-up was found on
  September 25. The two running Python processes were a VS Code notebook kernel
  and its interrupt helper; they were left alone. Do not infer an active run
  from historical PIDs, lock-file existence, or old status prose.
- Last commit is `9aca5ee` from September 7. Substantial later source, tests,
  plans, and evidence are **untracked or modified**. They are not backed up merely
  because this is a git repository. Existing book deletions predate this cleanup.
  No commit, push, promotion, or model/data deletion was performed today.

The immediate objective remains a reliably stronger engine on **both colors**,
with particular attention to Black conversion, not simply beating a predecessor
under one test instrument. The long-term objective is a self-improving training
loop and eventually a downloadable/website-usable engine.

## 2. Owner requirements and working style

- “PIW” means **plan → implement → wait**. Put a bounded plan and fixed test
  schedule in writing; rehearse the complete chain before an overnight run.
- Time budgets such as 8–10 hours have been guides, not permission to truncate
  required tests. Use long completion waits and milestone checks, not continual
  assistant polling. A background log watcher is not an assistant that can
  independently diagnose and edit code.
- Do not run heavy workloads while the owner is playing. Keep one heavy GPU
  stage at a time, normally at most eight workers, targeting at most about
  12 GiB of the 16 GiB GPU. Do not alter unrelated applications under contention.
- Do not hard-code opening bans, pawn-capture bonuses, or simple tactical
  “givens” to patch a particular human game. Prefer general search/data/value
  changes supported by controlled evidence.
- Normal-start model-chosen openings are the primary playing instrument.
  Different RNG seeds do not guarantee different strategic structures.
  Do not force variety or deduplicate the primary score to manufacture novelty.
  Repertoire concentration does not establish that other openings are unsound.
- All candidate arms must receive some play-testing. Offline accuracy/MSE is
  advisory; it is not sufficient for rejection or promotion. Conversely, testing
  every saved epoch exhaustively is unnecessary: bounded probes and screens
  precede independent confirmation.
- Never weaken gate thresholds after seeing results. A measured FAIL should
  not prevent the other predeclared diagnostic matches; an execution error
  should stop the chain safely.
- A public version needs automated evidence **and the owner's playtest/approval**.
  Passing a gate does not authorize automatic promotion or a new generation.
- Preserve interrupted training rather than silently restarting over its output.
  Preserve valid earlier work when repairing a chain; use a new namespace when
  semantics or pinned inputs change.
- Commit/push only when requested. Historical commit preference is the owner's
  identity (`Peng-AP`), one-line messages, no co-author trailer; verify local
  configuration rather than inventing an identity. This request did not ask for
  commits or a push.

## 3. Rules, perspectives, and engine invariants

White starts with a king and c2–f2 pawns and makes two primitive moves per turn.
Black has the standard army and makes one. A **king capture ends the game
unconditionally**; ordinary chess checkmate is not the terminal rule. Capturing
the enemy king wins even if the capturing side's king would otherwise be attacked.
White's first half-move may traverse check; the completed turn follows the
engine's king-safety legality contract, including its established forced-blunder
exception. White has no castling. En passant after White is granted only by the
last move of its turn. See `src/monster_chess.py` and the rules tests for exact
semantics rather than substituting standard-chess assumptions.

White's two search plies belong to the **same player**: do not negate a value
between them. Network and native search values are side-to-move values. A White
outcome remains the same sign for both White halves and flips for Black.
State reconstruction requires FEN **plus** `white_half_pending`, turn count,
clocks and history; FEN alone is not a complete experimental starting state.

In current match evidence, king captures are wins/losses and repetition or
turn-limit endings are draws. Legacy generation may use shaped cap labels;
never silently carry those into captures-only match scores. A long draw is not
proof of a fortress or of perfect play. The September calibration specifically
uses strict completed capture outcomes, not the older distance-tempered labels.

Search uses factorized White half-moves, policy-guided batched PUCT, and native
Rust move generation/tree operations with GPU neural evaluation. The Python
engine remains a reference and fallback, not disposable legacy code. Existing
king-safety overrides, finisher, tree reuse, and selected-child value reporting
are part of the measured engine. Pure root probes sometimes disable early stop
and finisher and use fresh trees; their results are not identical to played games.

Important recent correction: the original September 15 conditional study shared
a search tree across colors when the models were equal. It was stopped and
preserved as invalid for comparison. The corrected `mainline_study.py` uses
separate per-color engines even for selfplay, with full driver-history replay.
Do not pool `mainline_counterplay_20260915` with the corrected `_v2` results.
Native recent-history limitations were not silently changed by these studies.

## 4. Repository map and environment

| Area | Purpose / important entry points |
|---|---|
| `src/monster_chess.py`, `encoding.py` | Python rules and position encoding |
| `src/mcts.py`, `native_mcts.py`, `evaluation.py` | Reference search, Rust bridge, GPU evaluation |
| `native/src/` | Rust rules, MCTS, optional tactical/CPU search experiments |
| `src/train.py`, `data_generation.py`, `data_processor.py` | Main network training and data conversion |
| `src/iterate.py`, `tools/iterate_stateful.py` | Resumable iteration and stateful data adapter |
| `tools/stateful_generation.py`, `reanalyze_coverage.py` | Parent-linked continuations and deep teacher coverage |
| `tools/match.py`, `gate.py`, `src/match_evidence.py` | Playing evidence, gates, runtime/hash provenance |
| `tools/mainline_study.py` | Audited conditional games and probes with restored histories |
| `tools/runs.py` | Start/status/tail/archive managed jobs |
| `src/model_catalog.py`, `src/play.ipynb` | Model discovery and notebook play |
| `campaigns/` | Finished Sept 15–17 drivers (gen50, recovery, mainline extension, value calibration); frozen records, see its README |
| `campaigns/value_calibration/value_calibration.py`, `campaigns/gen50_recovery/recovery_probe.py` | Frozen-head fitting and diagnostic policy/value crossover |
| `tests/` | Contract tests (`py -3 -m pytest tests`); campaign tests stay with their frozen drivers |
| `data/raw/`, `data/processed/` | Recorded games, immutable increments and composed replay tensors |
| `iterations/` | Generation state, manifests, journals and selected-checkpoint reports |
| `models/` | Releases, candidates, retained research weights; gitignored |
| `benchmarks/` | Evidence **and some actual trained checkpoints**, not disposable reports |
| `logs/archive/<date>/` | Completed run logs and metadata, retained after cleanup |
| `books/`, `data/start_fens/` | Historical/stress openings; not the current primary test instrument |

The working machine is Windows/PowerShell, Ryzen 5700X, RTX 5060 Ti 16 GB,
32 GiB RAM. Python is installed at
`C:\Users\perfp\AppData\Local\Programs\Python\Python313\python.exe`.
The normal launcher is `py -3`. The September 25 restricted sandbox reported
“No installed Python found!” although the installation works outside that
sandbox. A launcher/permission failure is not evidence Python was uninstalled.
Request the appropriate execution permission rather than reinstalling packages.

Runtime dependencies are in `requirements.txt`; notebook dependencies are listed
there too. Pytest is used by the newer campaign suites even though not listed as
a core runtime dependency. Preserve the working local environment and identify
versions before upgrading it, because runtime files are provenance-pinned.

Native build/install, when actually needed:

```powershell
powershell -File tools/build_native.ps1
```

This requires the existing MSVC Build Tools/Rust toolchain, builds the release
DLL and copies it to `native/monster_native.pyd`. Do not rebuild just to inspect
a completed run: replacing the binary changes runtime identity. The installed
`.pyd` and historical rollback `.pyd` files were preserved during cleanup.

## 5. Models and identities

Paths below are relative to the repository. An `arena_selected.pt` identity is
more meaningful than `best_value_net.pt`; selection can choose a different epoch
from the offline-best checkpoint.

| Model | Exact path | Role |
|---|---|---|
| **Public v28** / gen50 epoch14 + calibrated value | `models/bootstrap_v28/best_value_net.pt` | Current release and gate bar (Sept 25) |
| Previous v27 / gen46 epoch7 | `models/bootstrap_v27/best_value_net.pt` | Previous release |
| B2 CNN epoch8 | `models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt` | Older diagnostic opponent; extensively studied, no longer blind |
| Gen48 epoch17 | `models/candidates/bootstrap_main_gen_0048/arena_selected.pt` | Previous GPU mainline candidate |
| Gen49 epoch7 | `models/candidates/bootstrap_main_gen_0049/arena_selected.pt` | Gen50 teacher/reference |
| Gen50 epoch14 | `models/candidates/bootstrap_main_gen_0050/arena_selected.pt` | Retained canonical gen50 selection |
| Gen50 epoch15 | `models/candidates/bootstrap_main_gen_0050/selected_epoch_015.pt` | Recovery nominee; no promotion |
| Calibration continuation | `benchmarks/value_calibration_20260917/production/fits/continuation/candidate.pt` | Latest ordinary-budget research improvement |
| Calibration replay control | `benchmarks/value_calibration_20260917/production/fits/replay/candidate.pt` | Failed nomination control, retained |

SHA256 identities:

```text
v27             976294daf7e3d6f0c51c358dd602f11997c7fdf2dc4b255b810b588c253e5459
B2              fc23076a9f5c7f237785f27cb1a665c10588ea8e8916cd743016d19a96999d15
gen48           a8c074390c93390ac974b1f58076a86aa0525d9a34f7cff66e12442bb7e07722
gen49           4bcc68a0219acf8c3dc53326d738789e6bd767fccba88fd4f471345567b4a647
gen50 epoch14   51b5ddb01db51ae9023eaaf8ccbd896b48a805a52b2707633dc1d7e3f8067f25
gen50 epoch15   85e5d01132f3c69ccd10b381b293476ef7b9fad56034fdedc57312c849f1803c
continuation    b651e7405afe4e5672676c6fb13bb5bc6e1f5ea0071fd5ce4f4d5318e229e35a
replay control  98a68b982a0a30f1dc928fdb40944e6d4b799bba9f3dce2024fefe6dab6f916b
native .pyd     8b3b72e1bdde1c3cb0ab5454bfbb81a6db19bf1a450acb2776ad1bb081dc3099
```

Notebook discovery scans the model catalog, not arbitrary benchmark directories.
The calibration checkpoint is therefore not automatically a dropdown entry.
It is an ordinary compatible checkpoint usable by an explicit model path; its
absence from the dropdown does not mean it was not trained. Do not silently
copy it over `arena_selected.pt` or the public release to make it selectable.

## 6. Latest base-generation recipe

The current gen49/gen50 family is the existing **15-plane** CNN: stem64,
two 64-channel blocks then six 128-channel blocks, attention-policy width64,
scalar global-average-pooled value head. Gen50 does not use WDL, moves-left,
or SE additions. Generic README defaults and the optional B2 architectures
are not a description of this checkpoint.

Gen50 (`campaigns/gen50/gen50_recipe.json`, `docs/experiments/gen50/GEN50_PLAN.md`) used:

- Frozen gen49 epoch7 teacher; 2,800 normal-start games at 1,600 simulations.
- 400 parent-linked completed continuations at 12,800 simulations, raised
  from gen49's 6,400. No forced prefix pool or league opponents in the new
  increment. All game outcomes retained.
- Reanalysis: 24,000 sampled positions, 12,000 retained, 12,800 simulations,
  60% retained Black, family limits/coverage retained. Policy teachers get
  multiplier four and zero teacher value weight; a search value is not an
  observed outcome.
- Eight-generation rolling replay, no external/human anchor. Historical replay
  still contains older mixed recipes: “mainline-only” describes the new increment,
  not every old training row.
- Scratch training, seed3173, AdamW LR0.002, weight decay0.0001, batch256,
  EMA0.999, warmup3, maximum30 epochs/patience10. Value floor0.5/horizon60.
  Gen50 actually trained23 epochs; saved-epoch play selection chose epoch14.
- Training exploration remains noisy/sampled; removing externally prescribed
  openings did not turn generation into a deterministic single line.
- Parent, fork, and reanalysis descendants share the same family split.
  Do not split descendants independently or split mirror copies across holdouts.

Important files: `data/processed/bootstrap_replay_main_gen_0050`,
`iterations/gen_0050/state.json`, and
`benchmarks/gen50_20260916/production/summary.json`.
The iteration state deliberately stops after checkpoint selection and can say
`partial`; the external research gate chain completed separately. Do not
restart a completed generation merely because that canonical state is partial.
`configs/bootstrap_generation_only.json` contains older fallback defaults;
the campaign's pinned recipe and recorded CLI overrides define gen50.

## 7. Results and what they mean

Scores below count a draw as half a point. “White” and “Black” are the candidate's
scores in that color, not a percentage of decisive games. Primary normal-start
matches sample temperature0.5 for the first16 primitive plies, then choose
temperature0; no opening book. Results at different search budgets answer
different questions. Different seeds/samples are not matched causal deltas.

### Progress before calibration

| Candidate / opponent | Sims each | H2H games | Overall | White | Black |
|---|---:|---:|---:|---:|---:|
| Gen49 / gen48 |3,200|800|94.00%|94.75%|93.25%|
| Gen49 / B2 |3,200|200|83.75%|96.00%|71.50%|
| Gen49 / v27 |3,200|200|98.25%|96.50%|100.00%|
| Gen49 / gen48 |12,800|160|78.4375%|98.75%|58.125%|
| Gen50 epoch14 / gen49 |3,200|800|75.00%|56.75%|93.25%|
| Gen50 epoch14 / gen49 |12,800|160|54.6875%|46.875%|62.50%|
| Gen50 epoch14 / B2 |3,200|200|82.00%|91.50%|72.50%|
| Gen50 epoch14 / v27 |3,200|200|92.25%|84.50%|100.00%|
| Gen50 epoch15 / gen49 |3,200|800|73.9375%|50.00%|97.875%|
| Gen50 epoch15 / gen49 |12,800|160|62.8125%|26.25%|99.375%|

Gen50 improved ordinary-budget H2H chiefly through Black. Epoch15's higher
deep aggregate is misleading if considered without its White collapse.
Gen50 epoch14 also scored99.375% against B2 at12,800: cross-opponent behavior
is strongly search-budget dependent. Never chain these matchup scores into a
single implied Elo ladder or an estimate of distance from perfect play.

The earlier book-based gen48 results did not establish normal-start failure.
The subsequent gen49 campaign independently tested gen48 from the normal start:
90.75% against gen47, 76.25% against B2. Keep book/stress and normal-start
instruments distinct when reading `docs/experiments/gpu48/GPU48_RESULTS.md` and later corrections.

### Latest calibration results

Completed September17,02:19:51–09:16:27, about6h57m. Total3,696 production
games:576 for new data,960 all-arm screening,2,160 independent confirmation
(the last count includes400 incumbent self-calibration games). A separate
120-game rehearsal verified the chain.

| Continuation candidate / opponent | Sims each | H2H games | Overall | White | Black |
|---|---:|---:|---:|---:|---:|
| Unchanged epoch14, first leg |3,200|400|56.50%|42.75%|70.25%|
| Unchanged epoch14, confirmation |3,200|400|60.375%|49.50%|71.25%|
| Unchanged epoch14, combined |3,200|800|58.4375%|46.125%|70.75%|
| Unchanged epoch14 |12,800|160|49.375%|41.25%|57.50%|
| Gen49 |3,200|160|83.125%|70.00%|96.25%|
| Gen49 |12,800|160|56.875%|48.75%|65.00%|
| Public v27 |3,200|160|90.9375%|81.875%|100.00%|
| B2 |3,200|160|73.75%|83.75%|63.75%|

Actual-color selfplay, not arbitrary model-A score:

| Model / sims | Games | White wins | Black wins | Draws | White score |
|---|---:|---:|---:|---:|---:|
| Gen49 /3,200 |200|14|136|50|19.50%|
| Gen49 /12,800 |160|14|36|110|43.125%|
| Gen50 epoch14 /3,200 |200|33|97|70|34.00%|
| Gen50 epoch14 /12,800 |160|13|57|90|36.25%|
| Gen50 epoch15 /3,200 |160|7|68|85|30.9375%|
| Calibration continuation /3,200 |160|37|73|50|38.75%|

The continuation candidate passed its predeclared ordinary sampled gate.
The deeper epoch14 comparison is essentially even; older-opponent results do
not establish improved general strength. Lower value MSE and one favorable
H2H matchup are not a release decision. **Retain it as a research candidate;
do not promote or overwrite epoch14.** Full details: `docs/experiments/value_calibration/VALUE_CALIBRATION_RESULTS.md`.
*Superseded September 25: the owner promoted this candidate as v28 (a copy; epoch14 is untouched).*

## 8. Why value calibration was tried, and its limitations

Human-game investigation found problematic White continuations after
`e4+d4 ...d5`. Gen50's `c4+c5` branch scored4W/4D/59L across its two gen49
legs, and1W/1D/6L against v27. These are correlated conditional samples, not
solved lines. Other White losses against B2 involved increased alternative
first turns rather than failure of the dominant `e4+d4` line.

Checkpoint recovery compared gen49 and gen50 epochs10,13,14,15,23 with fixed
conditional roots, normal-start screens, and diagnostic policy/value crossover.
After `e4+d4 ...d5 c4`, epoch14's raw c5 prior was2.25%, but pure3,200-simulation
search assigned35.70% to c5. Keeping epoch14 policy and substituting gen49 value
reduced that to4.81%; gen49 policy with epoch14 value raised it to54.52%.
However, at51,200 simulations the preferences changed again. This implicated
value/search interaction, **not a globally correct transplanted value head**.

The next experiment froze epoch14's backbone, policy and all non-value buffers.
Only six existing `value_head.*` tensors changed; the architecture and engine
were unchanged. Three arms were tested:

1. Unchanged epoch14 baseline.
2. Replay-only value-head fit.
3. Identical fit mixing half replay and half fresh deeper-continuation outcomes.

Fresh data:192 normal-start epoch14 selfplay parents at3,200; choose one root
per family by generic model/search value disagreement, assigned96 Black,
48 White-first,48 White-second. No move-name or win/loss-based selection.
Each root gets two completed6,400-simulation continuations, swapping gen49 and
epoch14 colors. Family split144train/24validation/24test. At most eight sampled
positions per phase per game; labels are strict capture outcomes in STM
perspective, with repetitions/caps0, never raw teacher evaluations.

Old replay sampling:32,768train and4,096validation positive-value-weight rows
from the original gen50 split. Both fitted arms use `capture_results.npy`,
not the old distance-tempered target, so this label change is shared by the
controls. A gain over baseline cannot be attributed solely to new continuations.

Exact encoded-input leakage filtering gives precedence to newtest, newval,
oldval, then train. Same-input conflicting outcomes are averaged within each
pool. Final unique rows: **newtrain1,833**, newval511, newtest1,069,
oldtrain27,995, oldval3,844. Of9,872 sampled new-training rows,6,424 were
excluded for held-out overlap before remaining duplicates were collapsed.
Thus576 games do not equal576 independent structures or a large new training
set. The pretrained network may previously have seen these encodings; the
split protects the incremental fit, not historical pretraining exposure.

Training: cached128-wide frozen GAP features with full-forward parity checked;
AdamW LR1e-4/weight decay1e-4, seed26017,12epochs×128updates, batch512,
50%Black/25%each White half; continuation batches256old+256new. Loss is outcome
MSE plus0.1MSE anchoring to the original prediction. Best mean of six
phase/source validation MSEs selects the checkpoint; all arms still play games.
Replay selectedepoch10, continuationepoch12. New held-out MSE was about0.1812
and0.1621 respectively versus original predictions0.2435, but play transfer was
not uniformly better. These are imperfect players' outcome labels, not
perfect-play ground truth.

Selection screening was3arms×4opponents×80games at3,200. Continuation alone
met the fixed nomination rule; its initial epoch14 screen score was only51.25%.
The independent800-game result confirmed an ordinary-budget gain. B2/v27 were
used in this screen and therefore were not blind opponents; later seed blocks
are independently sampled confirmation, not a previously unseen opponent set.

## 9. Evaluation contracts and failure modes

Read `docs/protocols/SAMPLED_GATE_PROTOCOL.md` for the current sampled protocol. The usual
binding budget is400 incumbent actual-color selfplay games plus two400-game
H2H legs (200 candidate games per color per leg), at3,200 simulations. Each
leg must exceed50% overall and each color must remain at least incumbent
same-color self-par minus five percentage points. This is an operational
point-estimate gate, not a proof that both colors improved.

- Old fixed0.40-color-floor/book gate descriptions refer to older instruments;
  do not mix them with sampled gates or silently change a saved report's meaning.
- Keep saved-epoch selection data separate from independent confirmation.
- Preserve repeated openings at their sampled frequency. Report endpoint
  concentration/unique coverage separately; nominal game-level error bars can
  overstate strategic independence.
- For selfplay combine both model roles into actual White/Black wins and draws.
  Model-A's score is not color skew even when A and B have identical weights.
- Record both search budgets, initial position/history, seeds, checkpoint hashes,
  native binary and engine settings. A null `sims_b` can mean inherited A budget;
  strict identity checks should record the intended effective budget explicitly.
- Restore full histories in conditional tests and audit legal replay/outcomes.
  The move-count limit and repetition history affect continuation outcomes.
- A reused-tree game, fresh-tree root probe, and pure no-finisher search are
  different instruments. Never compare them as if only the network changed.
- Do not infer that a model is “near perfect” from beating older models, or that
  search is solved because a brute-force/tactical solver exhausts its budget.

## 10. Running, resuming, and verification

Use these to inspect without launching training:

```powershell
py -3 -B tools/runs.py status
py -3 -B tools/runs.py tail --name value_calibration --lines 30
Get-Content benchmarks/value_calibration_20260917/production/status.json
```

After cleanup `status` can say “no runs recorded” because completed root records
were archived. `tail` searches archives; old logs remain in dated directories.
Status is liveness, not proof of successful completion: verify final summary,
stage receipts, output hashes and required game counts.

The latest driver's normal invocation was `py -3 run_value_calibration.py` from the root of snapshot `b46ce1c` (it now lives, frozen, in `campaigns/value_calibration/`);
`--rehearsal-only` exercises the rehearsal namespace. **Do not launch it now
just to see status.** It is a frozen completed campaign, not a generic next-gen
launcher. Exact resume verifies inputs and reuses completed receipts; changing
the recipe requires a new campaign, not editing an old manifest.

Provenance hazard: the calibration identity hashes **all `tools/*.py` and
`tests/*.py`**, its root scripts/test/plan, runtime files/native binary, four
models, and gen50 replay files. Even adding an unrelated Python file directly
to those pinned directories changes identity. Several earlier campaigns have
similar broad pins. Keep old inputs unchanged for exact resume; new opt-in root
drivers were used to avoid gratuitously invalidating older runs. This is a
maintenance limitation to address explicitly in future tooling, not permission
to weaken old checks retroactively.

Production evidence layout:

```text
benchmarks/value_calibration_20260917/production/
  manifest.json, status.json, summary.json
  parents/, continuations/         audited completed game tasks
  roots.json                      selected states, full prefixes, provenance
  data/                           split feature/label arrays and family IDs
  fits/replay/, fits/continuation/ trained checkpoint and complete.json
  screen/play/, screen/receipts/   all-arm matches and output hashes
  nominee.json                    fixed selection result
  confirmation/play/vs_initial/   self-par, two H2H legs, gate report
  confirmation/play/              gen49/v27/B2/self/deep matches and journals
  confirmation/receipts/, receipts/
```

Recorded September17 prelaunch validation:983tests +3subtests, complete
120-game rehearsal/two fits,26 rehearsal receipts, resume test and frozen-policy
checks. Do not present those as a fresh full-suite run on September25.

September25 evidence audit rehashed25 production receipts/82 referenced outputs,
four frozen models,32 runtime files,286 implementation files and three smaller
replay inputs: **zero missing files or mismatches**. It intentionally did not
rehash the15,169,843,328-byte replay `positions.npy`, replay every game, or
independently tensor-compare the heads again. Saved hash-verified fit artifacts
record exact non-value tensor/buffer and raw-policy equality.

Focused current verification and the exact cleanup inventory are recorded in
`docs/history/CLEANUP_20260925.md`. No GPU benchmark was restarted for this documentation task.

## 11. Cleanup performed and what remains

Removed regenerable `native/target` and Python/pytest caches. The installed
native extension was preserved and its SHA256 checked. Cargo intermediates can
be rebuilt; Python/test caches regenerate. Archived174 completed run records
(348 metadata/log files) plus the old March root log without deleting their
content or overwriting archive destinations. See `docs/history/CLEANUP_20260925.md`.

Kept all model weights, training data, iteration journals, benchmarks, rehearsal
evidence, scripts and plans. “Untracked,” “old,” or “rejected” does not imply
disposable. In particular, the newest candidate lives under `benchmarks/`.
Historical native binary backups were retained. Existing tracked book deletions
and other dirty source changes were neither reset nor attributed to this pass.

Most disk space is not cache: approximately285GiB of data,15.2GiB iterations,
14.6GiB models at audit time. Further bulk reclamation needs a separate inventory
of immutable raw/increment inputs, replay composition manifests, active pins and
reproducibility requirements. Old composed replay tensors may be regeneration
candidates, but were not deleted speculatively. There is no blanket promise that
git contains these large artifacts or even all the current source.

## 12. Concrete next research plan — proposed, not queued

The latest experiment isolates a useful value-side direction but does not
justify another architecture rewrite, promotion, or a move-specific patch.
Before spending another full generation:

1. **Audit the information content of the new targets.** Quantify family/input
   concentration, conflicting outcomes and phase coverage after leakage removal.
   Explain the1,833-row effective training set and avoid mistaking more duplicate
   trajectories for more supervision. Use existing artifacts first; do not
   recycle held-out gate games as training data.
2. **Specify one small conservative value-update experiment.** Keep the ordinary
   checkpoint format, policy/backbone frozen and unchanged epoch14 control.
   Candidate hypotheses include stronger anchoring/smaller updates and repeated
   independent continuations to estimate target variability. Select a bounded
   set of arms and label controls before generating data; the exact recipe is
   not approved or implemented by this handoff.
3. **Preserve causal controls.** If outcome target representation changes,
   include an old-data-only control with the same labels. Keep family-linked
   splits and exact-input exclusions, report effective unique counts, and never
   treat imperfect search outcomes as perfect-play labels.
4. **Predeclare multi-budget playing criteria.** Screen every arm, then freeze
   one nominee and run two independent3,200-simulation legs versus its actual
   initialization. Include12,800 comparison to that initialization, gen49,
   older-opponent diagnostics, and actual-color selfplay. Independent RNG blocks
   remain frequency-weighted normal starts. Both-color transfer matters; do not
   let a larger Black score conceal a new White collapse.
5. **Rehearse and chain everything before the overnight.** New immutable output
   namespace, source/model/data identities, no interrupted-fit overwrite,
   all post-selection diagnostics even after measured gateFAIL, one heavy job,
   <=12GiB target, infrequent health checks. No automatic promotion or next run.
6. **Only then consider a larger new generation.** Advance a frozen teacher and
   data recipe because multi-opponent/multi-budget evidence supports it, not just
   because the newest candidate beats its predecessor. Return to CPU/architecture
   work only with a concrete hypothesis that the present GPU evidence cannot
   address economically.

No time estimate here is a promise: the completed calibration took about7hours;
gen50's broader deep-target generation/training chain took substantially longer.
More repeated deep continuations directly increase game-generation cost.

## 13. Reading order and historical map

Paths are relative to the repository root; `docs/README.md` indexes everything.

1. This file, then `docs/plans/SHIPPABLE_ENGINE_PLAN.md` (the next step).
2. `docs/experiments/value_calibration/` (source of v28), then
   `docs/experiments/gen50/` for gen50, its regressions, checkpoint
   alternatives and the policy/value crossover.
3. `docs/experiments/gen49/` and `docs/experiments/mainline_counterplay/` for
   normal-start/mainline and depth evidence.
4. `docs/protocols/SAMPLED_GATE_PROTOCOL.md` for the primary measurement;
   `docs/protocols/FREE_GATE_PROTOCOL.md` for the older endpoint-uniform
   instrument, not current score weighting.
5. `CONTEXT.md` for durable rules, laws and hazards;
   `docs/history/HANDOFF_LOG.md` for the chronological operational log;
   `docs/history/REPORT.md` for the August experiment ledger. Date-check every
   “current” statement in them.
6. `docs/experiments/gpu48/` and `docs/experiments/search_targets/`: GPU return
   and search-backed training work. Later normal-start evidence can supersede
   early book-only conclusions without erasing those measurements.
7. `docs/experiments/search_first/` and `docs/experiments/search_cpu/`: retained
   cheap-eval/alpha-beta/CPU-GPU research. It improved efficiency without
   establishing a clean both-color replacement. Paused, not production default,
   but directly relevant to a CPU-only shippable engine.
8. `docs/experiments/b2/`: architecture arms and bridge performance work; do not
   infer the gen50 architecture from these experimental variants.
9. `docs/experiments/gen46/`, `docs/experiments/gen47/`: earlier
   bootstrap/data/selection development.
10. `docs/history/DIRECTIVE.md` is the **completed August native-rewrite scope
    record**, not an active directive. `README.md` is the broad entry guide;
    this handoff supersedes its historical defaults for latest runs.

When claims conflict, prefer the relevant immutable manifest, exact checkpoint
hash, completed game journal and final report for that protocol/date. Preserve
uncertainty rather than silently “reconciling” different experiments into one
strength claim.
