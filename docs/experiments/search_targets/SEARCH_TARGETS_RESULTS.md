# Search-backed CPU evaluator experiment — September 13, 2026

Completed at 06:17 Eastern. All 208 final real games passed replay audits. Two
successful rehearsals contributed 16 games each, kept separate. The recovered
real campaign took 4h 27m; the whole plan/implementation/recovery/testing block
took approximately 5h 21m from 00:56. No jobs remain queued or running.

**Conclusion: a small, inconclusive Black gain, not a new both-color improvement.**
The ranking-assisted evaluator scored 43.75% against GPU gen47 at 2s, versus 39.06%
for the unchanged CPU. White's score was unchanged; the paired uncertainty
interval includes regression. It did not beat gen47 overall, and scored 35.42%
against B2. Nothing was promoted, deleted, committed or pushed. The official
release remains v27/gen46; GPU gen47 and the default CPU leaf-epoch3 remain unchanged.

## What was implemented

The playing architecture and search were held fixed: absolute 840→512→32→1,
baseline native CPU search. These are three evaluator-training variants, not
three new playing architectures.

- A training-only native `LabelTree`: full-width minimax through two completed
  player turns, batched gen47 GPU values at its frontier, correct White max/max
  and Black min, capture-before-cap precedence, draw-valued caps, full settled
  repetition history, and no static evaluation during White's pending half-turn.
- Optional sampled-CPU-leaf move lineage, preserving the existing sampling API
  and default search results. Source game history plus the exact leaf path are
  retained so label generation uses the right state and repetition history.
- Provenance-checked generation, exact-input deduplication / split exclusions,
  three matched fine-tuning arms, native export verification, mandatory games,
  source/model-pinned orchestration, recovery and replay auditing.
- Bounded atomic-JSON retries for a real Windows reader-lock failure; permanent
  write failures still fail safely. Recovery reuses verified data/models/book
  and restarts game stages in a new directory without mixing partial evidence.

Why not simply use the existing GPU PUCT Q values? Its tree does not implement
the CPU's full threefold history and its shaped cap values differ. The new
teacher uses CPU-rule-compatible targets. It is still a bounded, imperfect
teacher, not perfect play or a demonstrated replacement for gen47 PUCT.

Primary additions: `native/src/label_tree.rs`, optional tracing in
`native/src/alphabeta.rs` / `leaf_recorder.rs`, and
`tools/{generate_search_targets,prepare_search_backed,train_search_backed,search_targets_campaign}.py`.

## Data and training

Generation used 1,024 TRAIN and 256 VAL roots from the existing family-isolated
B2 corpus, balanced between settled White and Black. It sampled up to two actual
NN leaves per root using the unchanged CPU at 50k nodes. No TEST, human-line,
opening-book or match positions were used for training.

The full job produced 3,770 teacher trees, 33,652 records and 5,264,384 GPU frontier
evaluations in 338.69s. No tree hit the 250k-node cap. After exact-input deduplication
and opposite-split exclusions, the new corpus was:

| Split | States | White to move | Black to move | Sibling ranking pairs |
| --- | ---: | ---: | ---: | ---: |
| TRAIN |22,742|10,428|12,314|12,890|
| VAL |5,712|2,597|3,115|3,274|

There were 4,343 duplicate records, 824 opposite-original-split exclusions and 31
new TRAIN/VAL overlap exclusions. Four retained TRAIN inputs had repeated target
spans above 0.25; their labels were averaged and the ambiguity reported. The
stateless 840 features do not encode repetition history. Original replay was
unchanged.

All arms started from CPU leaf-epoch3. Each step used 2,048 original replay rows,
2,048 side-balanced new rows, and 1,024 sibling pairs. All arms performed the same
forward shapes; only the ranked arm applied the 0.1 ranking-loss coefficient.
AdamW learning rate 1e-4, weight decay 1e-4, seed 913327, 12 full replay epochs, identical data
and optimizer schedule. Raw used gen47 raw values on the new states; backed
used searched values; ranked added same-parent move-preference training.

**An epoch was a 728,450-row original-replay pass**, not one pass over the new
22,742-state pool. New states were therefore sampled about 32 times per epoch,
or 384 times across 12 epochs on average. Selecting epoch 1 is not evidence that
the new states were barely trained. More epochs repeated a small independent
sample heavily, and the common held-out criterion worsened after epoch 1.

All 12 checkpoints per arm were retained. One checkpoint per arm was nominated
using 0.5 original weighted validation MSE + 0.5 side-balanced backed-target MSE.
Every nominee received actual games; there were no per-epoch match gates.

| Nominee | Epoch | Replay VAL MSE | Backed VAL MSE | Ranking agreement |
| --- | ---: | ---: | ---: | ---: |
| Unchanged initialization |—|0.03706|0.15049|77.61%|
| Raw control |1|0.03495|0.14831|77.70%|
| Backed |1|0.04116|0.11939|78.01%|
| Backed + ranking |1|0.04130|0.11901|78.25%|

Better target agreement did not establish better play. All exports matched the
PyTorch network within 3e-7. Peak training allocation was 3.10 GiB, below 12 GiB.
The small resident-data fine-tunes took about 19s each; games dominated runtime.
Candidates are `models/candidates/search_backed_{raw,backed,ranked}_001/epoch_001.bin`
with corresponding `.pt` files. They are CPU evaluators, not gen47-style GPU models.

## Games: development at 300ms

A new frozen 64-entry book was generated from randomly sampled gen47/B2 play,
seed 2143000000, 128 simulations, temperature 0.8, 16 half-moves, then shuffled.
All arms shared the first 12 paired starts; these were not training positions.
Both players received the same nominal per-half-move clock.

| CPU evaluator, vs GPU gen47 | Games | W/D/L | Overall | As White | As Black |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged |24|3/7/14|27.08%|29.17%|25.00%|
| Raw control |24|4/2/18|20.83%|8.33%|33.33%|
| Backed |24|3/5/16|22.92%|25.00%|20.83%|
| Backed + ranking |24|4/4/16|25.00%|25.00%|25.00%|

Unchanged won development overall. Ranked was the best trained arm and advanced
under the predeclared rule: it AND unchanged receive a fresh longer-clock
comparison regardless of the fast-clock result. These 24-game screens do not
prove that raw/backed cannot help at other clocks.

## Fresh confirmation at 2s

Both CPU arms played GPU gen47 on the same 16 NEW paired starts, book 12–27. These
are common-opponent comparisons, not direct ranked-versus-unchanged games.

| CPU evaluator | W/D/L, 32 games | Overall | As White | As Black |
| --- | ---: | ---: | ---: | ---: |
| Unchanged |11/3/18|39.06%|25.00%|53.13%|
| Backed + ranking |10/8/14|43.75%|25.00%|62.50%|

Paired gain: +4.6875 percentage points; 95% paired-start bootstrap interval
**[−6.25, +17.1875] points**. Black gained 9.375 points; White gained 0.
This is encouraging for Black, but inconclusive overall and not a both-color gain.

White outcomes were 3W/2D/11L unchanged versus 1W/6D/9L ranked: fewer losses,
but also fewer wins, leaving score equal. Three White outcomes improved, three
worsened, ten stayed equal. Black had three improvements, two regressions and
eleven unchanged outcomes. Unlike the previous CPU clock-scaling test, White's
individual outcomes were NOT all identical.

An exploratory opening-origin breakdown is heterogeneous: on the six gen47-
generated starts White scored 8.33%→33.33%, but on the ten B2-generated starts
35%→20%. Black rose 50%→66.67% and 55%→60%, respectively. These tiny post-hoc
subgroups are descriptive, not evidence for tuning against one opening source.
Fresh confirmation contained more B2-derived starts by random shuffle, not by
selection after outcomes. Do not compare raw color scores across different books
or treat the 300ms-versus-2s aggregate difference as a controlled scaling result.

## Independent reference and selfplay

Ranked versus GPU B2 at 2s, 12 further paired starts: **8W/1D/15L, 35.42%** overall;
White 16.67%, Black 54.17%. Unchanged was not also tested against B2 on this block,
so this cannot establish a B2-relative improvement over the baseline CPU.

Selfplay used 12 identical fresh starts per engine, book 40–51, 1s per half-move:

| Engine playing itself | White wins | Black wins | Draws | White score |
| --- | ---: | ---: | ---: | ---: |
| Ranked CPU |7|3|2|66.67%|
| GPU gen47 |3|4|5|45.83%|

This is a small skew diagnostic, not a strength ranking or proof of a theoretical
color advantage. The CPU's White-skewed selfplay alongside poor White scores
against GPU opponents shows that selfplay skew and cross-engine color strength
are not interchangeable. No unchanged-CPU selfplay block was run here, so it
also does not measure a training-induced change in CPU selfplay skew.

## Timing, correctness and recovery

The full suite passed 909 Python tests plus 3 subtests after the recovery fix;
26 Rust tests passed after the native changes. Fixed-node default moves, values,
depths and counters match the saved prior runtime, including a final recheck
after the campaign. Existing playing search options remained unchanged.

CPU 2s medians were about 2.006s; GPU 2s medians about 2.09s because it checks clocks
between complete batches. At the first 32 shared-history CPU roots, completed
depth averaged 6.656 unchanged and 6.594 ranked; four moves changed. These are
completed player turns, not conventional chess-engine ply depth. Later average
depths follow different game positions and are not clean same-workload comparisons.

No CPU node ceiling was reached. The 12-turn depth ceiling completed on three
unchanged confirmation decisions and 14 ranked decisions; these are logged,
not silently described as unbounded searches. One resident match worker ran
at a time. Peak match CUDA allocation was 168,495,104 bytes; allocator figures
are not total GPU memory usage.

All 208 final games replayed correctly with no proof contradictions: 70 White
king captures, 101 Black king captures, 28 repetition draws and 9 cap draws.
Cap draws are operational outcomes, not solved fortresses or theoretical draws.

The first real run failed during raw development because Windows denied a
status-file atomic replacement. The observer may have exposed the reader-lock
race. All original files were retained. The first recovery rehearsal then caught
a relative/absolute book-path mismatch; its dependent queue failed closed.
After correction, a new full recovery rehearsal passed. Recovery hash checks
allowed only receipt-I/O/orchestration changes, verified existing data/models/
book, and restarted ALL games in `campaign_recovered_v2`. Interrupted-run games
were excluded rather than pooled or selectively reused. Waiting then used the
process completion event; no live status-file watcher remained.

## Concrete next block — recommended, NOT launched

1. **Improve target consistency and independent coverage.** The diagnostic in
   `SEARCH_TARGETS_DATA_DIAGNOSTIC.md` found that 88.8% of pre-dedup records are
   one-turn successor targets, while actual CPU-leaf/source roots have two-turn
   targets. One-turn means shift about 0.15 toward the side to move. This is a
   plausible calibration concern, not a proven explanation of the game results.
2. Use one consistent completed-turn horizon for point-value targets on many
   more independently sampled actual CPU NN leaves. Either relabel successors
   at that same horizon or reserve them for within-parent ranking only. Only
   8.54% of existing records have exact ±1 labels: this is not a rationale for
   arbitrary pawn rules or claiming most of the corpus is trivial.
3. Define training exposure in new-state batches / independent roots as well
   as replay epochs. Avoid unintentionally repeating a 22k-state pool 32 times
   per epoch. Keep the matched raw control, replay anchor, current architecture
   and a small fixed number of nominated checkpoints with mandatory games.
4. Diagnose changed White wins/draws/losses on these games and known human lines
   as EVALUATION evidence only. Keep those positions out of training. Require
   fresh both-color confirmation against gen47 and a baseline-matched B2 check
   before promotion; enlarge a promising confirmation rather than claiming a
   32-game interval that crosses zero is a verified improvement.

Do not automatically scale this exact recipe, add another architecture family,
or treat better MSE as strength. The current evidence supports a focused target-
quality/exposure experiment, not replacing the default CPU or GPU release.

## Authoritative evidence

- Plan: `SEARCH_TARGETS_OVERNIGHT_PLAN.md`; operational history: `HANDOFF.md`.
- Generation/preparation: `benchmarks/search_targets_20260913/campaign/{corpus,prepared}`.
- Final games: `benchmarks/search_targets_20260913/campaign_recovered_v2/`.
- Final `summary.json`, `paired_confirmation.json`, `replay_audit.json`,
  `recovery.json`, and per-stage `analysis.json`, manifests and game journals.
- Original full rehearsal: `rehearsal/`; successful recovery: `recovery_rehearsal_v2/`.
- Final default-runtime parity: `final_default_parity.json` under the benchmark root.
- Runtime SHA256: `8b3b72e1bdde1c3cb0ab5454bfbb81a6db19bf1a450acb2776ad1bb081dc3099`.

Ranked nominee SHA256:
`36128dc5f1f981a397bd291aac37e796a6320e5f8bcdd21ed836650fc58c75ce`.
