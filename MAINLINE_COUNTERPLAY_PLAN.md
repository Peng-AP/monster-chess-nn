# Gen49 counterplay study — September 15 overnight

Owner: promising human playtest; plan -> implement -> wait for about eight hours.
This is a bounded strength/line investigation, not gen50 training or a release.
No model, training corpus, book, rule, native binary, or default search change.
No promotion, cleanup, commit or push. Preserve completed gen48/gen49 evidence.

## Questions and frozen references

Gen49 epoch7 scored94% against gen48 in800 independent sampled normal-start
games. Much of White's gain follows e4+d4 ...e5:369W/1D/3L against gen48.
Against gen48's ...d5:1W/14D/0L. In gen49 selfplay after ...d5:0W/48D/131L.
All96White wins against B2 followed ...e5. This makes ...d5 counterplay the
specific unresolved question, not a demand for more arbitrary opening variety.

The B2 Black-score decline is not established conversion regression: gen48
usually chose ...e5, gen49 usually ...d5, and both drew their ...d5 games.
56 of57gen49 Black draws reached the same opening endpoint; all56 share the
same continuation. B2 is now an explicitly examined diagnostic opponent, not
a fresh blind test. No source games from this study enter training.

Models remain frozen:

- gen48: `models/candidates/bootstrap_main_gen_0048/arena_selected.pt`, SHA256
  `a8c074390c93390ac974b1f58076a86aa0525d9a34f7cff66e12442bb7e07722`.
- gen49: `models/candidates/bootstrap_main_gen_0049/arena_selected.pt`, SHA256
  `4bcc68a0219acf8c3dc53326d738789e6bd767fccba88fd4f471345567b4a647`.
- B2: `models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt`, SHA256
  `fc23076a9f5c7f237785f27cb1a665c10588ea8e8916cd743016d19a96999d15`.

## State fidelity and interpretation

Four early roots are the two actual White orders e4,d4 and d4,e4, followed by
...e5 or ...d5. Replay from the normal board, including all clocks, half-turns,
move-stack history and driver repetition counts. These commuted histories
are not four independent strategic opening families.

Select the lowest-seed gen49-as-Black B2 draw from the most frequent exact
opening endpoint in the preserved `gen49_vs_b2.jsonl`. Audit its entire game
before extracting roots at absolute search plies16,17,41,62,65. These cover
the first sampled endpoint, a Black reply, later play, and entry into the
observed repetition loop. They are correlated diagnostic positions, not five
independent games. Store source task ID, journal hash and full prefix in every
case. No ambiguity reconstruction from FEN-only human logs is needed.

Each new continuation starts a fresh search tree with the genuine prefix.
This does not reproduce the original unrecorded cached search tree. The native
adapter receives its usual recent move history; the driver restores ALL past
repetition counts. Native MCTS does not itself adjudicate full threefold history.
That limitation is preserved and reported, not silently fixed during the study.

Repetition/turn limits score draws, not solved fortresses. A root value is a
model/search opinion; a win against one defender is not an adversarial proof.
Sampled normal-start scores remain primary. Conditional roots diagnose why
models differ; do not pool them into an Elo or replace free play with a book gate.

## Fixed production sequence

1. **Rehearse everything**, including two White halves, full-history restoration,
   repetition, legal replay audit, resumable tasks, model/hash drift rejection,
   unequal-search normal matches and correctly identified equal-search selfplay.
   Tiny games/search budgets are plumbing tests, never strength measurements.
2. **Conditional cross-play:** all nine ordered White/Black pairings of gen48,
   gen49 and B2 on all four early roots. Budgets3,200 and12,800 each side, with
   eight sampled continuations per case/pair/budget. Temperature.5 only until
   absolute ply16, then zero, no root noise. 576games. Then the same nine pairs
   and four roots at51,200 each, two continuations each:72games. Every result
   remains, including losses and repetitions. 648early-root games total.
3. **Dominant draw cross-play:** the five genuine-history later roots, all nine
   ordered model pairs, budgets3,200/12,800/51,200 each side. One temperature-zero
   continuation each:135games. No fake replication through random seeds after
   opening sampling has ended. Endings, exact repetition and policies recorded.
4. **Deep root decisions:** all nine roots and each of three models, budgets
   3,200/12,800/51,200/204,800:108probes. Complete the actor's turn, including
   both White halves when applicable. These probes deliberately use pure neural
   MCTS with early stopping OFF and no finisher interception, to see whether
   more actual search changes the decision. They are NOT a new game-playing
   configuration or a source of WDL unless the short probe actually terminates.
   Game stages retain the existing early-stop/reuse/finisher settings, so their
   simulation counts are ceilings. Record wall times; do not claim exact depths.
5. **Normal-start checks:**160games each,80per model-A color, opening sampler
   unchanged (temperature.5 through ply16), disjoint seed blocks:
   - gen49@12,800 vs gen49@3,200: does more search help the same weights?
   - gen49@3,200 vs B2@12,800: can stronger opponent search expose weaknesses?
   - gen49@12,800 vs B2@12,800: does the answer survive stronger search both ways?
   - gen49@12,800 selfplay: does the color skew persist?
   640games. Unequal-search games are not equal-compute strength comparisons
   or equal-agent selfplay. No operational binding gate or model promotion.

Total:1,423completed conditional/normal games plus108root-turn probes, excluding
rehearsal. All branches run regardless of preceding scores. No selecting the
most favorable budget after seeing results; report the entire ladder.

Seeds: conditional production2,220,000,000 plus deterministic task index;
normal blocks2,230,000,000 through2,233,000,000. Rehearsal conditional
2,240,000,000, normal2,250,000,000 through2,253,000,000. Model and budget are part
of task identity. Reusing a prefix is intentional, not a new strategic sample.

## Safety, implementation, time

Use a small separate full-history continuation/probe runner and the existing
normal-match engine and launcher receipts. No Rust rewrite and no architecture
addition. Atomic per-task result files allow exact resumption of missing work;
completed moves/outcomes/prefixes are replay-audited before publication and on
reuse. Hash frozen tasks, model files, source journals, code, rules, native
binary and settings. Implementation changes require a new evidence namespace.

Separate OS campaign lock and normal GPU worker lease. Maximum eight workers,
four workers for root probes (large tree RAM), one heavy stage at a time,
at most two evaluators per worker; campaign allocation
target <=12GiB VRAM. Do not close the user's applications. Per-game deadline
and pool-stall timeout stop a genuinely stuck run and retain completed evidence.

Target about eight hours including implementation/rehearsal: roughly45-75minutes
setup;2-3hours early-root cross-play;0.5-1.5hours later roots and probes;2-3hours
normal-start tests, plus the high-budget early-root block. These are estimates;
deep games can run longer. Finish each fixed scheduled block; no score-based
early stop or extension to manufacture significance. No automatic training next.

Launcher `tools/start_mainline_study.py`, worker `tools/mainline_study.py`.
Current evidence `benchmarks/mainline_counterplay_20260915_v2`; managed run
`mainline_counterplay_v2`. Wait on process completion with infrequent milestone
health reads, not repeated status probes or file watchers. Finish with a report
that distinguishes better opening selection, stronger continuation, and remaining
counterplay; use that evidence to propose the next training increment.

Prelaunch safety refinement at01:36: first real rehearsal passed169games and
27probes, plus977tests+3subtests. The machine had about10GiB free host RAM.
Cap root-probe pools at four rather than eight workers and add four real204,800
simulation probes to the rehearsal. Production budgets/counts remain unchanged.
Preserve `rehearsal/`; rerun the full immutable-source check under `rehearsal_v2/`
with169games/31probes. This is a resource preflight refinement, not a failed
production run or an outcome-driven experimental change.

Protocol correction at02:47: the first production runner deduplicated engines
by checkpoint/simulation budget, unintentionally sharing one search tree across
colors in equal-model conditional games. The ordinary match harness uses two
independent trees, even with equal weights. Stopped only managed PID69152 and
its descendants after verifying its name/start time. Preserve the original
directory and all completed tasks; its conditional self-play is not comparable
under the intended protocol. Do not pool it into the corrected result.

Corrected runner keeps separate White/Black search objects, with a regression
test for equal-checkpoint ownership. Root-turn probes still use one root actor's
tree, appropriately. Start a new fully rehearsed immutable namespace `_v2`;
rerun the original fixed task/seed schedule, including non-self comparisons,
without importing or relabeling old evidence. Normal-match engine and gen49
weights were never changed. The earlier gen49 strength results are unaffected.
The restart costs roughly one hour; the corrected full schedule may run beyond
the original eight-hour guide. No sample-count reduction based on outcomes.
