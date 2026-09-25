# Search-first experiment — September 11

Latest: the initial validation and label-control chains completed at04:11.
No stronger engine emerged. All completed game logs subsequently passed full
legal replay/outcome audit. Owner authorized the next king-relative evaluator
control; see `SEARCH_FIRST_RELATIVE.md` and current `HANDOFF.md`. The dated
checkpoint below records the earlier run in progress, not the live queue.

## Current checkpoint (~03:00 Eastern)

First validation leg completed: **12W/8D/44L against gen47 @300ms**,32 paired
starts/64 unique games. Overall25%, White23.4375%,Black26.5625%; paired-start
bootstrap95% interval17.1875%..32.8125%. This is clearly below gen47 on this
instrument, not merely a Black-only failure. No proof contradictions or decisions
without a completed first iteration. Mean depthW5.04/B4.62 completed turns;
~380k/346k nodes/sec. Actual medianAB302.4ms,PUCT313.1ms; p95304.7/320.8ms.
Longer-time validation follows, without runtime or nominee changes.

The2s leg subsequently completed **6W/0D/10L**,White12.5%,Black62.5%,overall37.5%.
On EXACTLY the same eight starts the300ms subset was also37.5% (White25%,Black50%).
Thus this small matched clock comparison does not show an aggregate gain from
6.7x more thinking time; it shifts the color split. Historical gen47 self3200 on
these eight starts scored81.25% Black, so62.5% Black is not standalone evidence
of improvement. Same-clock selfplay remains necessary context. Actual2s medians:
AB2.006s,PUCT2.083s; p952.008/2.144s. No proof contradictions. Mean depthW6.13/B5.41.
Secondary-opponent validation is now active.

`search_first_validation` is active, driven by `tools/search_first_validation.py`.
Do not edit its pinned runtime until it finishes. This is a new opt-in engine,
not a release candidate. No existing checkpoint, dataset, or release is replaced.

Completed locked 2x2 development screen, 16 games per arm at300ms/half-move:

| First-layer width | Threat extension turns | Overall score | White | Black |
|---|---|---|---|---|
|128|0|21.875%|37.5%|6.25%|
|128|2|25%|25%|25%|
|512|0|34.375%|50%|18.75%|
|512|2|28.125%|50%|6.25%|

These are the same eight selection starts, not independent confirmations.
Width512/q0 is nominated; no positive strength claim. Wider model is the same
840->width->32->1 architecture and teacher corpus, not a separate model family.

Ordering, static caching, exact-path/depth searched bounds, generated Move
application, compact repetition keys, exact fast capture scans, and bounded
whole-turn threat extensions are now implemented and tested. Earlier dated
sections below describe intermediate states, including formerly unlinked files.
Extensions remain optional and are disabled for the validation nominee.

CPU-only evaluator hot-path optimization removes per-leaf feature allocations,
reuses a per-search activation scratch buffer, and uses independent float-dot
accumulators. It is neither incremental NNUE nor quantization. Trained-model
prediction delta <=3.3e-7 on the fixed122-state check, below preregistered1e-5;
Torch parity also passes. Width512 inference including Python/FEN overhead:
12.04us ->7.35us; fixed12-state whole-search throughput267k ->367k nodes/sec.
Width128 throughput629k ->753k. Selected-sample measurements, not universal gains.
21 Rust tests passed; validation begins by rerunning the full Python suite.

Timed incumbent overhead was profiled and reduced with an opt-in fast reroot
that moves tree-node ownership instead of cloning. Node/state/statistics parity
against old reroot is tested through six decisions. Untimed defaults unchanged.
Actual300ms game clocks previously measured median AB304ms vs PUCT312ms.
These remain soft per-half-move budgets, NOT strictly equal end-to-end clocks.

Validation plan (one resident worker, fail-closed source/runtime hashes):
64 games gen47 @300ms on indices32..63;16 gen47 @2s on32..39;
32 games secondary B2 @300ms on48..63;16 common-start selfgames each engine on
32..47 @300ms; one deterministic free-play color pair @2s. Indices are new for
this search-first experiment but reused by older campaigns. No claim of globally
unseen positions. Every game records full moves, time, depth and proof claims;
contradicted proven wins stop the chain. Do not pool different clocks/opponents.

Reference remains gen47; official release remains v27. Human-line diagnostics
and interpretation follow completion. No destructive actions or commits done.

### Queued label alignment control

`search_first_label_control` waits without engine work until validation completes.
It then trains two same-capacity width512 nominees: existing shaped outcomes,
and existing recorded MCTS values. The latter are SIDE-TO-MOVE in processed data
and must be multiplied by the turn plane to become WHITE values; five dedicated
perspective/validation tests pass. Source teachers are v27 and gen47 epoch11,
not exclusively the stronger gen47 epoch17 raw-value teacher. Existing corpus,
weights and family split stay intact; no new labels are invented or old files
overwritten. Each nominee gets16 games on the original eight development starts.
Then known human game00031 rows7/14/15/16 and three legal row14 alternatives are
probed with raw-distilled512 and gen47,300ms/2s, complete root turns. These are
diagnostics and are excluded from new training.

After that, repeat raw-distilled512 on the same development starts under the
current optimized runtime to complete the same-runtime label comparison (the
first width512 development result predates the CPU hot-path optimization).
Prepared `tools/search_first_cpu_match.py` for a separate CPU-only deployment
control: hides CUDA before PyTorch initialization, fixes one neural CPU thread,
records FP32 versus the GPU run's FP16 distinction. This is not a way to relabel
a loss against GPU-backed gen47 as a new champion. It tests whether the small
native evaluator has practical value on hardware without a GPU. Not yet run.

## Opt-in use (not the production play engine)

Build with `powershell -ExecutionPolicy Bypass -File tools/build_native.ps1`.
From the repository root, a single search can be called as follows:

```python
import sys
sys.path.insert(0, 'native')
import monster_native as native

net = native.CheapValue('models/candidates/search_first_distilled_w512_001/epoch_007.bin')
result = native.alphabeta_search(
    'rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1',
    seconds=1.0, evaluator=net, pending=False, turn_count=0)
print(result.action, result.value, result.completed_depth)
```

`action` is a single UCI half-move, not White's complete pair. Supply
`pending=True` for White's second move and the actual completed-turn count.
`value` is always White's perspective; `None` means no iteration completed.
Root repetition history requires settled four-field FEN keys excluding the
current occurrence. Use `tools/search_first_match.py` as the stateful example;
do not silently omit history in actual games. `max_depth` counts completed
turns, not ordinary chess moves or White half-moves. Optional threat extensions
are disabled by default; exact king capture always outranks finite evaluation.

Match example (creates a NEW output directory):

```powershell
py -3 tools/search_first_match.py --value models/candidates/search_first_distilled_w512_001/epoch_007.bin --pairs 8 --seconds .3 --out benchmarks/my_search_first_check
py -3 tools/analyze_search_first.py benchmarks/my_search_first_check
```

Use `--selfplay` for AB on both sides or `--reference-selfplay` for the common
PUCT baseline. A selfplay score is White's share, not a challenger H2H score.
These binaries use a different format from CNN checkpoints and are intentionally
not listed by `play.ipynb`'s CNN model selector. No standalone executable or
browser integration is delivered by this experiment.

Owner authorized an unattended overnight session without a hard time limit.
Goal: determine whether native alpha-beta plus inexpensive learned CPU value
evaluation is a useful strength/compute trade against the verified current gens.
This is not a promotion authorization and not a replacement of MCTS.

## Boundaries

- Preserve the dirty worktree, existing checkpoints, datasets, and verdicts.
- One worker-heavy job at a time; at most eight workers. Long work uses runs.py.
- New engine is opt-in. Existing rule/search behavior stays unchanged.
- No king-safety heuristics masquerading as learned improvements. Exact rules,
  tactical terminal recognition and phase handling are correctness requirements.
- No null-move pruning, late-move reductions, or speculative ordinary-chess
  quiescence shortcuts in the first implementation.
- Baseline gen47 arena_selected (epoch17); secondary B2 seed9053 state epoch8.
  Locked September10 campaign completed; inspect final summary before matches.

## Stages and exit criteria

1. Correct native search: White max/max/Black min, completed-turn depths,
   iterative deepening, bounded time/nodes, transposition move ordering, explicit
   cap/repetition handling. Compare shallow results to exhaustive minimax;
   test both White phases, immediate king capture, repetition and timeout.
2. One compact sparse CPU value network trained on frozen existing generation
   data, with phase/rule-state inputs and no color-swap augmentation. Separate
   output directory, auditable split, training/runtime numerical parity.
3. Profile full search, not only inference. Add timed incumbent execution;
   record actual time and overruns. Equal simulations is not equal compute.
4. Small staged actual games first, then matched held-out starts and free play,
   per-color outcomes, common-start selfplay and known human-line diagnostics.
   Tactical tests are diagnostic, not new independent strength evidence.
5. Report go/no-go: faster nodes alone do not count as success. Keep a weaker
   challenger only if it demonstrates useful independent error discovery.

## Initial inspection

No active managed jobs at start. Native Game/bitboards/half-move generation and
exact Black forced-capture solver are reusable. No generic alpha-beta/NNUE exists.
Native MCTS still calls batched Python/PyTorch for NN inference. Game terminal cap
uses heuristic relabel internally while actual capture-only match score is draw;
the prototype must explicitly document its cap policy rather than silently
rewriting the incumbent. Repetition is driver-level and counts settled states
only. White first-half unsafe paths exist by design in the half-move API.

## Implementation and first evidence

- `native/src/alphabeta.rs`: opt-in iterative-deepening search. Depth counts
  completed turns; no leaf evaluation with White's second half pending. History
  tracks settled threefold repetitions, capture precedes cap, cap scores draw.
  TT is ordering-only (no history-unsafe score reuse), basic capture/promotion
  ordering; no selective pruning or qsearch yet. Existing heuristic is a plumbing
  fallback; learned model is optional. Timeout retains last complete iteration.
- `native/src/cheap_value.rs`, `tools/train_search_value.py`: sparse 840-feature
  absolute piece-square + 3 phase + rights + raw EP + remaining-budget evaluator,
  128/32 hidden widths, ReLU, White-perspective tanh. Float32 sparse recomputation,
  NOT yet incremental or quantized. Versioned MCSV001 binary, strict loading.
- `NativeMCTS.get_best_action(..., seconds=...)` / native PUCT optional clock:
  checked between completed batches, defaults remain simulation-limited. Actual
  time can overrun by batch cost; never label it a hard real-time limit.
- `tools/search_first_match.py`: single-resident-worker timed actual games,
  per-half-move budget, full moves/timing/depth/nodes, common starts both colors,
  repetition and captures-only scoring. No finisher wrapper on incumbent in this
  initial search-core comparison; this is explicitly not the standard gate.

Validation: five Rust tests; 50 initial Python tests; eight cheap-value/search
tests; 25 adapter/search/cheap-value tests after clock addition; three dedicated
timed-PUCT tests. Counts overlap, do not sum. Export/input parity is tested on
both colors, phases, castling, EP and turn budget. More randomized tests needed.

Training `search_first_value_001` completed 30 epochs on frozen B2 processed24,
913,960 rows with original train/val/test family split. Roughly 23 seconds for
this tiny network with features resident on GPU; peak allocation 3,197,395,968
bytes. Validation selected epoch2 (MSE .12640668), held-out MSE .12396580. This
selection is provisional offline nomination, not evidence of playing strength.
All 30 snapshots preserved; no per-epoch match sweep. Directory:
`models/candidates/search_first_001`, complete.json pins binary hash.

30ms/half-move smoke: eight games from first four reused September10 common
starts, all legal and all lost to gen47. Mean completed alpha-beta depth 3.43
turns, mean 6,500 nodes/decision. Median actual time AB30.25ms, PUCT32.12ms;
p95 AB30.50ms, PUCT33.94ms. These are tiny-budget integration results, not a
general conclusion. Artifacts `benchmarks/search_first_20260911/smoke`.

`search_first_screen_300ms` now runs 16 games, same first eight diagnostic starts,
at300ms/half-move. Do not edit loaded runtime until it completes. Next: inspect
time/depth/outcomes and failures, improve exact tactical frontier handling and
ordering where justified, then larger-time screens before expensive confirmation.

Potential correctness refinement already identified: incumbent threat clamp is
only .95, while a learned value may exceed it. In minimax a proven immediate
king-capture continuation should outrank any nonterminal estimate. Handle as an
explicit exact tactical frontier result with tests, not a heuristic preference.

Status: implementation/testing in progress; no strength claims or promotion.

## September11 continuation checkpoint (~01:40 Eastern)

Full suite passed **849 tests + 3 subtests**. Exact tactical frontier correction
implemented and verified in sixth Rust test; 16-game300ms result unchanged:
2 wins/3 draws/11 losses, White37.5%, Black6.25%. No logged proven-win claims
contradicted eventual results. Mean depth White4.73/Black4.29 completed turns;
~238k/242k nodes/sec. Actual time AB median301ms, PUCT322ms/p95350ms: nominal
equal budgets, not strict end-to-end equal time; preserve/report that caveat.

`search_first_compression_control` is active via `tools/search_first_followup.py`.
Sequential fail-closed stages, runtime hashes checked before each stage:
full tests -> gen47 teacher labels -> same-architecture student ->300ms16games
-> original outcome student2s8games -> distilled student2s8games.
No loaded runtime edits until this chain completes. New unlinked source modules
and analysis scripts may be prepared independently but are not yet active.

Teacher labeling all913,960 rows completed18.5s, raw NN outputs converted to
White perspective, checkpoint hash prefix810297fe (full receipt stored).
Legacy15ch plane14 is reconstructed from B2 plane15, not truncated signed rank;
dedicated encoding test passed. Label cache has source/split/checkpoint hashes.
Student `models/candidates/search_first_distilled_001` trained30epochs22s,
validation nominated epoch22, teacher-target MSE val.0358214/test.0361989.
These errors measure teacher approximation and are NOT comparable to original
outcome-target MSE. First16 games300ms scored18.75% (White31.25%,Black6.25%):
compression alone has not helped at short time. Await deeper paired screens.

Prepared `native/src/search_order.rs`, NOT YET LINKED/COMPILED/TESTED:
compact exact board/phase key for ordering-only transpositions, distinct
promotion move IDs, phase-specific history and ply-specific killers. Intended
next integration after locked chain finishes, with exhaustive-minimax parity
and move-set invariance tests. No speculative branch pruning.
Also consider a history-independent STATIC-evaluation cache keyed on exact
board/phase/turn budget: repetition terminal checks must precede it. This is
not TT search-score reuse; it avoids repeated threat scans/network evaluations
at commuting White-pair transpositions without graph-history mistakes.

Source files added in this session: SEARCH_FIRST_EXPERIMENT.md,
native/src/{alphabeta,cheap_value,search_order}.rs,
tools/{train_search_value,distill_search_value,search_first_match,
search_first_followup,analyze_search_first}.py,
tests/{test_alphabeta,test_cheap_value,test_timed_mcts}.py.
Small additive edits to native/src/lib.rs, native/src/mcts.rs and
src/native_mcts.py; the latter two already contained pre-session user changes.
Do not commit those whole preexisting diffs as this session's work.
Original runtime backup native/monster_native.pre_searchfirst_20260911.pyd;
initial prototype runtime native/monster_native.searchfirst_v0_20260911.pyd.
No deletes, promotion, commits or pushes performed.

### Next search work after frozen comparison

Loss-trace inspection supports a horizon concern: Black often reports a roughly
balanced value at depth4, then an exact forced loss becomes visible at depth5
one or two decisions later. This is not proof that extra depth fixes the earlier
position; test it. The first eight common starts are a small, nonrepresentative
sample, some already contain forced wins. Do not interpret Black6.25% without
the paired incumbent comparison, or extrapolate it to the whole opening space.

Priority after ordering/cache: conservative tactical frontier extensions. At a
nominal leaf, recognize own immediate capture exactly. If the mover's king is
under the opponent's immediate-turn capture threat, extend ALL legal responses
through a complete turn with a bounded extra-turn budget. White threats are
two-move threats, Black threats one-move; pending White must finish its turn.
This avoids blindly importing captures-only qsearch, stand-pat assumptions,
null moves or late-move reductions. A bounded extension that exhausts its budget
must return an ordinary finite estimate, not claim a proof. Record extension
nodes/depth and test full minimax parity when extension is disabled. Compare
actual equal-budget games with and without extensions before adopting them.

Prepared additional UNLINKED helpers `native/src/search_cache.rs` (static values)
and `native/src/search_bounds.rs` (history-safe searched bounds). Bound identity
includes exact depth, extension budget, turn budget, exact board/phase and exact
settled search-path suffix. Pre-root history is constant per Search object.
Never reuse across roots or skip repetition checks. When integrating, store
bounds relative to the alpha/beta window saved AFTER probing and BEFORE the
child loop; do not classify against the mutated final alpha/beta.

Planned hot-path cleanup with the helpers: native Move structs instead of UCI
strings at every node; compact repetition identities and a counted stack rather
than repeated FEN serialization. Add a generated-move apply helper to Game that
can omit unused oscillation history ONLY for the opt-in alpha-beta search.
Existing Game.apply_half keeps recording history and must retain exact parity.
This is implementation optimization, not a new pruning assumption.

Potential bounded evaluator-size control after search optimization: same inputs,
same gen47 teacher labels, same32-unit second layer, first layer128 ->512.
Current111,809-parameter network is intentionally tiny; cheap need not mean
minimum parameter count. Profile whole-search cost and actually play the nominee
before deciding. Do not start a large architecture sweep or reselect arbitrary
epochs until an apparent winner emerges. Original outcome2s screen finished
0wins/0draws/8losses; distilled2s screen still running at this checkpoint.

Also preserve nonterminal network distinctions near1: the current prototype
clips to .99 to reserve exact +/-1. A tighter numerical reservation (e.g.
1-1e-6) would avoid collapsing distinct confident predictions. Any such change
must be explicit and tested; do not change the running comparison.
