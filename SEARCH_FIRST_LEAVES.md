# Search-leaf control — September11

## Why this experiment

The architecture control failed: absolute512 scored18.75% and king-relative512
15.625% against gen47 on32games each (same16 paired starts64..79,300ms per
individual move). The more elaborate representation is not the next candidate.

The diagnostic at `benchmarks/search_leaf_audit_20260911` sampled15,812 actual
uncached NN evaluations from256 original TRAIN/VAL roots, preserving game history.
White-perspective compression MSE against the fixed gen47 raw evaluator:

| Validation position type | White to move | Black to move |
|---|---:|---:|
| Stored game roots, absolute512 |0.0501|0.0127|
| Actual search leaves, absolute512 |0.2240|0.0988|
| Actual search leaves, relative512 |0.2386|0.1058|

Root sample sizes are49White/43Black; leaf sample sizes5153White/2659Black.
Leaves within a root are correlated. These are descriptive compression errors,
not independent statistical strength estimates or errors against perfect play.
The teacher itself is imperfect. Negative mean White-perspective errors at
White-to-move leaves indicate systematic disagreement on this sampled search
distribution, not a proven cause of every game loss.

Feature coverage does not support a broad untrained-column explanation: only
29/15,812 absolute leaves and87relative leaves activate never-trained inputs.
Raw-versus-canonical EP differs on1088leaves and changes absolute predictions by
0.0938 on average there. We do not alter EP rules or erase inputs on that basis.

## Controlled change

Keep absolute840 ->512 ->32 ->1, FP32 native inference, existing alpha-beta,
extensions0, same phase/rule/repetition behavior. Change training distribution.

1. Sample2048 TRAIN and512VAL roots uniformly from the original family-isolated
   B2 corpus, seed9275. Restore every root's recorded full history; retain raw EP.
2. Search up to100,000nodes, depth8,60second safety timeout, using original
   absolute512. Reservoir-sample up to64 uncached, non-proven NN leaves per root.
   No mid-White-turn static leaves. Proven/terminal states are not teacher labels.
3. Label with the same frozen gen47epoch17 raw value, converted to White view.
   No test, gate, book or human roots become training data. This is compression
   under search distribution, not new solved truth and not bootstrapping strength.
4. Deduplicate new leaves by complete evaluator input, including phase, rights,
   rawEP and remaining budget. Exclude train leaves matching originalVAL/TEST
   inputs; excludeVAL leaves matching originalTRAIN inputs; remove train leaves
   shared with newVAL leaves. Original corpus itself remains unchanged.
5. Two matched fine-tuning arms from absoluteepoch7: replay-only control and
   50%replay/50%leaf loss. AdamWlr0.0002,wd0.0001,20epochs,seed3273. Each step
   uses2048 shuffled originalTRAIN rows plus2048 independently sampled rows
   (original replay in control, leaves in treatment). Original value weights
   retained on replay; unique leaf targets weighted uniformly.
6. Both nominate by the same metric: half original weighted validation MSE,
   half new leaf validation MSE. Every epoch saved; no test-based selection.
   Both nominated models must pass native/PyTorch export parity and play games.

## Chained checks

Managed `search_leaf_corpus` then `search_leaf_campaign`; latter waits for process
completion and explicitly requires a successful corpus receipt. One heavy job
at a time; VRAM below12GB. Implementation/runtime hashes pinned between stages.

- Full Python suite before training.
- Train replay-only and leaf-mix arms, preserving all old checkpoints.
- Original/replay/leaf each32games against gen47, starts96..111,300ms.
- Leaf improvement >=10percentage points above BOTH controls triggers64fresh
  gen47games and32B2games at300ms on starts160+. This is a development trigger,
  never an automatic promotion criterion.
- Regardless of trigger: original and leaf each16games at2seconds on identical
  starts128..135, to test whether better leaf judgment benefits longer search.
-16common-start selfgames per leaf engine/gen47 on160..175 at300ms; known human
  line diagnostic at300ms/2s. These self starts overlap confirmation starts and
  are explicitly a color-skew diagnostic, not independent held-out evidence.
- Actual elapsed time, completed depth, proof consistency and paired-start
  uncertainty recorded. Nominal budgets remain soft per-halfmove budgets.
- Full completed-game replay audit before the campaign success receipt.

## Boundaries and continuation

### Extended queued validation

`search_leaf_extended` waits for the entire first campaign AND its replay-audit
receipt. It selects one arm solely by the32development games (ties prefer the
unchanged original), then uses disjoint starts224..255 for64gen47games at2seconds
per half-move. Original control is tested on exactly the same starts/clock; if
the original wins nomination it is not redundantly run twice. Follow with32B2
games at2seconds,64CPU-only gen47games at300ms on256..287, and16common-start
selfgames per nominee/gen47 at2seconds. Final full replay audit is mandatory.

This extends the unattended work to several hours, plausibly most of the roughly
eight-hour allowance depending on game lengths and whether confirmation triggers.
No extra training or architecture changes occur during these locked tests.
CPU-only scores are separate deployment evidence (FP32 incumbent, one thread),
not pooled with GPU-backed gen47 scores. Common self starts overlap the fresh
H2H starts intentionally; they measure color skew rather than independent proof.

No production change, promotion, deletions, commits or pushes. v27 remains the
release and gen47 the strength reference. Preserve preexisting dirty worktree.
Optional leaf instrumentation defaults off and passes fixed-node default parity.
Runtime snapshot is `benchmarks/search_first_20260911/runtime_before_leaf`.

Exact PVS window helpers exist unlinked in native/src/search_window.rs. Do not
enable or rebuild them during this experiment. Evaluate this one training change
first. If useful, confirm/generalize before collecting another on-policy leaf
generation. If not, do not assume more identical data will fix search; inspect
the clock controls and root/leaf tradeoff before another intervention.
