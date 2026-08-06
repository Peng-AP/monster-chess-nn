# Monster Chess NN

A neural-network + Monte-Carlo-tree-search engine for **Monster Chess**, an
asymmetric chess variant:

- **White** has only a king and four pawns (c2–f2) — but makes **two moves per
  turn**.
- **Black** has the full standard army and moves once per turn.
- The game ends when a king is **captured**. A king capture wins unconditionally —
  there is no check or checkmate, and capturing is legal even when it would leave
  the capturer's own king attacked.
- A game that reaches the turn limit is scored by final position (win/loss lean or
  draw), applied symmetrically to both sides.

The double move makes White's lone king a genuine attacking piece and gives the
variant its character: Black is winning with correct play, but converting through
the pawn phase requires real technique. Measuring and improving **Black's
conversion strength** is the project's primary quality signal, which is why all
evaluation tooling reports per-side results rather than a single aggregate.

## How the engine works

**Rules layer** (`src/monster_chess.py`) wraps `python-chess` and exposes two
action APIs: an atomic `(m1, m2)` pair API used for game play and strict move
legality, and a *half-move* API that decomposes White's turn into two single-move
plies for search. The factorization collapses White's branching factor from ~900
move pairs to ~30 + ~30 single moves, and lets the policy network give each half
its own prior. White's first half-move may legally pass through check; only the
completed turn must leave the king safe.

**Search** (`src/mcts.py`) is AlphaZero-style batched PUCT when a network is
loaded (policy priors, FPU, Dirichlet root noise for self-play only, virtual loss,
tree reuse across White's two half-moves) and sequential UCB1 when running on the
heuristic evaluator. Search values are reported as the selected child's Q so proven
mates read ±1.0. Selection applies safety overrides that re-rank only among the
search's own children: never hand the opponent an immediate king capture when a
searched alternative survives, and never play a first half-move whose every
completion loses the king.

**Evaluation** (`src/evaluation.py`) provides three interchangeable evaluators:
a hand-tuned Monster-Chess-aware heuristic (material, pawn progress, king
confinement geometry, mating-barrier detection), `NNEvaluator` (dual-head ResNet),
and `HybridEvaluator` (NN policy priors with heuristic leaf values, used to
generate training data). Positions where the side to move can capture a king this
turn are clamped before any network call.

**Network** (`src/train.py`) is a ResNet (configurable stem and tower, default
8 blocks up to 128 channels) with a policy head over a flat 4096-move space
(`from_square × to_square`) and a value head that can train as a scalar regression
or a win/draw/loss classifier. The established policy head is dense; an opt-in
source-to-destination attention head provides the same logits with far fewer
parameters. An opt-in moves-left auxiliary head predicts remaining recorded
decisions without reshaping the value target. Checkpoint architecture is inferred
from the state dict, so older checkpoints remain loadable and inference still
returns the same `(value, policy)` pair.

**Position encoding** (`src/encoding.py`) is 17 planes: 12 piece planes, side to
move, a half-move indicator (distinguishing White's second half-turn), a signed
rank coordinate, and symmetric pawn-progress planes for both colors. Legacy
15-plane checkpoints are detected and served automatically. Training data is
doubled by horizontal mirroring (the variant is file-symmetric).

**Value targets** are grounded in game outcomes, with an optional end-anchored
ramp (`result × γ^min(plies_to_end, horizon)`) that restores a value gradient in
long won positions — without it, outcome labels saturate near ±1.0 across whole
games and search loses the signal that distinguishes progress from shuffling. The
ramp is symmetric in plies-to-end, so it does not tax either side's longer wins.

**Data pipeline**: self-play generation (`src/data_generation.py`, multiprocess,
curriculum start positions, deck-based starts, frozen-opponent alternating modes)
→ tensor conversion with leak-free game-level train/val/test splits
(`src/data_processor.py`) → training with per-side checkpoint selection
(minimum-over-sides policy accuracy + value sign accuracy, so a checkpoint cannot
be rescued by one color while the other collapses).

## Getting started

Requires Python 3.10+ and a CUDA-capable GPU (optional but strongly recommended
for NN search).

```bash
pip install -r requirements.txt
```

Play against the engine in the terminal:

```bash
python src/play.py --color black --model models/fresh_start_v19/best_value_net.pt
```

or open `src/play.ipynb` for a widget UI with model selection, curriculum decks,
and automatic game recording.

## Training pipeline

Generate self-play data:

```bash
python src/data_generation.py --num-games 800 --simulations 400 \
    --curriculum --curriculum-live-results --record-all-plies \
    --seed 42 --output-dir data/raw/my_run
```

Convert to training tensors (game-level stratified split, mirror augmentation):

```bash
python src/data_processor.py --raw-dir data/raw/my_run \
    --output-dir data/processed/my_run --seed 42
```

Sanity-check a merged corpus before spending a training run on it:

```bash
python tools/pretrain_check.py data/raw/my_run --reference data/raw/previous_run
```

`data/processed/` intentionally retains only the active v19_B-derived corpus;
see `data/processed/README.md` and the cleanup manifest under `logs/archive/`
before regenerating concluded experiment datasets.

Train:

```bash
python src/train.py --data-dir data/processed/my_run --model-dir models/my_model \
    --target game_result --value-head wdl --epochs 30 --seed 42
```

Ramp-target training uses the scalar head:
`--value-head scalar` after processing with `--value-floor 0.5 --value-horizon 60`.

LC0-inspired candidates are separate, opt-in experiments: `--moves-left-head`
adds masked Huber regression on decisive trusted trajectories;
`--legal-policy-mask` excludes illegal logits from policy loss and top-1;
`--policy-head attention` selects the compact policy head; and
`--ema-decay 0.999` validates and checkpoints an exponential weight average.
`--promotion-policy` enables the backward-compatible 4288-logit policy that
keeps q/r/b/n promotions distinct; its corpus must be built with
`data_processor.py --promotion-aware-policy`. `--train-promotion-head-only`
lifts an existing checkpoint and freezes every legacy parameter and BatchNorm
buffer. Checkpoint selection can be bounded with
`--max-policy-ce-regression` and `--max-side-top1-drop`.
None of these flags changes the default recipe.

Current promoted release and formal bar (2026-08-05):
`models/fresh_start_v20/best_value_net.pt`. It is the preserved
`lc0b_attention_ema_wide64` checkpoint: attention plus EMA on the exact v19_B
recipe, with the attention query/key width increased from 32 to 64. It improved
both colors over the approved 32-channel model in two calibrated reads, and the
owner promoted it as v20. Historical checkpoints and its campaign copy remain
intact. An 80-game equal-settings self-match at 400 sims measured a White score
of 0.6938 and Black score of 0.3063 (47 White wins, 16 Black wins, 17 draws),
so the current operating point remains materially White-skewed.

Preview one complete bootstrap generation without writing anything:

```bash
python src/iterate.py --dry-run
```

Then run it. Promotion is deliberately explicit and is allowed only after the
full binding gate; it advances the bootstrap champion pointer but does not
create a numbered release or bypass the owner's release playtest.

```bash
python src/iterate.py --generations 1 --promote-on-pass
```

The resumable state machine is `generate → reanalyze → process → compose →
train → checkpoint_screen → offline_gate → binding_gate →
high_fidelity_gate → self_skew → promote`. Every unique checkpoint preserved
by validation receives a same-openings arena screen; the best worst-color result
is only a nomination for the normal gates. The offline comparison is advisory
by default: it records held-out policy/value warnings, but actual games decide
rejection. A binding winner must also improve both calibrated colors in an
independent 80-game, 800-simulation confirmation before promotion. The legacy
hard offline behavior is available with `--reject-on-offline-regression`. Each generation
has immutable state, command logs, and reports under `iterations/gen_NNNN/`.
Resume an interrupted generation with the original experiment arguments plus
`--resume`; changing a training or data argument is rejected. `--through-phase`
can stop safely after any phase.

Self-play is augmented by ordinary-position deep search, not tactical rules:
`tools/reanalyze.py` selects positions where deeper champion search most
changes the policy/value and writes policy-only teachers, with 60% of the
pipeline's teacher budget reserved for Black. `tools/compose_processed.py`
combines the exact immutable v19_B/V20 anchor, recent accepted replay, and the
current generation while preserving validation/test membership. Processed
self-play is registered in `accepted_data.json` before candidate training, so
useful champion data survives a rejected model. The training split is
deterministically smoothed across side, true outcome, and corpus-derived
material-phase quantiles; this is general replay balancing, not a tactical
rule. `--continue-after-reject` permits explicit multi-generation data
accumulation while leaving the champion unchanged.

Reanalysis teachers inherit their source game's split, so an alternate deep
policy for a position can never cross from training into validation/test.
Reanalysis, generation, gates, and self-skew matches all have bounded
no-progress timeouts; every generated batch must meet its configured saved-game
rate before processing. Reanalysis and replay composition publish completed
directories atomically, accepted replay hashes every required artifact, and a
run-root lock prevents concurrent bootstrap loops. Large replay position and
policy arrays are memory-mapped during pipeline training to keep later
generations inside host-memory limits.

Game-playing phases use the measured eight-worker default on the 5060 Ti. That
setting delivered 7.11 decisions/s versus 5.39 at four workers; twelve workers
only reached 7.38 and fourteen exhausted GPU memory. Each NN worker owns a CUDA
context, so the worker count stays explicit and bounded rather than following
the host CPU count.

Pipeline training evaluates the incumbent on the same validation rows before
epoch one. Validation preserves checkpoints using worst-color policy and
value-sign gains over that fixed baseline; the checkpoint arena then tests
every preserved model and ranks by its calibrated worst color. Regression
guards remain fixed to the incumbent rather than walking between epochs. If no epoch is safe, training
emits `selection_rejected.json` and the generation becomes
`rejected_training`. The
moves-left head exists as an opt-in experiment but is off in the first pipeline
generation so infrastructure and architecture changes are not conflated.

The first fixed-architecture bootstrap successor to clear all automated gates
is
`models/candidates/bootstrap_gen5_teacher3200_full/selected_epoch_002.pt`.
It uses the full replay with 4x policy-only teachers searched at 3200
simulations. The calibrated 80x800 confirmation measured +0.0125 Black,
+0.1375 White, and +0.075 overall versus V20. A separate full binding gate
passed both V20 seeds (initial 0.650 overall / 0.825 White / 0.475 Black;
confirmation 0.550 / 0.675 / 0.425), plus ramp and heuristic retention. Its
80-game self-match reduced pooled White skew from V20's 0.6938 to 0.5875.
This is a candidate for the owner's release playtest, not a numbered V21.
The isolated moves-left auxiliary follow-up did not supersede it: epoch six
passed a direct 80x800 A/B but failed the fresh V20 Black floor at 0.375, and
the earlier epoch three lost 0.050 Black in its independent A/B confirmation.
The head remains available but off by default.

## Evaluation

Two complementary automated measurements, both reporting per-side results:

**Fixed anchor benchmark** — every model is scored against the heuristic MCTS, a
permanent yardstick that never moves, so progress is absolute rather than relative
to the previous model:

```bash
python src/benchmark.py --model models/my_model/best_value_net.pt --games 20 --sims 400
```

**Head-to-head matches** — parallel candidate-vs-incumbent games with sampled
openings (two deterministic engines at temperature 0 would replay one game N
times):

```bash
python tools/match.py --model-a models/my_model/best_value_net.pt \
    --model-b models/fresh_start_v20/best_value_net.pt --games 20
```

Supporting tools: `tools/model_diff.py` (cheap offline candidate-vs-incumbent
comparison on identical positions — informational only; offline metrics and play
strength are demonstrably decoupled in this project), `tools/heuristic_ab.py`
(same-side paired A/B for heuristic changes), `tools/promotion_probe.py`
(promotion prevention / defender survival), and deck builders
(`tools/make_human_deck.py`, `tools/make_promo_deck.py`) that turn recorded games
into targeted start-position decks.
`tools/search_sweep.py` runs resumable, non-binding one-factor PUCT sweeps against
the same checkpoint, ranks Black first, and retains the White/aggregate floors.

### Model lifecycle

Numbered models under `models/fresh_start_vN/` are promoted releases, not training
runs. A candidate earns a number through automated evidence — head-to-head and
anchor scores with per-side floors, never aggregates alone — plus a final human
playtest. Rejected candidates are archived under `models/rejected/` without
consuming the version number; probes live under `models/experiments/`. Evaluation
thresholds are never relaxed to let a candidate through.

## Repository layout

```
src/
  config.py            # tunables (build recipes documented at the top)
  curriculum.py        # curriculum start positions and tier structure
  monster_chess.py     # rules: atomic (m1,m2) API + half-move search API
  mcts.py              # batched PUCT (NN) and sequential UCB1 (heuristic)
  evaluation.py        # heuristic eval, NNEvaluator, HybridEvaluator
  encoding.py          # board/move tensor encoding, mirror augmentation
  data_generation.py   # multiprocess self-play generation
  data_processor.py    # raw JSONL -> training tensors, leak-free splits
  train.py             # network, training loop, checkpoint selection
  benchmark.py         # fixed heuristic-anchor benchmark
  iterate.py           # resumable self-play/reanalysis/replay/train/gate loop
  scripted_mate.py     # deterministic K+heavies-vs-bare-K conversion (verified)
  play.py / play.ipynb # play against the engine (terminal / notebook)
tools/                 # gate, matches, probes, corpus and deck builders
  gate.py              # the promotion protocol, thresholds as constants
  match.py             # head-to-head, the single match JSON schema
  value_side_bias.py   # per-side value calibration on held-out games
  promotion_defense_probe.py  # search behaviour + conversion from a deck
  reanalyze.py         # general deep-search policy teachers
  compose_processed.py # immutable processed-corpus replay composition
  phase3_driver.py     # train+gate a set of corpus arms unattended
tests/                 # contract tests
benchmarks/            # benchmark and match JSON history
data/                  # raw games, processed tensors, start-position decks
models/                # checkpoints (gitignored)
```

Finished experiment drivers and reports are removed from the tree when a run
concludes; git history is the archive (`git log --diff-filter=D --name-only`).

## Tests

```bash
python -m unittest discover -s tests
```

498 contract tests cover the rules (including the unconditional-king-capture edge
cases), search invariants, encoding round-trips, data-pipeline contracts, the
training CLI schema, the corpus gates, the promotion protocol's thresholds, and
several hazards that have produced wrong numbers here before (ramp labels are
positional, so filtering records silently relabels survivors; match seeds closer
than the game count replay the same games; worker defaults derived from
`cpu_count()` crash CUDA init on this box). CI runs the suite on push
(`.github/workflows/contract-tests.yml`).
