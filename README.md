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
or a win/draw/loss classifier. Checkpoint architecture is inferred from the state
dict at load time, so older checkpoints with different widths or encodings remain
loadable.

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
python src/play.py --color black --model models/fresh_start_v17/best_value_net.pt
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

Train:

```bash
python src/train.py --data-dir data/processed/my_run --model-dir models/my_model \
    --target game_result --value-head wdl --epochs 30 --seed 42
```

Ramp-target training uses the scalar head:
`--value-head scalar` after processing with `--value-floor 0.5 --value-horizon 60`.

Run one full generate → process → train → gate cycle:

```bash
python src/iterate.py --generations 1
```

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
    --model-b models/fresh_start_v17/best_value_net.pt --games 20
```

Supporting tools: `tools/model_diff.py` (cheap offline candidate-vs-incumbent
comparison on identical positions — informational only; offline metrics and play
strength are demonstrably decoupled in this project), `tools/heuristic_ab.py`
(same-side paired A/B for heuristic changes), `tools/promotion_probe.py`
(promotion prevention / defender survival), and deck builders
(`tools/make_human_deck.py`, `tools/make_promo_deck.py`) that turn recorded games
into targeted start-position decks.

Anywhere a `.pt` path is accepted (benchmark, match, the notebook), a
`router.json` phase-router spec is too: a composite engine that plays different
game phases with different checkpoints (`src/router.py`).

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
  config.py            # tunables + curriculum start positions
  monster_chess.py     # rules: atomic (m1,m2) API + half-move search API
  mcts.py              # batched PUCT (NN) and sequential UCB1 (heuristic)
  evaluation.py        # heuristic eval, NNEvaluator, HybridEvaluator
  encoding.py          # board/move tensor encoding, mirror augmentation
  router.py            # phase router (composite engines)
  data_generation.py   # multiprocess self-play generation
  data_processor.py    # raw JSONL -> training tensors, leak-free splits
  train.py             # network, training loop, checkpoint selection
  benchmark.py         # fixed heuristic-anchor benchmark
  iterate.py           # generate -> process -> train -> gate loop
  scripted_mate.py     # deterministic K+heavies-vs-K conversion (verified)
  play.py / play.ipynb # play against the engine (terminal / notebook)
tools/                 # matches, corpus gates, diffs, deck builders
tests/                 # contract tests
benchmarks/            # benchmark and match JSON history
data/                  # raw games, processed tensors, start-position decks
models/                # checkpoints (gitignored except router specs)
```

Finished experiment drivers and reports are removed from the tree when a run
concludes; git history is the archive (`git log --diff-filter=D --name-only`).

## Tests

```bash
python -m unittest discover -s tests
```

121 contract tests cover the rules (including the unconditional-king-capture edge
cases), search invariants, encoding round-trips, data-pipeline contracts, the
training CLI schema, and the corpus gates. CI runs the suite on push
(`.github/workflows/contract-tests.yml`).
