# Monster Chess NN

Current project state, model identities, latest results and next steps:
[consolidated handoff](HANDOFF.md). **Current release: v28**
(`models/bootstrap_v28`, gen50 epoch14 with the September 17 calibrated value
head, promoted September 25). Plans and results for every campaign are indexed
in [docs/README.md](docs/README.md).

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
variant its character. Black holds far more material and wins by consolidating;
White has no chances once that happens, so White lives on a fast attack. That
tension is the whole game, and it is why all evaluation tooling reports per-side
results rather than a single aggregate.

**Who is better depends on search depth *and* on which engine is playing.**
Measured 2026-08-18 on the then-strongest model playing itself, White dominated
at depth: Black won 33% of games at 400 simulations and only 6% at 3,200. That
result is era-specific and has since narrowed sharply. White's advantage comes
largely from *choosing the opening*, and successive models have eroded it —
under free play a model playing itself scores White 0.8717 (v24), 0.7933
(gen33), 0.5833 (gen38). By generation 41, self-play at 6,400 simulations
produced 13 draws to 2 White wins and 2 Black wins.

Two cautions follow, and the tooling here is built around both: a per-colour
score is meaningless without a *measured* baseline for the same conditions, and
a result at one simulation count is a statement about that count, not about the
game.

## How the engine works

An experimental, opt-in CPU value network plus native alpha-beta search is
described in [SEARCH_FIRST_EXPERIMENT.md](docs/experiments/search_first/SEARCH_FIRST_EXPERIMENT.md). It does
not replace the standard engine or release models; strength validation is ongoing.

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

`data/processed/` contains historical corpora, immutable generation increments,
and composed replay snapshots through gen50. Some are pinned inputs to completed
experiments; do not delete them merely because their generation is old. See the
current handoff before retiring or regenerating datasets.

Train:

```bash
python src/train.py --data-dir data/processed/my_run --model-dir models/my_model \
    --target game_result --value-head wdl --epochs 30 --seed 42
```

Ramp-target training uses the scalar head:
`--value-head scalar` after processing with `--value-floor 0.5 --value-horizon 60`.

LC0-inspired candidates are separate, opt-in experiments: `--moves-left-head`
adds masked Huber regression on decisive trusted trajectories;
`--train-moves-left-head-only` makes an exact frozen lift of a resumed model;
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

Moves-left can now be consumed by Python or native PUCT through a bounded,
opt-in utility. The match harness exposes it independently per side with
`--moves-left-a` / `--moves-left-b` and tunable max-effect, threshold, and
slope flags. The first exact Gen9 lift learned a real length signal but was a
clean playing null (0.5062 over 80 paired games; 0/16 conversion moves changed),
so it remains infrastructure rather than a successor. See `REPORT.md` §32.

Current numbered release: **`models/bootstrap_v27/best_value_net.pt`**, promoted
with owner authorization on 2026-09-07 from gen46's selected epoch7 checkpoint.
The same milestone promotion created v25 from gen42 and v26 from gen45. Each
release has a source hash and evidence manifest; older models are preserved.
`gate.BAR` is `vs_v27`, and the bootstrap champion pointer selects v27.

Gen47's revised stateful/league recipe and queued end-to-end preflight are in
[GEN47_RUN.md](docs/experiments/gen47/GEN47_RUN.md). Use its adapter entry point to run/resume gen47,
not the historical plain-iterate example below. During gen46's ongoing checks,
legacy CLI config fallbacks remain v24 to preserve runtime identity; pass the
v27 model path explicitly for CLI play. See `HANDOFF.md` for current operations.

Preview one complete bootstrap generation without writing anything:

```bash
python src/iterate.py --dry-run
```

Then run one generation with an explicit generating model. The default does
not promote anything; a passing candidate remains available for owner review.

```bash
python src/iterate.py --generations 1 --incumbent models/candidates/bootstrap_main_gen_0044/best_value_net.pt --seed 3173
```

The resumable state machine is `generate → reanalyze → process → compose →
train → checkpoint_screen → offline_gate → binding_gate →
high_fidelity_gate → self_skew → promote`. Up to eight representative saved
epochs receive 40-game probes at 1,600 simulations. Two aggregate leaders,
plus Black-best and offline-best safeguards, advance to 200-game screens at
3,200. Ranking favors aggregate score after a calibrated color-collapse guard;
the nominee must still pass an independent binding test. Free probes and full
screens use disjoint RNG blocks, not prescribed board positions.

The default `--gate-backend sampled` uses a fresh 400-game actual-color
self-par, then two independent H2H legs of 200 games per candidate color at
3,200 simulations. Repeated opening draws retain their sampled frequency;
deduplication and novelty are reported separately. Each H2H leg must score above
50%, with each color at least incumbent same-color par minus .05. A PASS is an
operational point-estimate screen, not proof that both colors improved.
Missing scheduled evidence is INCONCLUSIVE. This backend skips the redundant
legacy high-fidelity gate. See [SAMPLED_GATE_PROTOCOL.md](docs/protocols/SAMPLED_GATE_PROTOCOL.md).

`--gate-backend free` retains the endpoint-uniform v2 instrument;
`--gate-backend legacy` retains the older book-compatible gates. Old reports
and completed generation states are not silently migrated between protocols.
The offline comparison remains advisory unless
`--reject-on-offline-regression` is supplied. Each generation
has immutable state, command logs, and reports under `iterations/gen_NNNN/`.
Resume an interrupted generation with the original experiment arguments plus
`--resume`; changing a training or data argument is rejected. `--through-phase`
can stop safely after any phase.

Self-play is augmented by ordinary-position deep search, not tactical rules:
`tools/reanalyze.py` selects positions where deeper champion search most
changes the policy/value and writes policy-only teachers, with 60% of the
pipeline's teacher budget reserved for Black. Production defaults are pinned
in `configs/bootstrap_generation_only.json`: 1,000 free games plus 400
book-seeded games at 700 simulations, then sample 20,000 ordinary positions
and retain 10,000 teachers searched at 3,200 simulations. Training is
fresh with the successful scratch optimizer recipe; it does not resume incumbent
weights. Every processed generation must pass an exact teacher census before
it can be registered or composed: one-row retention, mirrored row count,
60/40 side split, policy-only value mask, and source-linked split membership.
`tools/compose_processed.py`
combines eight accepted generation increments, including the current one,
while preserving validation/test membership. There is no v19-era anchor,
human-game training source, or outside corpus. Teacher policy weight multiplier
is four, with zero teacher value weight. Processed
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
generations inside host-memory limits. Sampled-backend deep reanalysis also
keeps a durable per-position search journal outside the raw training tree.
Matches and checkpoint screens retain provenance-checked per-game journals;
resuming schedules only missing tasks. Completed evidence is hash-validated.

For legacy evaluation only, pass `--gate-backend legacy --book books/<pinned-book>.json`
to reserve disjoint paired blocks for
checkpoint screening, the binding gate, high-fidelity confirmation, and
self-skew automatically. Books are pinned artifacts: changing one silently
invalidates comparison against every score measured under the old one. The
legacy book campaigns used 3,000-entry p8 books drawn equally from six models
spanning v22 to the working bar. These evaluation books are distinct from
the book-seeded subset retained in generation-only training.

**A per-colour score is meaningless without its block's baseline.** Block colour
bias runs to +-0.056: a model played against *itself* -- true value 0.5000 by
construction -- has scored White 0.4437 on one block and 0.3000 on another from
the same book. The gate therefore plays the bar against itself on the bar leg's
own block and reports each binding leg against that baseline (`REPORT.md` §48).

Game-playing phases use the measured eight-worker default on the 5060 Ti. That
setting delivered 7.11 decisions/s versus 5.39 at four workers; twelve workers
only reached 7.38 and fourteen exhausted GPU memory. Each NN worker owns a CUDA
context, so the worker count stays explicit and bounded rather than following
the host CPU count.

Pipeline training preserves epoch snapshots and selects an offline reference
using decisive-position validation metrics. Play-based selection then tests a
bounded representative shortlist; it does not reject trained models solely on
offline accuracy. The recipe uses 30 epochs, patience ten, AdamW at .002,
three warmup epochs and EMA .999. Promotion requires explicit
`--promote-on-pass`, advances only the working champion pointer, and never
creates a numbered release or bypasses the owner's release playtest. The
moves-left head remains an opt-in experiment, off in this controlled recipe.

Training hyperparameters can be searched with multi-fidelity Optuna trials
whose objective is calibrated arena play rather than validation loss. The
tuner accepts the current corpus, bar, architecture, EMA setting, and a paired
book; every trial at a fidelity rung sees the same openings, while any winner
must still use an untouched block for the normal binding gate. Example for the
post-Gen10 state:

```bash
python tools/tune_training.py --trials 8 --timeout-hours 10 \
  --study-name gen10_training_hpo \
  --storage logs/hpo/gen10_training_hpo.sqlite3 \
  --model-root models/tuning/gen10_training_hpo \
  --log-root logs/hpo/gen10_training_hpo \
  --report-prefix hpo_gen10_training \
  --data data/processed/bootstrap_replay_main_gen_0010 \
  --bar models/fresh_start_v22/best_value_net.pt \
  --policy-head attention --policy-attention-channels 64 \
  --ema-decay 0.999 --memory-map-data \
  --book books/gate_mixed_v21b_gen7_gen9_p16_20260815.json \
  --book-offset 680
```

Historical bootstrap milestone: the first fixed-architecture successor to
clear all automated gates was
`models/candidates/bootstrap_gen5_teacher3200_full/selected_epoch_002.pt`;
that lineage became V21.
It uses the full replay with 4x policy-only teachers searched at 3200
simulations. The calibrated 80x800 confirmation measured +0.0125 Black,
+0.1375 White, and +0.075 overall versus V20. A separate full binding gate
passed both V20 seeds (initial 0.650 overall / 0.825 White / 0.475 Black;
confirmation 0.550 / 0.675 / 0.425), plus ramp and heuristic retention. Its
80-game self-match reduced pooled White skew from V20's 0.6938 to 0.5875.
Those figures explain the V21 promotion; they are not the current candidate.
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
    --model-b models/fresh_start_v22/best_value_net.pt --games 20
```

Matches run on one of two **instruments**, and they do not measure the same
thing:

- **Book** (`--book`) plays a fixed set of openings, identical for every
  pairing. Transitive, tighter error bars, same positions for everyone — but it
  forbids the opening choice that is most of White's game.
- **Free** (no book) lets both models pick their own openings. This is the game
  as actually played, and the instrument gates now use — but 40–76% of its games
  are *exact replays* of one another, so results must be deduped on the opening
  record (`--game-log` writes it) before they count as a sample.

A 45-pairing round robin on both instruments (2026-09-04, 36,000 games) found
they disagree about ordering, not merely scale: the post-gen33 cohort sits
135–246 free Elo above its predecessors while book compresses the same gap into
8–28 Elo. See `HANDOFF.md` §2.

Two further flags matter for honest measurement. `--book-temp-plies N` samples
N plies after each book position, so a repeated entry yields *different* games —
without it a book line is n=1, and ~45% of single-line verdicts flip on
resampling. Book results also carry roughly **±20 Elo of block-to-block noise**
that the reported SE does not include, so compare models on the *same* block or
not at all.

**Gates** run on free play: `tools/gate_free.py` plays a candidate against the
bar, deduped, stopping on unique games or a wall-clock budget, with a cached
self-match "par" leg for the bar so the per-side check has a real baseline —
free-play par is nowhere near 0.50. The older book gate `tools/gate.py` is
retained unchanged for continuity with historical results.

The current version requires per-color and unseen-confirmation coverage, saves
finished games incrementally, and returns **INCONCLUSIVE** when its budget cannot
supply the evidence. See [FREE_GATE_PROTOCOL.md](docs/protocols/FREE_GATE_PROTOCOL.md) for the
scoring contract, provenance/resume commands and production bootstrap recipe.
`src/iterate.py` defaults to that free gate and generation-only replay; legacy
book evaluation requires `--gate-backend legacy`.

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
campaigns/             # finished root campaign drivers, frozen (see its README)
docs/                  # protocols, per-campaign plans/results, history ledgers
benchmarks/            # benchmark and match JSON history
data/                  # raw games, processed tensors, start-position decks
models/                # checkpoints (gitignored)
```

Retain experiment drivers, plans and reports required by saved provenance.
Several campaigns hash all `tools/*.py` and `tests/*.py`; moving or removing
those files invalidates exact resume, so resume a finished campaign from a git
worktree at the commit it ran on (September work: `b46ce1c`). Git holds source,
docs and small evidence summaries; weights, arrays, per-game task records and
JSONL journals stay on disk only. `benchmarks/` can also contain trained
checkpoints. Only regenerable caches were deleted in the September 25 cleanup;
completed logs were archived, not discarded. See [CLEANUP_20260925.md](docs/history/CLEANUP_20260925.md).

## Tests

```bash
python -m unittest discover -s tests
```

Contract tests cover the rules (including the unconditional-king-capture edge
cases), search invariants, encoding round-trips, data-pipeline contracts, the
training CLI schema, the corpus gates, the promotion protocol's thresholds, and
several hazards that have produced wrong numbers here before (ramp labels are
positional, so filtering records silently relabels survivors; match seeds closer
than the game count replay the same games; worker defaults derived from
`cpu_count()` crash CUDA init on this box). CI runs the suite on push
(`.github/workflows/contract-tests.yml`).
