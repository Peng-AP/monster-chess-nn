# Monster Chess NN

Neural-network + MCTS engine for the Monster Chess variant: White has a king and
four pawns but moves **twice** per turn; Black has a full standard army and moves
once. The game ends when a king is captured — a king capture wins
unconditionally, even if the capturing side would be "in check".

Established balance conclusion: **Black wins with correct play**, so Black's
conversion strength is the project's primary quality signal.

## Layout

```
src/
  config.py          # all tunables (~15 knobs) + curriculum start positions
  monster_chess.py   # rules, atomic (m1, m2) API + half-move search API
  mcts.py            # batched PUCT (NN) + sequential UCB1 (heuristic)
  evaluation.py      # heuristic eval, NNEvaluator, HybridEvaluator
  encoding.py        # fen_to_tensor, policy encoding, mirror augmentation
  data_generation.py # self-play game generation (multiprocess)
  data_processor.py  # raw JSONL -> training tensors (flat, game-level split)
  train.py           # resnet policy + scalar/WDL/hybrid value training
  benchmark.py       # fixed heuristic-anchor yardstick (JSON history in benchmarks/)
  iterate.py         # loop: generate -> process -> train -> gate -> archive
  scripted_mate.py   # deterministic Black K+heavies conversion (demo games)
  verify_scripted_mate.py
  make_blackfocus_starts.py  # backward-chained Black-won start FENs
  play.py / play.ipynb       # play against the engine (terminal / notebook)
tools/               # one-off calibration tools (bruteforce gen, human eval)
tests/               # contract tests (run: py -3 -m unittest discover -s tests)
benchmarks/          # anchor benchmark JSON history
```

## Commands

Generate self-play data (heuristic eval, curriculum starts, live results):

```bash
py -3 src/data_generation.py --num-games 800 --simulations 400 \
    --curriculum --curriculum-live-results --record-all-plies \
    --seed 42 --output-dir data/raw/my_run
```

Process raw games into tensors:

```bash
py -3 src/data_processor.py --raw-dir data/raw/my_run --output-dir data/processed/my_run --seed 42
```

Train (outcome-grounded WDL value):

```bash
py -3 src/train.py --data-dir data/processed/my_run --model-dir models/my_model \
    --target game_result --value-head wdl --epochs 30 --seed 42
```

Current v18 processing uses `--value-discount-mode near_mate` with horizon 10
and floor 0.97. Full-game progress is not a training or search objective:
Black's sound conversions are systematically longer than White's, which made
that target side-biased. Current 17-channel data encodes rank and pawn progress
symmetrically; 15-channel v16/v17 checkpoints remain loadable.

Benchmark against the fixed heuristic anchor (the project yardstick):

```bash
py -3 src/benchmark.py --model models/my_model/best_value_net.pt --games 20 --sims 400
```

Run one full loop generation with gated promotion:

```bash
py -3 src/iterate.py --generations 1
```

Verify the scripted-mate conversion algorithm vs MCTS White:

```bash
py -3 src/verify_scripted_mate.py --games 16 --white-sims 200 --seed 7
```

## Model promotion

Numbered models are promoted releases, not training-run counters. The incumbent
keeps its version until a candidate demonstrates a concrete improvement through
side-aware automated evidence and the project owner's final play-strength gate.
A rejected candidate is archived without consuming the target version number.

Automated matches establish eligibility only; they never promote a model by
themselves. The currently running replacement remains a v18 candidate until
owner approval, while v17 remains the incumbent.

## v18 learning cleanup

`overnight_human_v18.py` rebuilds the corpus without promotion injection. It
keeps all v17 focus outcomes, includes each human game once, trains policy only
from the eventual human winner in human games, and uses outcome-grounded WDL
learning with only a near-mate tiebreak. Its anchor and incumbent matches are
informational, not hard specialist gates.

## Historical v17 promotion experiment

Generated promotion training is White-runner-only. The mixed probe file must
always be source-filtered; generated `promo_black_runner` games are forbidden.

```bash
py -3 src/data_generation.py --num-games 160 --simulations 400 \
    --start-fen-file data/start_fens/promo_races_probe.jsonl \
    --start-fen-source promo_white_runner --record-all-plies \
    --output-dir data/raw/promo_races_raw
py -3 src/promotion_data.py data/raw/promo_races_raw data/raw/promo_races \
    --expected-start-source promo_white_runner
```

Preparation keeps every outcome/value target, but gives failed Black-defense
moves policy weight zero. `tools/pretrain_check.py` rejects Black-runner
contamination or incorrect promotion policy weights. `tools/promotion_probe.py`
compares prevention, defender-king survival, and game score separately.

`overnight_human_v17.py` preserves that experiment for reproducibility. It is
not the current training recipe.

Play against a model: open `src/play.ipynb` (widget UI, saves games to
`data/raw/human_games/`) or `py -3 src/play.py`.

## Promotion gate (iterate.py)

A candidate is promoted to `models/best_value_net.pt` only if it

1. scores >= 0.55 against the incumbent (both colors, temperature 0, no noise), and
2. does not regress against the heuristic anchor by more than the configured epsilon.

Every candidate and its gate report are archived under `models/candidates/gen_<N>/`;
the loop history lives in `models/iterate_history.json`.

## History

The engine went through a documented rework (`REWORK_PLAN.md`): search fix (D1),
outcome-grounded value targets (D2), half-move factorization for White, a core
rules correction (unconditional king capture, 2026-07-04), a side-to-move eval
clamp fix (2026-07-05), and the Phase 5 deletion of the compensation machinery
that predated those fixes.
