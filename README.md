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
  train.py           # dual-head resnet, game_result target, optional WDL head
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

Train (outcome-grounded value target, WDL head):

```bash
py -3 src/train.py --data-dir data/processed/my_run --model-dir models/my_model \
    --target game_result --value-head wdl --epochs 30 --seed 42
```

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
