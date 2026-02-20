# Monster Chess NN

Neural-network + MCTS engine for the Monster Chess variant (White gets two consecutive moves, Black gets one).

This repository now includes a full self-play -> processing -> training -> gated-promotion loop, plus baseline and human-game evaluation tools.

## Current Project State

As of 2026-02-19 (local machine snapshot):

- Highest detected generation directory: `data/raw/nn_gen24`
- Processed dataset shape: `(86990, 8, 8, 15)`
- Processed splits: `train=69592`, `val=8699`, `test=8699`
- Human game files: `25` in `data/raw/human_games`
- Current model: `models/best_value_net.pt`
- Model SHA256: `D54E3117A7A28BD4C0C678FE2D3FAB231AA8943660CFBC1FF6958E9B70BC815B`

Important behavioral note:

- White-side play remains much stronger than Black-side play; side-aware gating can still reject candidates that only improve White.

## What Is Implemented

- Game-level train/val/test splitting to prevent leakage (`src/data_processor.py`)
- Deterministic seeding and run metadata for training/iteration runs (`src/train.py`, `src/iterate.py`)
- True MSE/MAE reporting alongside value power loss (`src/train.py`)
- Candidate vs incumbent arena gating with side-aware thresholds (`src/iterate.py`)
- Alternating training mode with frozen/pool opponents (`src/iterate.py`, `src/data_generation.py`)
- Human-seeded self-play starts from recorded human positions (`src/iterate.py`, `src/data_generation.py`)
- AdamW with decoupled weight decay groups (`src/train.py`)
- Gradient clipping (`src/train.py`)
- Fine-tune from incumbent with LR warmup (`src/train.py`, `src/iterate.py`)
- Baseline snapshot tool (`src/baseline_snapshot.py`)
- Human game evaluation tool (`src/human_eval.py`)

## Remaining High-Priority Gaps

- Sliding window is still fixed-count generation based (`SLIDING_WINDOW=2`) instead of position-budget based (`src/config.py`)
- No FPU reduction in PUCT path yet (`src/mcts.py`)
- Tactical position filtering is not yet configurable (check positions are always filtered) (`src/data_generation.py`)
- Policy head bottleneck remains (2 channels before FC projection) (`src/train.py`)

## Repository Layout

- `src/config.py`: central hyperparameters and curriculum FENs
- `src/monster_chess.py`: Monster Chess rules wrapper around `python-chess`
- `src/mcts.py`: UCB/PUCT MCTS search
- `src/evaluation.py`: heuristic evaluator and neural evaluator wrapper
- `src/data_generation.py`: self-play game generation
- `src/data_processor.py`: JSONL -> tensor dataset conversion
- `src/train.py`: dual-head residual network training
- `src/iterate.py`: automated generation/process/train/gate loop
- `src/play.py`: CLI human vs AI gameplay
- `src/baseline_snapshot.py`: stage-0 baseline capture tool
- `src/human_eval.py`: post-run human-game evaluation

## Setup

Tested environment:

- Windows + Python 3.13
- CUDA-enabled PyTorch

Install the core dependencies:

```bash
py -3 -m pip install chess numpy torch tqdm
```

Optional notebook/UI dependencies:

```bash
py -3 -m pip install jupyter ipywidgets
```

Note: `requirements.txt` contains legacy packages and does not fully reflect the current PyTorch-based pipeline.

## Common Commands

### 1) Capture a baseline snapshot

```bash
py -3 src/baseline_snapshot.py
```

### 2) Generate self-play data

```bash
py -3 src/data_generation.py --num-games 200 --simulations 200 --use-model models/best_value_net.pt --output-dir data/raw/nn_genX
```

### 3) Process raw data

```bash
py -3 src/data_processor.py --raw-dir data/raw --output-dir data/processed --keep-generations 3
```

### 4) Train a model (fine-tuning example)

```bash
py -3 src/train.py --data-dir data/processed --model-dir models/candidates/manual_run --target mcts_value --epochs 12 --resume-from models/best_value_net.pt --warmup-epochs 3 --warmup-start-factor 0.1
```

### 5) Run iterative training with gating

```bash
py -3 src/iterate.py --iterations 2 --games 180 --curriculum-games 220 --black-focus-games 260 --human-seed-games 200 --simulations 120 --curriculum-simulations 50 --black-focus-simulations 100 --human-seed-simulations 120 --epochs 12 --warmup-epochs 3 --warmup-start-factor 0.1 --position-budget 220000 --position-budget-max 280000 --alternating --opponent-sims 140 --pool-size 6 --arena-games 80 --arena-sims 80 --arena-workers 4 --gate-threshold 0.54 --gate-min-other-side 0.42 --human-seed-dir data/raw/human_games --human-seed-side auto --seed 20260219 --human-eval
```

### 6) Play against the model

```bash
py -3 src/play.py --model models/best_value_net.pt --color black --sims 200 --save-data
```

### 7) Evaluate recorded human games

```bash
py -3 src/human_eval.py --human-dir data/raw/human_games --model models/best_value_net.pt
```

## Documentation Map

- `IMPROVEMENT_PLAN.md`: living status of the improvement roadmap
- `IMPROVEMENT_EXECUTION_ORDER.md`: next implementation order from current repo state
- `TRANSFER_HANDOFF_2026-02-19.md`: transfer handoff notes (historical + current-state guidance)
- `context.txt`: compact technical context summary for agent handoff
