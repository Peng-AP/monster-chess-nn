# Monster Chess NN

## Project

Neural-network + MCTS engine for the **Monster Chess** variant:
- **White**: King + 4 pawns on c2-f2, takes **two** consecutive moves per turn
- **Black**: Full standard army, takes **one** move per turn
- King capture ends the game (no checkmate semantics)
- White cannot castle
- Starting FEN: `rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1`

The project implements a full self-play training pipeline: generate games with MCTS, process to tensors, train a dual-head ResNet, gate candidates against the incumbent.

## Architecture

```
iterate.py (orchestrator)
  |-> data_generation.py  (parallel MCTS self-play -> JSONL)
  |-> data_processor.py   (JSONL -> numpy tensors, game-level splits)
  |-> train.py            (fine-tune candidate dual-head ResNet)
  |-> arena evaluation    (candidate vs incumbent, color-swapped)
  \-> gate decision       (promote only if thresholds pass)
```

## File Map

| File | Role |
|------|------|
| `src/config.py` | Central hyperparameters, curriculum FENs, paths |
| `src/monster_chess.py` | Game rules wrapper around python-chess |
| `src/mcts.py` | UCB1 (heuristic) and PUCT (NN) MCTS search |
| `src/evaluation.py` | Heuristic evaluator + NN evaluator wrapper |
| `src/data_generation.py` | Parallel self-play game generation |
| `src/data_processor.py` | JSONL to tensor conversion, split integrity |
| `src/train.py` | DualHeadNet model definition and training |
| `src/iterate.py` | Automated generation/process/train/gate loop |
| `src/play.py` | Interactive CLI human vs AI |
| `src/encoding.py` | Shared move encoding utilities (move_to_index, mirror) |
| `src/baseline_snapshot.py` | Baseline metrics capture tool |
| `src/human_eval.py` | Human game evaluation diagnostics |
| `src/scripted_endgame.py` | Heuristic endgame move selection for curriculum |

## Model

**DualHeadNet**: shared residual backbone with separate value and policy heads.

- **Input**: `(batch, 15, 8, 8)` tensors (transposed from storage format `(8, 8, 15)`)
- **Backbone**: Configurable stem (64 channels) + 8 residual blocks (channels defined by `RESIDUAL_BLOCK_CHANNELS`), optional SE modules
- **Value head**: GAP -> Dense(128) -> Dense(64) -> Dense(1) + tanh. Output in `[-1, 1]` from White's perspective.
- **Policy head**: Conv1x1 (POLICY_HEAD_CHANNELS=16) -> BN -> ReLU -> Flatten -> Dense(4096 logits). Move index = `from_square * 64 + to_square`.

## Tensor Encoding

Shape: `(8, 8, 15)` — stored as numpy, transposed to `(15, 8, 8)` for PyTorch.

| Layers | Content |
|--------|---------|
| 0-5 | White piece occupancy (pawn, knight, bishop, rook, queen, king) |
| 6-11 | Black piece occupancy (same order) |
| 12 | Turn indicator (+1 White, -1 Black) |
| 13 | Move count within turn (reserved, always 0) |
| 14 | White pawn advancement gradient (0.0 at rank 2, 1.0 at rank 8) |

## Key Conventions

- **Hyperparameters**: All in `src/config.py` or exposed via CLI argparse flags
- **Deterministic seeding**: `set_seed()` in train.py, per-game seeds in data_generation.py
- **Game-level splits**: Train/val/test split at game level (never position level) to prevent leakage
- **Gating is mandatory**: Candidate models only promoted if they beat the incumbent in arena matches
- **One axis per experiment**: Change a single major parameter between iteration batches
- **Run metadata**: Every iterate run saves a JSON artifact with all parameters, gate outcomes, and timing
- **Value perspective**: Values stored from White's perspective; converted to side-to-move perspective for training
- **White actions**: `(move1, move2)` tuples; Black actions: single `chess.Move` objects
- **Policy prediction**: For White, the policy head predicts P(m1) only; m2 is handled by MCTS from the post-m1 board

## Development Workflow

### Running Tests

```bash
python3 -m pytest tests/
```

### Common Commands

```bash
# Baseline snapshot
python3 src/baseline_snapshot.py

# Generate self-play data
python3 src/data_generation.py --num-games 200 --simulations 200 \
  --use-model models/best_value_net.pt --output-dir data/raw/nn_genX

# Process raw data
python3 src/data_processor.py --raw-dir data/raw --output-dir data/processed \
  --keep-generations 3

# Train (fine-tune from incumbent)
python3 src/train.py --data-dir data/processed \
  --model-dir models/candidates/manual_run --target mcts_value \
  --epochs 12 --resume-from models/best_value_net.pt \
  --warmup-epochs 3 --warmup-start-factor 0.1

# Full iteration with gating
python3 src/iterate.py --iterations 2 --games 180 --curriculum-games 220 \
  --black-focus-games 260 --simulations 120 --curriculum-simulations 50 \
  --black-focus-simulations 100 --epochs 12 --warmup-epochs 3 \
  --warmup-start-factor 0.1 --keep-generations 3 --alternating \
  --opponent-sims 140 --pool-size 6 --arena-games 80 --arena-sims 80 \
  --arena-workers 4 --gate-threshold 0.54 --gate-min-other-side 0.42 \
  --seed 20260219 --human-eval

# Play against the model
python3 src/play.py --model models/best_value_net.pt --color black --sims 200

# Evaluate human games
python3 src/human_eval.py --human-dir data/raw/human_games \
  --model models/best_value_net.pt
```

On Windows, substitute `py -3` for `python3`.

### Data Directories (git-ignored)

- `data/raw/` — JSONL game files organized by generation (`nn_gen1/`, `nn_gen2/`, etc.)
- `data/processed/` — Numpy tensors (`positions.npy`, `splits.npz`)
- `models/` — Model checkpoints, run metadata, arena logs

## Known Debt

- `iterate.py` is ~1400 lines and handles too many responsibilities (generation scheduling, arena, adaptive curriculum, checkpointing). Candidate for decomposition.
- `evaluation.py` delays import of `train.load_model_for_inference` inside `NNEvaluator.__init__` to avoid circular dependency. A proper fix requires extracting model definitions into a separate `model.py`.
- All output uses `print()` instead of the `logging` module.
- Type hints are sparse; add incrementally when modifying functions.
- Stalemate detection is disabled in `monster_chess.py` for performance; `MAX_GAME_TURNS` prevents infinite loops.
- Move count tensor layer (13) is always zero; reserved for future per-half-move encoding.

## Documentation Map

- `IMPROVEMENT_PLAN.md` — Living status of the improvement roadmap
- `IMPROVEMENT_EXECUTION_ORDER.md` — Next implementation sequence
- `TRANSFER_HANDOFF_2026-02-19.md` — Transfer handoff notes
- `context.txt` — Compact technical context for agent handoff
