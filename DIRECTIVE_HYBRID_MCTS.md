# Directive: Hybrid MCTS — NN Policy + Heuristic Value

*From the directing instance, after Checkpoint F (self-play batch 002 failure)*

---

## The Diagnosis is Now Precise

Two self-play batches, both 0% Black wins, despite the policy head improving from 26% to 50% top-1 match with heuristic MCTS. The failure is isolated to the value head.

**The value head cannot distinguish winning from losing positions:**

| Position type | Avg value prediction | What it should be |
|---------------|---------------------|-------------------|
| White-win positions | +0.001 | > +0.3 |
| Black-win positions | -0.089 | < -0.3 |
| Draw positions | ~0.0 | -0.3 to +0.3 |

Everything evaluates near zero. The MCTS explores lines, reaches leaf positions, asks the value head "is this good or bad?", and the value head answers "I don't know" regardless. Move selection becomes effectively random despite the policy head correctly suggesting which moves to consider.

**The policy head is fixed. The value head is the sole remaining bottleneck.**

---

## Why Previous Approaches Couldn't Fix This

The value head trains on game results. For calibration to improve, the training data needs positions where `game_result` correlates with meaningful position features — the value head needs to see "positions with these features led to White wins" and "positions with those features led to Black wins," with enough variety that it learns the features rather than memorizing specific positions.

The clean corpus has 76,000 weighted positions with 74.6% Black-win and 19.2% White-win. This should be enough to learn *something* — and it did (the value distribution is spread with std=0.62, not clustered at +0.7 like the old model). But the calibration is flat: the model learned that positions vary in value without learning *which* positions are good for which side.

The likely cause: the clean corpus positions are dominated by mid-game and endgame states. The value head learned to evaluate positions that are already clearly winning or losing (Tier 4-6 curriculum) but didn't learn the subtle mid-range features that distinguish "White is slightly better" from "Black is slightly better" in contested positions. The Tier 10 opening data added coverage but couldn't fix calibration because the value head's problem is depth of understanding, not breadth of coverage.

**Self-play can't fix this because self-play depends on the value head working.** Circular. The only way to break the loop is to use an evaluator that already works for the value signal while the NN provides the policy signal.

---

## The Fix: Hybrid MCTS

### Concept

Run MCTS where:
- **Move selection (policy):** guided by the NN policy head. The policy head suggests which moves to explore. At 50% top-1 agreement with heuristic MCTS, this is a genuinely useful prior that makes the search efficient.
- **Position evaluation (value):** performed by `evaluation.py` (the heuristic evaluator). The heuristic reliably scores positions — it produced 28% Black wins from the opening, 50% from Tier 7, 67% from Tier 6. It works.

This combines the best of both systems:
- Pure heuristic MCTS uses **uniform priors** — it wastes simulations exploring bad moves because it doesn't know which moves are likely good. It works but is slow.
- Pure NN MCTS uses **learned priors** (good) but **NN value evaluation** (broken). It explores the right moves but can't tell if they lead to good positions.
- Hybrid MCTS uses **learned priors** (good) and **heuristic evaluation** (good). It explores the right moves AND correctly evaluates what it finds.

The hybrid should produce *better* games than either component alone. Better than pure heuristic MCTS because the policy focuses the search. Better than pure NN MCTS because the value signal is reliable.

### What the games train

The resulting games serve as training data for the value head specifically:

- **Game results** are ground truth value labels (as always).
- **Positions** span the full game from opening through endgame, with both sides playing well (policy head for move quality, heuristic for evaluation quality).
- **The diversity of outcomes** (expect 20-40% Black wins based on heuristic calibration data) gives the value head the correlational signal it needs: "positions with these features tend to appear in Black-win games; positions with those features tend to appear in White-win games."

Over time, the value head absorbs this signal and its calibration improves. The D2 diagnostic (avg prediction by game result) is the metric to watch.

---

## Implementation

### Step 1: Implement hybrid MCTS mode in `src/mcts.py`

Add a mode where MCTS uses two separate components:

```python
# Pseudocode for leaf evaluation in hybrid mode:
def evaluate_leaf(position):
    # Value comes from heuristic
    value = heuristic_evaluate(position)  # evaluation.py
    
    # Policy comes from NN
    policy = nn_policy_head(position)  # forward pass, policy output only
    
    return value, policy
```

**Key implementation details:**

- The NN forward pass still runs (for the policy head output), but the value head output is discarded and replaced with `evaluation.py`'s score.
- This means the forward pass cost is unchanged — you're still doing a full NN inference per leaf. The heuristic evaluation adds negligible cost on top (milliseconds).
- The policy output should be used exactly as in normal NN MCTS: softmax over legal moves, used as the prior probability in PUCT.
- The heuristic value should be normalized to [-1, +1] if it isn't already. Check `evaluation.py`'s output range — if it returns values in a different scale (e.g., centipawn-like), apply a tanh or linear rescaling.

**Flag:** Add a `--hybrid-eval` flag to `data_generation.py` and `mcts.py` that activates this mode. Keep the pure NN and pure heuristic modes available for comparison.

### Step 2: Calibration (30 minutes)

Run 10 games from the standard opening at 800 sims in hybrid mode:

```bash
py -3 src/generate_bruteforce.py \
  --num-games 10 \
  --simulations 800 \
  --use-model models/fresh_start_v2/best_value_net.pt \
  --hybrid-eval \
  --start-fen "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1" \
  --output-dir data/raw/hybrid_calibration \
  --seed 20260307
```

(Adjust command flags to match actual implementation. The key: `--hybrid-eval` uses NN policy + heuristic value.)

**Record and report:**

| Metric | Hybrid (800 sims) | Pure heuristic (1000 sims) | Pure NN (800 sims) |
|--------|-------------------|---------------------------|-------------------|
| Black wins / 10 | ? | ~3 (from Tier 10 calibration) | 0 |
| Avg game length | ? | ~25 (from Tier 10 data) | ~7.8 |
| Time per game | ? | ~35s | ~? |
| Qualitative move quality | ? | decent | policy-good, value-random |

**What we expect:** Black win rate between the heuristic (28%) and NN (0%) baselines, probably closer to heuristic. Game length should be substantially longer than pure NN. If hybrid produces ≥20% Black wins and ≥15 avg game length, proceed to production.

**What would be surprising (in a good way):** If hybrid outperforms pure heuristic — meaning the NN policy + heuristic value combination is stronger than heuristic alone. This would mean the policy head is contributing real search efficiency. Watch for this.

**What would be concerning:** If hybrid produces 0% Black wins despite the heuristic working fine on its own. This would mean the NN policy is actively harmful — steering the search toward bad moves that the heuristic can't compensate for. Unlikely given 50% top-1 agreement, but flag if it happens.

**Stop and report calibration results before proceeding.**

### Step 3: Production generation (1-2 hours)

Generate from the standard opening:

```bash
py -3 src/generate_bruteforce.py \
  --num-games 100 \
  --simulations 800 \
  --use-model models/fresh_start_v2/best_value_net.pt \
  --hybrid-eval \
  --start-fen "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1" \
  --output-dir data/raw/hybrid_selfplay_opening \
  --seed 20260308
```

Also generate from mid-game positions to improve value calibration across the full game:

```bash
py -3 src/generate_bruteforce.py \
  --num-games 50 \
  --simulations 800 \
  --use-model models/fresh_start_v2/best_value_net.pt \
  --hybrid-eval \
  --start-fen-dir data/clean/human_games \
  --output-dir data/raw/hybrid_selfplay_midgame \
  --seed 20260309
```

**Target: 150 games total (100 opening + 50 mid-game), expecting 5,000-10,000 positions.**

### Step 4: Add to clean corpus and retrain

Add hybrid games to the clean data:

```
data/clean/hybrid_opening/   ← opening games
data/clean/hybrid_midgame/   ← mid-game games
```

**Source weight:** 2 (same as bruteforce). These are higher quality than old self-play but lower than human games.

**MCTS_LAMBDA:** 0.0 (train on game result, not the heuristic value). The heuristic value guided the search, but the game result is the training target. This ensures the value head learns from ground truth outcomes, not from the heuristic's approximation.

**Retrain: warm-start from fresh_start_v2.**

```bash
py -3 src/train.py \
  --data-dir data/clean_processed \
  --model-dir models/fresh_start_v3 \
  --epochs 30 \
  --batch-size 256 \
  --lr 5e-4 \
  --warmup-epochs 3 \
  --warmup-start-factor 0.2 \
  --target game_result \
  --policy-loss-weight 1.0 \
  --grad-clip 1.0 \
  --resume-from models/fresh_start_v2/best_value_net.pt
```

### Step 5: Diagnostics — focus on value calibration

The critical diagnostic is D2 (avg prediction by game result). This is the metric that has persistently failed.

**D2 targets for fresh_start_v3:**

| Game result | Previous avg | Target avg | Direction needed |
|-------------|-------------|------------|-----------------|
| White win (+1) | +0.001 | > +0.15 | Must move positive |
| Draw (0/-0.5) | ~0.0 | -0.2 to +0.1 | Acceptable as-is |
| Black win (-1) | -0.089 | < -0.15 | Must move more negative |

These are deliberately modest targets. We're not expecting perfect calibration — just measurable directional improvement. If White-win positions average above +0.15 and Black-win positions average below -0.15, the value head is starting to learn which side is winning. That's enough for the MCTS to get some useful signal.

**Also run:**
- Sim ceiling test (should still pass — the heuristic value in hybrid games doesn't poison the value head because we train on game results)
- Policy top-1 match (should remain ~50% or improve)
- Opening evaluation (should remain in the -0.5 to 0.0 range)

### Step 6: Test pure NN self-play again

**Only if D2 shows improvement (White-win avg > +0.15, Black-win avg < -0.15).**

Run 20 games of pure NN self-play from the standard opening at 800 sims. Not a full batch — just a probe to see if the improved value calibration changes the self-play outcome.

If any Black wins appear: the value head is crossing the threshold. Run the full 50-game batch and apply the cautious self-play protocol.

If still 0% Black wins but average game length has increased (>12 moves): progress. Run another cycle of hybrid generation → retrain.

If no change at all: the value head needs more hybrid data. Generate another 150 games and retrain.

---

## The Iterative Loop

This directive describes one cycle. The expected path is multiple cycles:

**Cycle 1:** Hybrid games → retrain → test D2 → test NN self-play
**Cycle 2:** More hybrid games → retrain → test D2 → test NN self-play
**Cycle N:** D2 passes → NN self-play produces contested games → transition to pure NN self-play

Each cycle, the value head improves because it trains on game results from well-played games (hybrid MCTS). Each cycle, test whether the improvement is sufficient for pure NN self-play. When it is, the hybrid scaffolding is no longer needed.

**The heuristic evaluator is training wheels for the value head.** The policy head already graduated (50% top-1 match). The value head hasn't yet. Keep the training wheels on until it does.

---

## Priority Sequence

| Priority | Action | Effort |
|----------|--------|--------|
| 1 | Implement `--hybrid-eval` mode in mcts.py / data_generation.py | 1-2 hours |
| 2 | Calibration: 10 hybrid games from opening | 30 min |
| **CHECKPOINT: Report calibration. Stop if 0% Black wins.** | | |
| 3 | Production: 100 opening + 50 mid-game hybrid games | 1-2 hours |
| 4 | Add to clean corpus, reprocess, verify distribution | 30 min |
| 5 | Retrain (warm-start from v2) | 1-2 hours |
| 6 | Diagnostics — especially D2 value calibration | 30 min |
| **CHECKPOINT: Report D2 results. Stop if no improvement.** | | |
| 7 | Probe: 20 pure NN self-play games from opening | 30 min |
| **CHECKPOINT: Report probe results to Aaron.** | | |
| 8 | If probe shows progress, full 50-game batch + cautious self-play protocol | 1-2 hours |

---

## Success Criteria

**This directive succeeds when:**
- D2 shows measurable improvement (White-win avg > +0.15, Black-win avg < -0.15)
- Pure NN self-play from the opening produces ≥1 Black win in 20 games
- Average game length in NN self-play exceeds 15 moves

**The project reaches its next major milestone when:**
- Pure NN self-play produces ≥10% Black wins from the opening
- The cautious self-play loop (generate → inspect → validate → train) runs successfully
- The heuristic evaluator is no longer needed for data generation

At that point, the engine has a functional self-play loop for the first time, and improvement becomes a matter of scale rather than architecture.
