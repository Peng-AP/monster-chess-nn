# Directive: Opening Bruteforce — Tier 10

*From the directing instance, after reviewing Checkpoint E and self-play batch 001 discussion*

---

## Situation

Self-play batch 001 failed all quality criteria. 49 games, 100% White wins, average 7.8 turns. The root cause is precisely identified: **Black has zero training positions from the standard opening.** Every position in the clean corpus comes from mid-game or later. The policy head literally does not know what Black's first move should be.

The discussion report proposed five options. All five are rejected.

- Options 2, 3, 5 (start from mid-game FENs): dodge the problem. The model already plays adequately from mid-game positions. The gap is the opening.
- Options 1, 4 (more sims from the opening): throw compute at a structural gap. The policy head has never seen the opening. 2,000 sims with a blind policy head is still blind.

**The correct solution is the one that has worked every previous time: generate bruteforce data using heuristic MCTS.**

---

## The Fix

Use `generate_bruteforce.py` from the **standard opening FEN** at 1000-2000 sims. The heuristic MCTS uses `evaluation.py` with barrier detection, pawn advancement scoring, and king exposure — no neural network, no policy head. It uses uniform or heuristic-guided priors. It does not have the opening blind spot because it doesn't depend on learned move priors.

The calibration proved this works:

| Tier | Sims | Black wins | Time/game |
|------|------|------------|-----------|
| T7 (near-opening) | 1000 | 50.0% | ~35s |

The standard opening is one step beyond Tier 7. Expect somewhat lower Black win rates (30-50%) and somewhat longer games. Even 20% Black wins would be sufficient — the model just needs to see Black surviving the opening for the first time.

---

## Implementation

### Step 1: Calibration (30 minutes)

Before committing to production, calibrate from the exact opening FEN.

```
FEN: rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1
```

Run 5 games at each of 1000 and 2000 sims:

| Sims | Games | Record: time/game, result, game length |
|------|-------|----------------------------------------|
| 1000 | 5 | ? |
| 2000 | 5 | ? |

**What we need to see:**
- At least 1 Black win out of 10 games total (confirms the opening is contestable with heuristic MCTS)
- Average game length ≥ 15 moves (confirms games are not blowouts)
- Qualitative: does heuristic-guided Black make reasonable opening moves? Does it develop pieces, respond to pawn advances, attempt to set up barriers?

**If zero Black wins across all 10 calibration games:** The heuristic MCTS also cannot find Black's opening resources. This would be a significant finding — it would mean the opening is either genuinely White-won at this search depth or the heuristic evaluator doesn't capture the right opening concepts. Report back before proceeding.

**If ≥1 Black win:** Proceed to production generation at whichever sim count produced the win.

### Step 2: Production generation (1-2 hours)

Generate from the standard opening FEN:

```bash
py -3 src/generate_bruteforce.py \
  --num-games 100 \
  --simulations [CALIBRATION_WINNER] \
  --start-fen "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1" \
  --output-dir data/raw/fresh_bruteforce_t10_opening \
  --seed 20260307
```

If the command format doesn't support `--start-fen`, place the opening FEN in a file and use whatever mechanism `generate_bruteforce.py` uses for starting positions. The key requirement: every game starts from the exact standard Monster Chess opening.

**Target: 100 games.** Based on Tier 7 timing (~35s/game at 1000 sims), expect ~1 hour. At 2000 sims, expect ~2 hours. Parallelize across CPU cores.

**Expected output:**
- 100 games, 20-50% Black wins (based on Tier 7 extrapolation)
- Average game length 20-40 moves
- 3,000-6,000 positions of opening and early middlegame play from both sides

This is "Tier 10" — the opening itself. Every previous tier started further into the game. This completes the coverage.

### Step 3: Add to clean corpus

Place the generated data into the clean data pipeline:

```
data/clean/bruteforce_t10_opening/  ← copy JSONL files here
```

**Source weight:** 2 (same as other bruteforce data). Do not over-weight — the opening is one phase of the game, not the whole game. The model already plays mid-game and endgame adequately.

**MCTS_LAMBDA:** 0.0 (train on game result). Consistent with all other bruteforce sources.

### Step 4: Reprocess and retrain

Reprocess the expanded clean corpus:

```bash
py -3 src/data_processor.py --raw-dir data/clean --output-dir data/clean_processed [existing flags]
```

**Before training, verify the processed data distribution:**
- Tier 10 positions should be 5-15% of total training data (not dominant, just filling the gap)
- Overall Black-win percentage should remain 60-75% (Tier 10 will add some White-win positions, slightly rebalancing)
- Report the distribution before training

**Train: warm-start from the current fresh start model.**

```bash
py -3 src/train.py \
  --data-dir data/clean_processed \
  --model-dir models/fresh_start_v2 \
  --epochs 30 \
  --batch-size 256 \
  --lr 5e-4 \
  --warmup-epochs 3 \
  --warmup-start-factor 0.2 \
  --target game_result \
  --policy-loss-weight 1.0 \
  --grad-clip 1.0 \
  --resume-from models/fresh_start/best_value_net.pt
```

Warm-start is correct here — the fresh start model's value head and mid-game policy are good. We're adding opening knowledge on top, not replacing what exists.

Lower learning rate (5e-4 vs 1e-3) and warmup (3 epochs) because we're fine-tuning, not training from scratch. Early stopping on val loss as before.

### Step 5: Post-training diagnostics

Run the same diagnostic suite as Checkpoint D:

1. **Sim ceiling test:** Does the model still improve from 200 → 800 → 2000 sims? If this regresses, the new data hurt the model.
2. **Opening position evaluation:** Should shift from -0.498 toward something less extreme (the opening data will add White-win signal from the opening).
3. **Value distribution:** Should remain spread across [-1, +1].
4. **Policy spot-check from the opening FEN:** Does the policy head now suggest reasonable Black first moves? This is the key new diagnostic — check the top-3 policy moves for Black from the standard opening. Do they look like real chess moves (develop a knight, move a central pawn) or random?

**Save diagnostics to `models/fresh_start_v2/diagnostics.md`.**

### Step 6: Retry self-play from the opening

If diagnostics pass, regenerate self-play batch 002:

- 50 games from the standard opening
- 800 sims both sides
- Same protocol as batch 001

**Apply the same quality floor:**
- Average game length ≥ 15 moves
- Black win rate ≥ 10%
- Games under 10 moves ≤ 5 out of 50

If batch 002 passes: proceed with Phase 3 from the cautious self-play directive (add to training, train one iteration, check for regression).

If batch 002 still fails: report back. At that point we would know that even with opening training data, NN self-play from the opening doesn't work, and we'd need to fundamentally reconsider the approach.

---

## Priority Sequence

| Priority | Action | Effort |
|----------|--------|--------|
| 1 | Calibrate from opening FEN (10 games) | 30 min |
| **CHECKPOINT: Report calibration results. Stop if zero Black wins.** | | |
| 2 | Generate 100 opening games (Tier 10) | 1-2 hours |
| 3 | Add to clean corpus, reprocess | 15 min |
| 4 | Verify data distribution | 10 min |
| **CHECKPOINT: Report distribution. Confirm before training.** | | |
| 5 | Retrain (warm-start from fresh model) | 1-2 hours |
| 6 | Run diagnostics | 30 min |
| **CHECKPOINT: Report diagnostics to Aaron.** | | |
| 7 | Retry self-play batch 002 from opening | 1-2 hours |
| **CHECKPOINT: Report batch 002 quality metrics. Aaron reviews.** | | |

---

## Why This Should Work

Every previous tier of bruteforce data successfully taught the model the strategic concepts it needed:

- Tier 4-6 taught Black that it can win from advantaged positions → model learned
- Tier 8 taught White that overextension is punished → model learned
- Tier 9 taught White to advance the king → model learned (Aaron confirmed)

Tier 10 teaches both sides how the opening works. There is no reason to expect this will fail when every previous tier succeeded. The heuristic MCTS produces good data, the training pipeline absorbs it, and the model's behavior changes accordingly. We're just filling the last gap in coverage.

The model already plays coherent mid-game Monster chess. It already has a value head that improves with search depth. It just doesn't know what to do on move 1. A hundred games from the opening should fix that.
