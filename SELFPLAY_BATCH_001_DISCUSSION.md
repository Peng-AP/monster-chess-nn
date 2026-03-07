# Self-Play Batch 001 — Discussion Report

*Generated 2026-03-07. For Aaron + directing instance discussion.*

---

## What Happened

50 self-play games were generated from the standard Monster Chess opening position
(`rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3`) using the fresh start model at 800 sims per
side, temperature 1.0 for the first 15 moves then 0.1.

**Results:**

| Metric | Target | Actual |
|--------|--------|--------|
| Black win rate | ≥ 10% | **0%** |
| Average game length | ≥ 15 turns | **7.8 complete turns** |
| Games ≥ 15 turns | most | **2 / 49 (4%)** |
| Games ≥ 10 turns | ≥ 45 | **14 / 49 (29%)** |

*Note on game length: recorded positions per game averaged 7.2, but this undercounts
actual game length because `early_cutoff=5` (first 5 actions never recorded) and a 30%
random position subsample. FEN fullmove counter analysis gives the true figure of 7.8
complete turn cycles.*

All 49 completed games: White win. Zero Black wins. Zero draws.

---

## Root Cause

**Black has never seen the opening.**

The clean corpus (76,942 positions, weighted) was built from:

| Source | Positions | How Black positions arose |
|--------|-----------|--------------------------|
| Bruteforce tiers 4-7 | ~17,719 | Mid-game starts where Black already has material advantage |
| Bruteforce tier 8 | ~4,684 | Mid-game starts where White has overextended pawns |
| Bruteforce tier 9 | ~3,616 | Mid-game starts where White king is lagging |
| Human games (Aaron) | ~1,055 | Mid-game play, Aaron already past the opening |
| Curriculum gen 347 | ~9,921 | Scripted Black endgame play, not from move 1 |

**Zero positions in training start from the standard opening FEN with Black to move.**

The policy head has no learned prior for Black's first 10-15 moves. At 800 sims, MCTS
with a bad policy prior still searches — but it searches badly. White's double-move
advantage allows White's pawns to reach rank 5-6 within 3-4 turns. Black's policy,
seeing an opening position it has never encountered in training, cannot find defensive
structures in time.

This is **not a value head problem**. Checkpoint D confirmed the value head correctly
identifies extreme positions (White 4 pawns on rank 6 → +0.9998; Black rooks+queen vs
lone king → -0.9128). The sim ceiling test passes (model improves with more sims at the
opening). The model *can* evaluate, but the policy head can't guide search toward good
Black moves because it has never seen the positions.

---

## The Bootstrapping Problem

This is a genuine chicken-and-egg problem:

- Self-play needs a functional Black policy to generate useful games
- A functional Black policy requires training on games where Black survives the opening
- Black doesn't survive the opening without a functional Black policy

The previous training pipeline fell into this trap over 347 generations of all-White-win
self-play. We recognized it and curated mid-game positions specifically to give Black a
signal. That worked for mid-game play — but it left the opening untouched.

**The model now plays better Monster Chess from mid-game positions than any previous
checkpoint.** Aaron confirmed this in the Checkpoint D play-test. The problem is that
self-play starts from move 1, not from move 15.

---

## Options Discussed

### Option 1: Asymmetric sims (Black 2000, White 200)
Give Black 10× more search. Likely some Black survivals. Game quality asymmetric.

### Option 2: Human-game FEN starts
Start from mid-game positions the model has seen. More contested games but no opening
theory learned.

### Option 3: Combined — human FENs + asymmetric sims
Best chance of Black wins in batch 002, still no opening theory.

### Option 4: Higher sims for both (1600 each)
Modest improvement. Policy failure is structural, not computational.

### Option 5: Curriculum FEN starts (tiers 4-6)
Similar to Option 2 with bruteforce positions instead of human games.

**All five options rejected by directing instance** in favor of Tier 10 bruteforce.

---

## Resolution: DIRECTIVE_TIER10_OPENING.md

The correct solution: generate bruteforce data using heuristic MCTS from the **standard
opening FEN**. Heuristic MCTS uses `evaluation.py` directly — barrier detection, pawn
advancement scoring, king exposure. No neural network, no policy head blind spot.

Calibration confirmed viability:

| Sims | Black wins | Avg game length | Time/game |
|------|-----------|----------------|-----------|
| 1000 | 1/5 (20%) | 70 half-moves | 28.5s |
| 2000 | 1/5 (20%) + 1 draw | 79 half-moves | 75.9s |

Production run underway: 100 games at 1000 sims → `data/raw/fresh_bruteforce_t10_opening/`.

---

## Current State of the Model (for reference)

- **Value head**: Spread across [-1, +1] with std=0.62. Gets better with more sims.
  Black-biased prior (opening at -0.498 vs ~0.0 target) — correctable by self-play.
- **Policy head**: 26% top-1 match vs MCTS. Has mid-game knowledge. No opening theory.
- **White play**: King advancement confirmed by Aaron. Real behavioral improvement.
- **Black play**: Functional from mid-game positions. No opening theory.

---

## Files

| File | Status |
|------|--------|
| `data/raw/fresh_selfplay_batch_001/` | 49 games, 351 positions — do not add to training |
| `models/fresh_start/best_value_net.pt` | Current best model |
| `CHECKPOINT_E_SELFPLAY_BATCH_001.md` | Formal checkpoint report |
| `DIRECTIVE_TIER10_OPENING.md` | Resolution directive |
| `DIRECTIVE_CAUTIOUS_SELFPLAY.md` | Self-play protocol and guardrails |
