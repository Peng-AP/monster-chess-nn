# Checkpoint E: Self-Play Batch 001 — Phase 2 Report

*Generated 2026-03-07. Stop point — Aaron reviews before any action.*

---

## Quantitative Results

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Total games | 50 | 49 saved (1 skipped: 0 positions) | — |
| Average game length | ≥ 15 moves | **7.8 complete turns** (7.2 recorded positions) | ❌ FAIL |
| Black win rate | ≥ 10% | **0%** | ❌ FAIL |
| White win rate | — | **100%** | — |
| Games under 10 moves | ≤ 5 | **35 / 49 (71%)** | ❌ FAIL |
| Games ≥ 15 complete turns | — | **2 / 49 (4%)** | — |

*Note on recorded positions: `early_cutoff=5` (first 5 actions not recorded) and 30%
random subsampling mean recorded positions (avg 7.2) undercount actual game length.
FEN fullmove counter analysis gives actual avg of 7.8 complete turn cycles.*

**All 49 games: White win. Zero Black wins. Zero draws.**

---

## Verdict: Batch Rejected

All three quality floor criteria fail. Per the directive: do not add to training. Diagnose and report.

---

## Diagnosis

**Root cause: Policy head has no opening theory for Black.**

The training corpus was designed to teach the model that Black can win — but all of
Black's training positions came from:
1. **Curriculum games** — scripted Black wins from mid-game positions where Black already has overwhelming material advantage
2. **Human games** — Aaron's targeted play from mid-game positions
3. **Bruteforce** — carefully constructed positions where Black has a tactical advantage

None of these gave the model Black opening play from the **standard Monster Chess start**
(`rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3`). Black's first 15 moves from the opening
position were never in training — so the policy head has no idea what to do.

**White's double-move advantage from the opening is overwhelming.** With 800 sims,
White's MCTS can find an aggressive pawn/king advance that Black cannot refute because
Black's policy has never learned defensive structures from move 1.

**This is a policy failure, not a value failure.** The value head *does* recognize that
Black-winning positions are negative — it just can't guide the search toward them when
the policy head doesn't know the openings.

---

## Resolution

Addressed by DIRECTIVE_TIER10_OPENING.md: generate heuristic MCTS bruteforce data from
the standard opening FEN (Tier 10), retrain, then retry self-play.

**Calibration results (2026-03-07):**

| Sims | Black wins | Avg game length | Time/game |
|------|-----------|----------------|-----------|
| 1000 | 1/5 (20%) | 70 half-moves | 28.5s |
| 2000 | 1/5 (20%) + 1 draw | 79 half-moves | 75.9s |

Heuristic MCTS passes calibration. Production generation (100 games at 1000 sims)
in progress.

---

## Files

| File | Description |
|------|-------------|
| `data/raw/fresh_selfplay_batch_001/` | 49 game files, 351 positions — DO NOT add to training |
| `data/raw/fresh_selfplay_batch_001/generation_summary.json` | Full generation metadata |
| `DIRECTIVE_TIER10_OPENING.md` | Resolution directive |
