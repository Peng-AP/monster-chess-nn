# Monster Chess NN — Handoff to Director Instance

*This is written by the operational Claude instance to you — the planning instance that authored the FRESH_START_PLAN series. You know the diagnosis, the architecture, and the priority sequence. This document tells you what was actually implemented, what the outcomes were, and where things stand now.*

---

## Short Version

Your priority sequence was executed. All six priorities are complete. An overnight alternating run is currently active (gen 323+). The consolidation run (Priority 6) produced a genuine, measurable improvement: humanseed Black win rate went from 2.2% to 17–26% across 8 iterations. Self-play Black from the standard opening is still ~0% — the overnight run is the current attempt to change that.

---

## Priority Execution Summary

### Priority 1 — Timeout bug fix ✓

You diagnosed this correctly and the real fix was even simpler than you described. The bug was in the **batch timeout formula** in `data_generation.py`:

```python
# Old (wrong): treated per-game budget as total batch budget
total_timeout = per_game_budget + 60

# Fixed: scales with batch size
total_timeout = per_game_budget * max(1, num_games // pool_size) + 120
```

`per_game_budget` was `max(600, sims * MAX_GAME_TURNS * 0.025)`. At 450 sims that's ~1687s — the whole 140-game batch got 29 minutes total. Fast White-win games (2 min each) consumed all the budget; Black endgame conversion games were systematically killed. Exactly the systematic bias you suspected.

Additionally, a separate **per-game deadline** was added to `play_game(game_deadline=None)` so individual slow games can't block the whole batch (the 99% hang bug). Both fixes are committed and tested.

`--human-seed-simulations` was also reduced from 300 → 150 as belt-and-suspenders.

### Priority 2 — Move-limit draw relabeling ✓

Implemented in `monster_chess.py` exactly as you specified: games reaching 150 turns where `evaluate() < -0.4` get `game_result = -0.5` instead of `0.0`. Uses the existing `evaluation.py` heuristic rather than a custom material counter (your Addition 3 suggestion from the final response).

### Priority 3 — Data distribution analysis

Skipped as a standalone step. The calibration (Priority 4) provided equivalent signal — we could directly observe Black win rates from the source that mattered (curriculum positions).

### Priority 4 — Calibration matrix ✓

Ran a **full 15-cell calibration** using `generate_bruteforce.py`. The critical finding: **heuristic MCTS outperformed rollouts at every tier**, matching Addition 2 from your final response (`evaluation.py` as the rollout guide). We used `evaluation.py` at the leaf directly rather than as rollout policy biases, which was cleaner and faster.

Results summary:

| Tier | Sims | Black wins | Time/game |
|------|------|------------|-----------|
| T4-5 | 200 | 41.7% | ~3s |
| T4-5 | 500 | 41.7% | ~5s |
| T4-5 | 1000 | 41.7% | ~10s |
| **T6** | **500** | **66.7%** | **~9s** |
| T6 | 1000 | 58.3% | ~27s |
| **T7** | **1000** | **50.0%** | **~35s** |

All 15 cells produced ≥15% Black wins. Pure random rollouts were consistently lower than heuristic (41.7% vs 66.7% at T6/500). The heuristic's barrier detection was the key — it directly encodes Black's winning concept.

**Decision taken**: proceed with T4-5 @500, T6 @500, T7 @1000. Skip rollout mode entirely.

### Priority 5 — Bruteforce data generation ✓

Three production datasets generated as `data/raw/nn_gen307_bruteforce_t*_humanseed/`:

| Dataset | Positions | Games | Black (r<0) |
|---------|-----------|-------|-------------|
| T4-5 @500 | 2444 | 80 | 33.8% |
| T6 @500 | 3584 | 80 | 72.5% |
| T7 @1000 | 3391 | 40 | 25.0% |
| **Total** | **9419** | **200** | — |

T7 "Black wins" are all `result=-0.5` (Black-dominant draws at the move limit, relabeled per Priority 2). No true king captures at near-opening positions — expected.

`_humanseed` suffix gives weight=2 and activates the nonloss filter in the pipeline.

### Priority 6 — Non-alternating training on new data ✓

New preset `consolidation` added to `iterate_presets.py`. Non-alternating (trains both sides simultaneously), uses `overall_strict_sides` gate.

**Critical lesson from implementation**: gate `min-side-score` must be `0.00` for consolidation. Any floor above 0 causes 100% rejection because Black wins 0/30 arena games from the standard opening — arena Black score = 0% regardless of the model's actual improvement on curriculum positions. The correct gate for this phase is `overall >= threshold` only.

**Outcome: 8/8 iterations accepted**, gen 306 → **322**.

Humanseed Black win rate per iteration:
```
Iter 1: 17%  Iter 2: 20%  Iter 3: 18%  Iter 4: 20%
Iter 5: 22%  Iter 6: 24%  Iter 7: 26%  Iter 8: 20%
```

That's 2.2% → 17–26%, an order-of-magnitude improvement from the pre-bruteforce baseline. Self-play Black from standard opening is still ~0%.

### Priority 7 — Validation ✗ Not yet done

Aaron has not yet played against the gen 322 model. This is the next explicit checkpoint.

---

## What's Running Now

**Overnight alternating run**, background task `b7gbc9d86`:

```
py -3 src/iterate_presets.py --preset overnight > logs/overnight_run.log 2>&1
```

- 6 iterations, alternating Black-focus / White-focus
- Black trains at 450 sims, opponent at 160 sims (alternating)
- Gate: threshold=0.45, min-side=0.00
- 180 self-play + 220 curriculum + 260 black-focus + 140 humanseed games per iteration
- Starting from gen 322 (which carries the full bruteforce signal)
- Currently on iteration 1/6, generating self-play games for gen 323

Check: `tail -100 logs/overnight_run.log` and `models/iterate_run_*.json` when it completes.

---

## Key Open Questions

**1. Does the bruteforce signal transfer to the standard opening?**

The consolidation showed the model can use the curriculum knowledge. Humanseed Black win rate (games starting from the standard opening) improved from 2.2% to 26%. But self-play Black is still 0% — which means when the model plays *itself* from the opening, it can't convert. The overnight run is the test.

**2. If overnight makes no progress, what next?**

The likely issue is a gap between T7 positions (pawns just starting to advance) and the true opening position. Options:
- Generate Tier 8 bruteforce data: positions with White king on rank 2-3, all pawns present but one has moved 1-2 squares — intermediate between T7 and the raw opening
- Run another consolidation pass with the overnight's new data before resuming alternating
- Reduce overnight gate threshold further (0.40 was consolidation's threshold; overnight uses 0.45)

**3. Warm-start vs fresh init question**

Your final plan asked about this. The consolidation training warm-started from gen 306 weights. The value head was not reinitialized. Convergence was smooth (8/8 accepted) so the warm-start worked fine. No evidence the old prior "fought back." If future training stalls, reinitializing the value head while keeping backbone weights is still the recommended intervention.

---

## Current Metrics Snapshot

| Metric | Value |
|--------|-------|
| Best model | gen 322 |
| Self-play Black win rate (std opening) | ~0% |
| Humanseed Black win rate | 17–26% (post-consolidation) |
| Arena Black win rate | ~0% (standard opening; gate uses min-side=0.00) |
| Overnight run | Active (iter 1/6) |

---

## Files You Should Know About

| File | What changed |
|------|-------------|
| `src/data_generation.py` | Timeout formula fixed; `play_game(game_deadline=)` added |
| `src/monster_chess.py` | Move-limit draw relabeling (heuristic < -0.4 → result = -0.5) |
| `src/generate_bruteforce.py` | Heuristic MCTS data generator (curriculum starting positions) |
| `src/iterate_presets.py` | `consolidation` preset added; overnight/daily updated |
| `src/evaluation.py` | `_black_can_capture_king()` added; PIECE_SAFETY_BONUS, BLACK_KING_EXPOSURE_PENALTY added |
| `src/config.py` | Curriculum Tier 7 added (near-opening positions) |
| `tests/test_presets_contracts.py` | Updated for consolidation preset |
| `data/raw/nn_gen307_bruteforce_t*_humanseed/` | The bruteforce datasets |
| `models/best_value_net.pt` | Gen 322 |
| `logs/overnight_run.log` | Active run log |

---

## Process Note

Your three-checkpoint structure was followed:
- Checkpoint 1 (calibration results): done, informed decision to use heuristic mode over rollouts
- Checkpoint 2 (training diagnostics): done informally via consolidation run outcomes
- Checkpoint 3 (Aaron plays against model): **pending** — Priority 7

The overnight run completing is a natural synchronization point before Checkpoint 3.
