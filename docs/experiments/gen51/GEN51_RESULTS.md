# Gen51 results — completed September 27, 2026

Campaign `tools/gen51_campaign.py` (plan `docs/plans/GEN51_STRENGTH_PLAN.md`
§7). Production 2026-09-25 18:20 → 2026-09-27 03:01 Eastern (32.7 h), one
managed job, no failures or restarts. Evidence:
`benchmarks/gen51_program/gen51_20260925/production/summary.json` and its
stage receipts, journals and reports. Every journal was legally replay-audited.
**Nothing was promoted.**

## Bottom line

1. **Both gen51 arms passed gate v4 against v28 by wide margins**, in both
   colours and at both depths. The first gate v4 passes.
2. **Much of the margin over v28 is real:** against the fully held-out B2,
   gen51 scores 98.75–99.1% where v28 scored 73.75%.
3. **The control arm has one serious White hole:** after e4+d4 …d5 it plays
   c4 + Ke2, which beats v28's defence but loses to gen49's (50 of 61 games).
   Against gen49 it scores 47.2% overall, where v28 scored 83.1%.
4. **The deep-value arm fixes that hole.** Same line, but 9 W / 57 D / 0 L
   against gen49 (75.6% overall), a balanced self-play at 3,200, and it beats
   the control arm 67.8% at 12,800 (50.1% at 3,200).
5. Self-play diversity rose 42% (30-ply exploration, owner-approved), which
   reversed the gen47–gen50 narrowing.

Teacher and exploration changed together, so the gen51-vs-v28 gain cannot be
split between them. Only the arm comparison is causal: it isolates the
deep-value source.

## Recipe (shared by both arms)

- Teacher v28 (gen50 epoch14 + calibrated value head).
- 2,800 self-play games at 1,600 simulations with **temperature 1.0 for 30
  primitive plies** (was 15), plus 400 parent-linked forks at 12,800.
- Reanalysis: 24,000 positions at 12,800, 12,000 kept (60% Black).
- 8-generation replay (gen45–gen51), scratch training, seed 3173.
- All 3,200 games saved, none failed. Reanalysis took 176 minutes.
- Search constants unchanged (Stage 1 null: `SEARCH_CONSTANTS_RESULTS.md`).

**Deep-value arm only:**

- 768 disagreement roots (384 Black / 192 / 192, plies 4–118), each played
  out twice at 6,400: 1,536 games and 82,302 positions.
  - Outcomes (White perspective): 928 Black wins, 246 White wins, 362 draws.
  - The two playouts of a root agreed on 663 of 768 roots (86%).
- Strict capture labels, value weight 4, policy weight 0, parent-linked
  splits (0 mismatches in rehearsal): 164,604 rows added to the 4,300,952-row
  replay.
- Training early-stopped at epoch 20 (control: 27).

**Diversity** (`DIVERSITY_AUDIT.md`): distinct positions 121,075 → 171,464;
value rows per position 3.77 → 2.77; middlegame top-100 share 23.6% → 3.1%.

## Selection (matched seeds; 3,200 screen, then 12,800 probes vs v28)

| Arm | Epoch | Screen @3,200 (200 g) | Probe @12,800 (80 g) | Nominee |
|---|---|---:|---:|---|
| control | 16 | 82.0% | 93.75% | **yes** |
| control | 17 | 79.3% | 92.5% | |
| control | 18 | 78.0% | 83.1% | |
| deep-value | 9 | 77.75% | 80.6% | **yes** |
| deep-value | 13 | 77.0% | 83.75% | |
| deep-value | 10 | 67.75% | 83.1% | |

All six probes passed the guard rule.

- Control nominee: `models/candidates/bootstrap_main_gen_0051/arena_selected.pt`
  (SHA256 `c1f91215…6db69`).
- Deep-value nominee:
  `models/candidates/bootstrap_main_gen_0051_deepvalue/arena_selected.pt`
  (SHA256 `dbf26b9e…e84e2`).

## Gate v4 against v28 (v28 par reused from Stage 1)

| Leg | Control | Deep-value |
|---|---:|---:|
| vs_bar @3,200 (400) | 79.37% (W 71.0 / B 87.7) | 74.12% (W 67.5 / B 80.7) |
| vs_bar_confirm @3,200 (400) | 79.25% (W 70.5 / B 88.0) | 75.75% (W 69.8 / B 81.8) |
| deep_guard @12,800 (160) | 95.63% (W 97.5 / B 93.8) | 85.00% (W 96.9 / B 73.1) |
| Endpoint-unique, both 3,200 legs | 74.4% / 77.3% | 69.6% / 75.8% |
| **Verdict** | **PASS** | **PASS** |

Combined 800-game colour deltas against v28's self-par (the Stage 1 par:
White 31.75% / Black 68.25% at 3,200 over 400 games):

- Control: White +39.0 pp (95% range +34 to +44), Black +19.6 pp (+15 to +24).
- Deep-value: White +36.9 pp (+32 to +42), Black +13.0 pp (+9 to +17).

## Diagnostics (3,200 simulations, 160 games unless noted)

| Opponent | Control | Deep-value | v28 (Sept 17) |
|---|---:|---:|---:|
| gen49 | **47.2%** (W 18.8, B 75.6) | **75.6%** (W 55.0, B 96.3) | 83.1% (W 70.0, B 96.25) |
| v27 | 86.25% (W 72.5, B 100) | 86.25% (W 72.5, B 100) | 90.9% (W 81.9, B 100) |
| B2 (held out) | 98.75% (W 98.1, B 99.4) | 99.06% (W 98.1, B 100) | 73.75% (W 83.75, B 63.75) |

The line that decides it (gen51 as White after e4+d4):

| Reply | Follow-up | Control | Deep-value |
|---|---|---|---|
| …e5 (B2 78/80, v27 29/80) | f4, fxe5 | 106 W / 1 D / 0 L | similar (B2 78/1/1) |
| …d5, v28's defence | c4 + Ke2 | 154 W / 150 D / 13 L (gate) | wins and draws |
| …d5, v27's defence | c4 + Ke2 | 3 W / 33 D / 1 L | mostly draws |
| …d5, gen49's defence | c4 + Ke2 | **9 W / 2 D / 50 L** | **9 W / 57 D / 0 L** |

Actual-colour self-play (White W/D/L; the same c4 + Ke2 line in nearly
every game):

| | @3,200 (200 g) | @12,800 (160 g) |
|---|---|---|
| Control | 3 / 9 / 188 (White 3.7%) | 0 / 138 / 22 (White 43.1%) |
| Deep-value | 2 / 189 / 9 (White 48.2%) | 0 / 70 / 90 (White 21.9%) |
| v28 (reference) | White 31.75% | 14 / 118 / 28 (White 45.6%) |

## Arm vs arm (deep-value as A, both passed)

| Budget | Games | Deep-value score | White W/D/L | Black W/D/L |
|---|---:|---:|---|---|
| 3,200 | 400 | 50.1% | 4 / 169 / 27 | 26 / 172 / 2 |
| 12,800 | 160 | **67.8%** | 7 / 53 / 20 | 70 / 10 / 0 |

## Interpretation

- **The generation change produced a large, mostly general gain.** Both arms
  crush B2, which neither ever trained against. Screens, probes, gates and
  diagnostics agree across six checkpoints. v28's weakest opponent (B2 as
  White) became the gen51 arms' strongest.
- **Head-to-head against the teacher overstates it.** Against gen49 and v27,
  the control arm is not better than v28, and is much worse against gen49,
  because of one learned White line. That line is the same e4+d4 …d5 c4
  structure that troubled gen50. Gate v4 plays only the incumbent and cannot
  see this; the older-opponent diagnostics did.
- **Deep disagreement-outcome value targets are worth keeping.** At equal
  data they made the model hold the problem line instead of losing it, won
  the deeper arm-vs-arm clearly, and cost only a little score against v28 at
  3,200. This replicates, in full training, the direction of the September 17
  head-only calibration.
- Self-play colour splits flip between depths in both arms. The c4 + Ke2
  structure is balanced on a knife edge for these networks; this describes the
  models' two sides, not the game's value.

## Decisions taken from this

- **gen52 teacher = the deep-value nominee**, by the rule declared before the
  arm-vs-arm result (`docs/plans/gen52_teacher_decision.json`): higher mean
  outside score (87.0% vs 77.4%), and the control did not reach 55% in the
  arm-vs-arm match. The deep-value source stays in the recipe. The teacher is
  used unpromoted.
- **Owner's model pool for gen52** (`GEN52_PLAN.md`): 1,200 games against
  v28, gen49, gen48 and v26 as a second arm. **B2 and v27 stay held out.**
- **Promotion is not decided.** Both nominees are playtest candidates; the
  deep-value one is the recommended first playtest. Suggested test: play Black,
  answer e4+d4 with …d5, and see how it handles c4 + Ke2.
- Proposed, not adopted: require held-out-opponent non-regression against the
  previous release as part of promotion evidence (gate v5). Owner decision.
