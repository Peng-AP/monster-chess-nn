# Gen52 Arm L (2× wider network): results, completed October 1, 2026

Plan `docs/plans/GEN52_LARGE_PLAN.md` (criteria fixed before any result).
Driver `tools/gen52_large_campaign.py`. Production ran as one managed job,
September 30 11:15 → October 1 04:31, with no failures. Training took 5.95 h
(early stop at epoch 20; the nominee is epoch 8). Evidence is in
`benchmarks/gen52_program/gen52_large_20260930/production/`.
**Nothing was promoted.**

## Verdict: capacity helps (by the pre-declared criteria), narrowly

| Criterion | Arm L | Required | Met |
|---|---:|---|---|
| Score vs Arm B at 3,200 (400 games) | **55.25%**, 95% range 52.7–57.8% | lower end > 50% | yes |
| Distinct-game score vs Arm B | **50.5%** (100 distinct games) | > 50% | yes, barely |
| Held-out mean (B2, v27) | **92.5%** | ≥ 91.4% − 1.0 pp | yes |

**The gain is real but not across the board.**

- Against Arm B, all of it comes from Black: 110 W / 82 D / 8 L. As White,
  Arm L scored 1 W / 138 D / 61 L.
- With each distinct game counted once, it is a coin flip. The 55% rests on
  openings the sampler repeated, which is why the plan required both
  conditions.
- **Against v29, Arm L is level, not better** (see the gate and Elo below).

## The 4.0M network against the 1.9M one, everything else equal

| | Arm L (4.0M) | Arm B (1.9M) |
|---|---:|---:|
| Best validation loss | 1.5609 | 1.5695 |
| Value MSE at best | 0.1060 | 0.1062 |
| Gate v4 vs v29, 3,200 legs (800 g) | 45.9% (White −14.6 pp, Black +6.4 pp) | 47.4% (White −13.4, Black +8.1) |
| Gate v4 deep guard @12,800 (160 g) | **73.75%** (pass) | 51.25% |
| Gate verdict | FAIL (aggregate ≤ 50%, White floor) | FAIL |
| B2 (held out) | 90.9% | 87.8% |
| v27 (held out) | 94.1% | 95.0% |
| gen49 | 64.1% | 60.6% |
| v28 | 70.9% | 70.9% |

Head-to-head, Arm L against Arm B:

| Simulations | Games | Arm L | as White W/D/L | as Black W/D/L | Distinct games only |
|---:|---:|---:|---|---|---:|
| 3,200 | 400 | **55.25%** | 1/138/61 | 110/82/8 | 50.5% |
| 12,800 | 160 | 59.4% | 14/50/16 | 40/32/8 | 51.5% |

## Finding: Arm L's White does not collapse under deep search

Self-play, counted by the colour actually played (White W/D/L):

| | @3,200 | @12,800 |
|---|---|---|
| **Arm L** | 1 / 105 / 94 (26.8%) | **53 / 78 / 29 (57.5%)** |
| Arm B | 1 / 137 / 62 (34.7%) | 39 / 57 / 64 (42.2%) |
| Arm C | 2 / 107 / 91 (27.8%) | 8 / 51 / 101 (20.9%) |
| v29 (teacher) | 6 / 383 / 11 (49.4%) | 1 / 58 / 101 (18.8%) |

Against v29 at 12,800, Arm L's White scored 51 W / 12 D / 17 L (71.25%).

- Every 1.9M-parameter model loses White under deep search: v29 falls to
  18.8% and Arm C to 20.9%.
- The 4.0M network goes the other way, from 26.8% to 57.5%. Deep search
  helps its White.
- This is consistent with the bigger network judging White positions better,
  so deeper search finds real resources instead of amplifying a bias.
- It is not proof. Each cell is one block of 160–200 games.
- **The depth-scaling ladder (running now, seed-matched to v29's) is the
  direct test.** It measures whether Arm L keeps gaining Elo beyond 3,200
  where v29 flattens.

## Elo placement (3,200 simulations, joint fit with the round robin, v21 = 1600)

| Model | Elo | 95% interval |
|---|---:|---|
| **Arm L** | **2579** | 2522–2641 |
| v29 | 2574 | 2519–2632 |
| gen52 Arm C | 2563 | 2506–2628 |
| gen52 Arm B | 2550 | 2497–2614 |
| v28 | 2470 | 2420–2532 |

This fit includes only the round robin and this placement, so its scale
differs from the September 30 ladder's (where v29 is 2450). **At 3,200, Arm L
is level with v29 and the gen52 arms.**

## Conclusions

1. **Network size is no longer a measured null.** At today's data volume, a
   2× wider network beats its 1.9M twin on identical data and training, by
   the pre-declared rule. The margin over the distinct-game count is thin.
2. **It is not a stronger engine at the site's setting.** Arm L is level with
   v29 at 3,200 (Elo) and fails gate v4 there, with the same White weakness
   at 3,200 that every gen52 training shows.
3. **Its search scales differently.** It passes the deep guard at 73.75%, and
   its self-play White rises with depth where every smaller model's falls.
   If the depth ladder confirms it, Arm L is the gen53 teacher candidate:
   deep-search data distilled from a network whose deep search is worth
   copying.
4. Promotion: not eligible (gate v4 FAIL against v29). Nothing promoted.

## Depth scaling (October 1): the lead is not confirmed

Driver `tools/elo_depth_scaling.py`; evidence in
`benchmarks/elo_depth_scaling_20261001_armL/` (`joint_ratings.json`).

- Arm L played the 16 rated models at 800, 3,200, 6,400 and 12,800
  simulations, 320 games per depth.
- At 800, 6,400 and 12,800 its stage-1 games reuse **v29's ladder opening
  seeds**, so the comparison is game-for-game.
- A Windows restart at 04:49 interrupted the run. It resumed from its game
  journals at 10:18 with no games lost or repeated, and completed at 13:30.

**Joint fit:** round robin + v29 ladder + this run, 406 pairings, v21 = 1600,
1,000-rep bootstrap. Adding games compresses the scale again (v29 at 3,200 is
2408 here).

| Simulations | v29 | Arm L |
|---:|---:|---:|
| 800 | 2293 (2248–2339) | 2312 (2271–2357) |
| 3,200 | 2408 (2371–2447) | 2398 (2357–2444) |
| 6,400 | 2435 (2397–2476) | 2396 (2354–2444) |
| 12,800 | 2444 (2402–2490) | 2455 (2415–2502) |

Gain from 3,200 to 12,800: v29 +36, Arm L +57, with intervals of about ±45
on each point. **The two curves are statistically the same.**

Matched openings (stage 1: same seeds, the same 16 opponents, 128 games):

| Simulations | v29 | Arm L |
|---:|---:|---:|
| 800 | 74.6% | 77.7% |
| 6,400 | 82.8% | 81.6% |
| 12,800 | 87.5% | 84.0% |

Direct games against v29 at 3,200 (40 each):

| Player | Score |
|---|---:|
| Arm L at 12,800 | 71.3% |
| v29 at 12,800 | 55.0% |
| Arm L at 6,400 | 45.0% |
| Arm L at 3,200 | 46.3% |

**Reading:**

- Against the rated field, Arm L does **not** gain more from deep search than
  v29. Both are nearly flat beyond 3,200.
- Arm L's large deep-search wins are **specific to v29**:
  - 73.75% in gate v4's 12,800 guard;
  - 71.3% here at 12,800 against v29 at 3,200.

  They are not a general rise against the field, so the self-play finding
  above (White 57.5% at 12,800) looks like a match-up effect, not general
  depth scaling.
- **Teacher consequence:** the case for Arm L as gen53's teacher *because its
  search scales* is not supported.
- **Capacity verdict unchanged:** the 2× network beats its 1.9M twin (55.25%)
  but is level with v29.

## Where this leaves the strength programme

- **Search is not the lever** for either network size.
- **Capacity gives a small edge at equal data**, not a step change.
- **The remaining untested lever is the training target.** That is the
  teacher-pessimism hypothesis from GEN52_RESULTS / POOLCAP_RESULTS: value
  labels for White taken from a different evaluator, or game outcomes
  weighted up.
- Combining it with the wider network would test both together. Nothing is
  queued; this is the owner's decision.
