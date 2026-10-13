# Opening census and draw test: what the game is worth, October 12, 2026

This follows `GAME_VALUE_20261012.md` (owner: "take as much time as needed and
be very thorough").

- Drivers: `tools/opening_census.py` (phases 1, 2, 3 and the better-sampled
  rerun, phase 1b) and `tools/draw_conversion_test.py`.
- Evidence: `benchmarks/opening_census_20261012/<phase>/{report.json,summary.md,book.json}`
  and `benchmarks/game_value_20261012/draw_conversion/`.
- Players: gen55 (the strongest Black) and v29 (the release), in every colour
  combination, on the repetition-aware engine.
- Every selection rule was written into the driver before its results existed.

**Scores are White's**, with a draw counted as half. Material counts are not
used anywhere: Black starts 39 vs 4.

## 1. Every White first turn, at 3,200 (phase 1)

58 first-turn positions × 10 copies; 4 sampled half-moves after each position;
3,480 games.

| White vs Black | White W/D/L | White score |
|---|---|---:|
| gen55 vs gen55 | 60/76/1,024 | 8.4% |
| gen55 vs v29 | 46/28/506 | 10.3% |
| v29 vs gen55 | 59/14/507 | 11.4% |
| v29 vs v29 | 18/78/1,064 | 4.9% |

- **No first turn reaches 50% for White in all four pairings.**
- 7 first turns lost every game.
- **Caveat:** 4 sampled half-moves produced too little variety. Many cells
  rested on 1–7 distinct games, so per-position rankings from this phase are
  fragile. The totals across 58 positions are solid. Phase 1b fixes the
  per-position view.

## 2. The 14 best first turns, better sampled (phase 1b, 3,200)

20 copies, 8 sampled half-moves, 2,240 games. Distinct games: 330, 181, 200 and
357 per pairing.

| White vs Black | White W/D/L | White score |
|---|---|---:|
| gen55 vs gen55 | 72/99/389 | 21.7% |
| gen55 vs v29 | 30/27/223 | 15.5% |
| v29 vs gen55 | 23/18/239 | 11.4% |
| v29 vs v29 | 12/85/463 | 9.7% |

| First turn | Mean over pairings | Worst pairing | Phase 1 mean | 12,800 mean (phase 2) |
|---|---:|---:|---:|---:|
| e4 + d4 | **27.5%** | 0.0% | 31.2% | 33.3% |
| f4 + f5 | 25.6% | **20.0%** | 38.1% | 20.8% |
| d4 + f4 | 21.2% | 0.0% | 24.4% | 33.3% |
| c4 + e4 | 20.9% | 7.5% | 30.6% | 4.2% |
| c4 + d4 | 20.6% | 11.2% | 21.9% | 5.2% |
| Ke2 + e3 | 15.9% | 1.2% | 24.4% | 0.0% |
| c4 + Kf1 | 15.3% | 0.0% | 15.6% | 25.0% |
| e3 + f4 | 13.1% | 0.0% | 17.5% | 5.2% |
| Kd1 + f3 | 10.6% | 1.2% | 26.9% | 0.0% |
| e4 + e5 | 9.7% | 0.0% | 12.5% | 22.9% |
| c4 + f3 | 7.5% | 0.0% | 20.6% | 8.3% |
| Kf1 + e3 | 7.5% | 0.0% | 15.0% | 1.0% |
| c3 + Kf1 | 5.0% | 0.0% | 15.0% | 0.0% |
| d4 + d5 | 3.8% | 0.0% | 9.4% | 0.0% |

- **No first turn holds.**
- **e4 + d4**, the line the strong models chose on their own, is White's best
  on average.
- **f4 + f5** has the best worst case: White scores at least 20% in every
  pairing.
- e4 + d4 and d4 + f4 still produce few distinct games in the v29 pairings
  (3–10 per cell): those are narrow, forcing lines.

## 3. At 12,800 (phase 2)

14 positions × 6 copies, 672 games.

| White vs Black | White W/D/L | White score |
|---|---|---:|
| gen55 vs gen55 | 0/32/136 | 9.5% |
| gen55 vs v29 | 14/2/68 | 17.9% |
| v29 vs gen55 | 5/0/79 | 6.0% |
| v29 vs v29 | 0/41/127 | 12.2% |

- Deeper search makes White's result worse in 10 of the 14 first turns.
- 5 first turns lose every game at 12,800.
- No first turn holds.

## 4. Black's answers to White's best 6 first turns (phase 3, 3,200)

Every Black reply × 4 copies, gen55 vs v29 both ways, 944 games.

| First turn | White's score, averaged over all Black replies | Black's best reply | White's score after it |
|---|---:|---|---:|
| e4 + d4 | 82.2% | …d5 or …f5 | **19%** |
| d4 + f4 | 78.1% | …d5 | **0%** |
| e4 + e5 | 81.6% | …f5 or …f6 | **0%** |
| f4 + f5 | 81.6% | …d6 | **6%** |
| c4 + Kf1 | 50.3% | …c5 or …e6 | **0%** |
| c4 + f3 | 53.4% | …d5 or …Nh6 | **0%** |

- **Black's advantage depends on one or two precise replies.** Against most
  Black replies, White scores 50–100%.
- With the right reply, White is held to 0–19%.

## 5. Are strong-model draws winnable? (draw test)

- **Setup:** 100 strong-model repetition draws from gen55's run, restarted 20
  half-moves before the draw (Black to move). gen55 plays both sides,
  repetition-aware.
- **Results:**

  | Depth | Games | Draws | White wins | Black wins |
  |---:|---:|---:|---:|---:|
  | 3,200 | 200 | 164 (140 repetition, 24 turn limit) | 30 | 6 |
  | 12,800 | 200 | **192** (152 repetition, 40 turn limit) | 0 | 8 |

**Those drawn structures are real holds.** Black converts 3–4% of them, even
deeper and repetition-aware.

## 6. What the evidence says about the game

1. **Black is better, and the strongest Black we have refutes every White first
   turn tested.**
   - From all 58 first turns, White scores 5–22% against strong Blacks at
     3,200, and 6–18% at 12,800 on the best 14.
   - Each of White's best 6 first turns has a Black reply that holds White to
     0–19%.
2. **The edge is precise.** Black must find one or two specific replies; most
   others lose. A Black that misses them, as v29 does against some lines that
   gen55's White plays, gets punished.
3. **Drawn structures exist and hold.** Once reached, Black converts only 3–4%,
   even at 12,800. **The question is whether White can force its way there
   against the best Black. The census says it currently cannot.**
   - This corrects the October 12 notification, which said "game looks like a
     draw with best play".
   - The defensible statement: the practical value is a Black win against
     today's best Black. The perfect-play value is a Black win or a draw, and
     unproven.
4. **How close to perfect play:** not close.
   - Results for the same first turn swing by tens of points between 3,200 and
     12,800 (c4 + e4: 20.9% → 4.2%; e4 + e5: 9.7% → 22.9%).
   - The two models each refute lines the other plays.
   - The space, though, is now mapped:
     - 58 first turns;
     - for the 6 best, every Black reply with its result;
     - the key refutations named.

## 7. Each model's holes

| Line | gen55 | v29 |
|---|---|---|
| d4 + f4 | Its White loses it to v29's Black (phase 1: 0/0/10) | Its White draws it against gen55's Black (0/10/0) |
| e4 + d4 …c5 or …e6 | Its White beats v29's Black 4/4 | Its White loses both to gen55's Black, 4/4 |

Self-play never shows a model these holes: gen55's own Black does not refute
d4 + f4.

## 8. Recommendations

1. **A per-colour instrument.**
   - Two numbers per model: White's score against a strong reference Black
     over the 14 key first turns, and Black's score against a strong reference
     White over the same.
   - Sampling as phase 1b: 20 copies and 8 sampled half-moves gave 180–360
     distinct games per pairing.
   - Report both colours separately, instead of one aggregate from
     model-chosen openings.
   - **This revises the standing rule that normal-start openings are the
     primary instrument, so the owner decides.**
   - Cost: about 1–1.5 hours per model per reference at 3,200.
2. **White-focused training data.**
   - Start the teacher's White games from the census positions, weighted
     toward the lines that score best (e4 + d4, f4 + f5, d4 + f4, c4 + e4,
     c4 + d4), against the strongest Blacks.
   - Include the positions after Black's refuting replies, so White learns to
     defend the critical lines rather than only its preferred one.
3. **Black-side completeness.** Make sure the Black side knows every
   refutation: …d5 or …f5 against e4 + d4, …d5 against d4 + f4, …f5 against
   e4 + e5, …d6 against f4 + f5. v29's gaps show that a model can miss one.
4. **Next generation design (needs the owner's approval):**
   - census-book starts in self-play, in place of 30-ply exploration from the
     normal start;
   - the per-colour instrument for selection and gating;
   - the repetition-aware engine.
5. **For the engine as a product:**
   - **as White:** steer toward the lines with the best worst case (f4 + f5,
     e4 + d4) and into the known drawn structures;
   - **as Black:** play the refutation for each setup, and convert.
