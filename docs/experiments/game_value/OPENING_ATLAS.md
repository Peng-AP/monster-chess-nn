# Opening atlas: the opening tree mapped by games and search, October 19–21, 2026

Owner, October 19: "Run an extremely comprehensive and deep opening
categorization and search, to map out basically everything."

- **Driver:** `tools/opening_atlas.py`. Every phase rule was declared before
  any result.
- **Report:** `tools/opening_atlas_report.py`.
- **Evidence:** `benchmarks/opening_atlas_20261019/`, holding each phase's
  `report.json`, `selection.json` and `atlas_summary.json`. The game and
  search journals stay local.
- **Run:** 44.5 h, October 19 13:30 → October 21 10:00. No failures.

**What was mapped:**

- **The tree:**
  - 58 positions after White's first turn (W1);
  - **1,156** after Black's reply (B1);
  - **70,537** distinct positions after White's second turn (W2), with
    transpositions merged.
- **Players:**
  - White: gen56 and gen55, the best per-colour Whites;
  - Black: gen56 and gen53, the best Blacks.
- **What a value means:**
  - It is White's score, with draws counted as half, averaged over the four
    White × Black pairings.
  - All games run at 3,200 sims unless stated, on the repetition-aware engine.
- **Games:** 20,776 in total: 20,200 at 3,200 sims in four phases:

  | Phase | Positions | Games |
  |---|---|---:|
  | G0 | all W1 | 1,856 |
  | G1 | all B1 | 9,248 |
  | G2 | 251 refinements | 6,024 |
  | G3 | 384 W2 positions on the critical lines | 3,072 |

  and 576 at 12,800 sims in the depth check (D).
- **Search** (12,800 sims, no noise):
  - all B1 positions with four models (S1);
  - all 70,537 W2 positions with gen56 and gen53 (S2a and S2b, 3,200 sims).

**Judge by outcomes.** Search values are a map. Section 4 measures how far
they can be trusted.

## Headline

1. **Every White first turn loses to Black's best reply.**
   - 42 of 58 first turns score **0%** after it.
   - The best is e4–e5, at **25%** against …Nf6.
   - In the 12,800 depth check, no first turn reached 50% against both of
     Black's best replies, even with White's best second turn.
   - The best first turns at depth are c4 + d4 (≈37.5%) and c4 + f3 (≈31%).
2. **Black's refutations are always a centre-pawn move or a knight move.**
   - A flank-pawn reply was never Black's best.
   - After a flank-pawn reply, White scores 77–85%.
3. **White's best first turns push one pawn twice** (e4–e5, f4–f5).
   - King walks are the worst: 0% for all four.
4. **The engines' values are miscalibrated in the opening.**
   - A search value of 0.0 ("equal") corresponds to White scoring about 25%.
   - One turn deeper, the search values are 0.57–0.60 too favourable to
     White.
5. **The models choose different moves at critical points.**
   - Two models pick the same best turn in only 35–47% of positions.
   - At several critical nodes, all four engines pick a second turn that
     scored worse in games than an alternative they ranked lower.

## 1. White's first turns (W1)

**Columns:**

- **Minimax:** White's score after Black's best reply. G1 values are used,
  pooled with G2 for the replies G2 refined (about 20 target games each).
- **≤10%, ≤20%:** how many of Black's 19–20 replies hold White to at most
  that score.
- **G0:** the direct value, with the models choosing Black's reply.
- **Deep:** the 12,800 check for the first turns in G3. Black chooses the
  worse of its two best replies; White chooses the better of its two best
  second turns.
- **Engine:** the S1 minimax (mean of four models).

| First turn | Family | Minimax | Best reply | ≤10% | ≤20% | Mean over replies | G0 | Deep | Engine |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| e2e4 e4e5 | one pawn twice | **25.0%** | …Nf6 | 0 | 0 | 81.0% | 37.5% | 25.0% | −0.12 |
| f2f4 f4f5 | one pawn twice | **22.5%** | …Nf6 | 0 | 0 | 75.0% | 28.1% | 12.5% | −0.07 |
| c2c4 f2f3 | two pawns | 17.5% | …c6 | 0 | 2 | 48.1% | 0.0% | 31.2% | −0.19 |
| c2c4 d2d4 | two pawns | 12.5% | …d5 | 0 | 3 | 71.4% | 28.1% | **37.5%** | −0.12 |
| c2c4 c4c5 | one pawn twice | 5.0% | …Nc6 | 3 | 3 | 68.9% | 6.2% | 12.5% | −0.15 |
| c2c4 e1d1 | pawn + king | 5.0% | …d6 | 1 | 5 | 51.2% | 9.4% | 0.0% | −0.17 |
| c2c4 e2e4 | two pawns | 5.0% | …Nc6 | 2 | 4 | 69.3% | 6.2% | 25.0% | −0.12 |
| d2d3 e1d2 | pawn + king | 5.0% | …d6 | 1 | 6 | 47.0% | 3.1% | 0.0% | −0.14 |
| d2d3 f2f4 | two pawns | 5.0% | …e5 | 1 | 6 | 45.3% | 21.9% | 12.5% | −0.18 |
| d2d4 d4d5 | one pawn twice | 5.0% | …c5 | 1 | 2 | 82.7% | 9.4% | 12.5% | −0.11 |
| d2d4 f2f4 | two pawns | 5.0% | …e6 | 2 | 3 | 68.9% | 25.0% | 25.0% | −0.08 |
| e1f2 f2f3 | pawn + king | 5.0% | …e6 | 2 | 6 | 51.6% | 0.0% | — | −0.19 |
| e1f2 f2f4 | pawn + king | 5.0% | …e6 | 1 | 2 | 56.4% | 12.5% | — | −0.15 |
| d2d4 e1d2 | pawn + king | 2.5% | …e6 | 3 | 3 | 63.9% | 6.2% | — | −0.14 |
| e2e3 f2f4 | two pawns | 2.5% | …d5 | 3 | 4 | 48.8% | 0.0% | — | −0.19 |
| e2e4 f2f4 | two pawns | 2.5% | …e6 | 2 | 3 | 72.8% | 0.0% | — | −0.14 |
| d2d4 e2e4 | two pawns | 0.0% | …f5 | 1 | 2 | 74.1% | 28.1% | 25.0% | **−0.01** |
| the other 41 | | 0.0% | | 1–9 | 2–12 | 30.6–64.0% | 0–15.6% | — | −0.12 to −0.24 |

The full table for all 58 is in `atlas_summary.json` under `w1`.

- **The models' favourite, e4 + d4, is refuted.**
  - …f5 holds White to 0% at 3,200 and …d5 to 17.5%.
  - The engines rate e4 + d4 as the best first turn (−0.01), which is the
    largest gap in the table between engine opinion and the game result.
- **G0 and minimax rank the first turns differently.**
  - c4 + f3 scores 0% in G0 because the models, choosing Black's reply
    themselves, pick a refutation.
  - Its minimax value is 17.5% because …c6, its best-scoring reply, is not
    the reply they pick.

## 2. Categories

**White's first turn, by family:**

| Family | First turns | Mean minimax | At 0% | Best | Mean G0 |
|---|---:|---:|---:|---|---:|
| One pawn twice (pawn on the 4th or 5th rank) | 8 | **7.2%** | 4 | e4–e5, 25.0% | 12.3% |
| Two pawns | 24 | 2.1% | 17 | c4 + f3, 17.5% | 6.9% |
| Pawn + king | 22 | 1.0% | 17 | 5.0% | 3.6% |
| King walk | 4 | 0.0% | 4 | — | 0.8% |

**Black's first reply, by family** (all 1,156 positions):

| Family | Positions | White's mean | Share holding White ≤10% | Times Black's best reply |
|---|---:|---:|---:|---:|
| Centre pawn, one step | 232 | 26.9% | 31.9% | 23 |
| Centre pawn, two steps | 228 | 29.3% | 26.8% | 12 |
| Knight | 232 | 27.0% | 28.4% | 23 |
| Flank pawn, one step | 232 | 77.5% | 0.0% | 0 |
| Flank pawn, two steps | 232 | 84.8% | 0.4% | 0 |

**White's second turn, by family** (on the critical lines, G3, 384
positions):

| Family | Positions | White's mean | Best |
|---|---:|---:|---:|
| One pawn twice | 54 | **18.5%** | 75% |
| Two pawns | 90 | 14.2% | 75% |
| Pawn + king | 172 | 12.8% | 100% (a weak Black reply, …a5) |
| King walk | 68 | 11.1% | 62.5% |

**Engine-preferred second turns across all 1,156 B1 positions** (S2,
gen56 and gen53): two pawns 332, one pawn twice 292, king walk 269,
pawn + king 263.

- The engines often prefer king walks, which never ranked well in games at
  the first turn.
- That is consistent with the calibration gap in section 4.

## 3. The critical lines and the depth check

G3 takes the 10 first turns with the highest minimax, plus the most-played
e4 + d4 and d4 + f4. For each, Black's two best replies are the critical lines, 24 in all.
On each line, White's 12 best second turns by engine search plus 4 random
second turns were played.

- **The engine ranking is informative:** its top 12 averaged 16.4%, the
  random controls 5.2%.
- **D replayed each line, and its two best G3 second turns, at 12,800.**
  These were fresh games, about 5 target games per position, so ±20 pp.

| Positions | Mean at 3,200 | Mean at 12,800 | ≥30% at 12,800 |
|---|---:|---:|---:|
| The 24 critical lines (Black's best replies) | 13.2% | 17.7% | 5 of 24 |
| Their best two second turns each (48) | 44.7% | **22.1%** | 14 of 48 |

- **The best second turns halved in fresh games.**
  - Each was the top of 16 candidates on 5 games, so most of the 3,200 value
    was selection.
  - Some may be depth; the design cannot separate the two.
- The lines themselves moved up slightly, since they had been chosen for
  being low.

**Second turns that held at both depths:**

| Line | White's second turn | 3,200 | 12,800 (W/D/L) |
|---|---|---:|---:|
| c4 + f3 …c6 | Kf2, Kg3 | 62.5% | 62.5% (2/2/1) |
| e4–e5 …Nf6 | f4 + d4 | 50.0% | 50.0% (2/0/3) |
| c4 + d4 …d6 | Kd2 + e4 | 25.0% | 50.0% (0/5/0) |
| c4–c5 …d6 | d4–d5 | 62.5% | 50.0% (0/5/0) |
| c4 + d4 …d5 | f4 + Kf2 | 50.0% | 37.5% (0/4/1) |
| f4–f5 …Nf6 | d3 + c4 | 75.0% | 37.5% (1/1/3) |

**Each first turn at 12,800**, taking the worse of Black's two best replies
and the better of White's two best second turns:

| First turn | Value |
|---|---:|
| c4 + d4 | 37.5% |
| c4 + f3 | 31.2% |
| e4–e5, c4 + e4, e4 + d4, d4 + f4 | 25% |
| f4–f5, c4–c5, d4–d5, d3 + f4 | 12.5% |
| c4 + Kd1, d3 + Kd2 | 0% |

## 4. Can the search values be trusted?

Search value (White's view) against the game outcome (2 × score − 1):

| Comparison | Model | Mean search | Mean outcome | Bias (search − outcome) | Correlation |
|---|---|---:|---:|---:|---:|
| S1 vs G1 (1,156 B1 positions, 12,800) | gen56 | +0.153 | −0.039 | +0.192 | 0.858 |
| | gen55 | +0.172 | | +0.212 | 0.858 |
| | gen53 | +0.178 | | +0.217 | 0.861 |
| | v29 | +0.186 | | +0.225 | 0.858 |
| S2a vs G3 (384 W2 positions, 3,200) | gen56 | −0.155 | −0.727 | **+0.573** | 0.544 |
| | gen53 | −0.124 | | **+0.603** | 0.511 |

**By bucket** (S1, gen56):

| Search value | Positions | Outcome | White's score |
|---:|---:|---:|---:|
| −0.25 | 215 | −0.78 | 11% |
| 0.0 | 420 | −0.49 | 26% |
| +0.25 | 213 | +0.35 | 67% |
| +0.50 | 240 | +0.79 | 89% |

- **Search ranks positions well and scales them badly.**
- What the engines call equal is a clear Black win, and the error grows on
  the critical lines one turn deeper.
- All four models show the same bias, so it is not specific to gen56.
- The value audit's White pessimism measured game positions overall. In these
  opening positions the search is White-optimistic.

## 5. Where the models disagree

- **Agreement overall:** at the 1,156 B1 positions, the four models' values
  correlate 0.96–0.98. Any two pick the same best White turn in only
  35–47%.
- **On the critical lines** (S1 best turn per model; G3 game value of that
  turn where it was played):

  | Line | gen56 | gen55 | gen53 | v29 | Best in games (3,200 → 12,800) |
  |---|---|---|---|---|---|
  | e4–e5 …Nf6 | d4–d5 (25%) | d4–d5 | d4–d5 | d4–d5 | f4 + d4 (50% → 50%) |
  | e4 + d4 …d5 | **e5 + Ke2 (50%)** | Ke2 + e5 (50%) | Ke2 + e3 (0%) | c4 + Ke2 (0%) | Ke2 + e5 (50% → **25%**) |
  | e4 + d4 …f5 | d5 + exf5 (0%) | same | same | same | Ke2 + e5 (50% → 25%); d5 + c4 (25% → 37.5%) |
  | c4 + d4 …d5 | c5 + f3 (6%) | f4 + Kf2 (50%) | f4 + Kf2 | f4 + Kf2 | f4 + Kf2 (50% → 37.5%) |
  | c4–c5 …Nc6 | e4 + Ke2 (31%) | f4–f5 (0%) | f4–f5 (0%) | f4–f5 (0%) | e4 + Ke2 (31% → 0%) |
  | c4 + f3 …c6 | f4 + d4 (0%) | f4 + Kf2 (0%) | e4 + Ke2 (25%) | Kf2 + f4 (0%) | Kf2, Kg3 (62.5% → 62.5%) |

- **gen56's deep-White line.**
  - In e4 + d4 …d5, gen56 and gen55 choose Ke2 + e5. It is the best second
    turn found there (50% at 3,200), but it falls to 25% (0/2/3) at 12,800.
  - This matches gen56's deep losses there (GEN56_RESULTS.md).
  - The atlas found nothing better for White at that node, so …d5 holds
    White to about 25% at depth.
- **Every model misses the e4–e5 …Nf6 line.** All four pick d4–d5 (25%) over
  f4 + d4 (50% at both depths).

## 6. Conclusions

**For the game's value:**

- Against today's best Blacks, White has no first turn that holds:
  - 42 of 58 lose outright to the best reply;
  - the best handful reach 25–37% with best play two turns deep.
- Black's practical win is now mapped at full width to Black's first reply.
- White's chances live in about six narrow lines. All begin with a pawn pushed
  twice or a c-pawn first turn (c4 + d4, c4 + f3), and all have a single
  centre-pawn or knight refutation.
- Perfect-play value is still unproven: these are the models' best play, not
  a proof.

**For training:**

- **(a) Value calibration in the opening is the largest measured error**
  (+0.19 to +0.60).
  - The atlas games are game-outcome labels for 1,598 real opening positions.
    They are not authored positions; they come from the full tree.
  - Using them as an opening-value correction is an option for the owner.
- **(b) Opening repertoire matters more than the generation-to-generation
  model.**
  - Book starts aimed at the atlas lines would teach White its best tries and
    Black its refutations:
    - e4–e5 …Nf6 f4 + d4;
    - c4 + f3 …c6 Kf2–g3;
    - c4 + d4 lines;
    - e4 + d4 …d5 and …f5.
- **(c) Depth matters on exactly these lines.** Training-game sims (1,600)
  may be too shallow for them, as in gen56.

## Next: the deep dive (done October 23)

See `OPENING_DEEPDIVE.md`. The six first turns were taken two turns deep,
including Black's answer to White's best second turn, with strict minimax on
fresh games.

| First turn | Value |
|---|---:|
| e4 + d4 | 25.0% (…f5 d5 + c4 …e6, 9/15 draws) |
| e4–e5 | 20.8% (…Nf6 d4 + c4 …c5, Black's only move) |
| c4 + d4 | 12.5% |
| f4–f5 | 8.3% |
| d4 + f4 | 2.1% |
| c4 + f3 | 0.0% |

The depth-check leaders here, c4 + d4 and c4 + f3, are refuted by Black's
third-move answers.