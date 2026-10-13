# Per-colour baselines on the census book, October 13, 2026

Owner, October 12, on the census recommendations: "Yes, work on it".
Recommendation 1 of `OPENING_CENSUS.md` was a per-colour instrument. This page
holds its baselines.

## What was measured

- **Book:** the census key positions, the 14 best White first turns with
  Black to move. Each position has 20 copies and 8 sampled half-moves, and
  every entry is played with colours swapped. All games at 3,200 sims on the
  repetition-aware engine.
- **References:** gen55 and v29.
- **Columns:**
  - "White vs gen55" is the model's White against gen55's Black.
  - "Black vs v29" is the model's Black against v29's White, and so on.
  - Draws count as half, including the ±0.5 move-limit relabel.
- **Sources:**
  - gen55 and v29 come from the census phase 1b cells, the same book and
    sampling.
  - gen54, gen53, v28 and gen49 come from `tools/colour_instrument.py`, 280
    games per cell.
  - Each reference against itself has 560 games, and both colour cells come
    from those same games.
- **Evidence:**
  - `benchmarks/colour_instrument/<model>/{result.json,vs_*.json}`;
  - `benchmarks/opening_census_20261012/phase1b/report.json`.
- **Table:** `py -3 tools/colour_baselines_table.py`.
- **Run:** driver `tools/colour_baselines_20261013.py`, 22:13 to 07:08
  (8.9 h).

## Results

| Model | White vs gen55 | White vs v29 | Black vs gen55 | Black vs v29 | White mean | Black mean |
|---|---:|---:|---:|---:|---:|---:|
| gen55 | **21.7%** (self, 330/560) | 15.5% (181) | 78.3% (self, 330/560) | 88.6% (200) | **18.6%** | 83.4% |
| v29 (release) | 11.4% (200) | 9.7% (self, 357/560) | 84.5% (181) | 90.3% (self, 357/560) | 10.6% | 87.4% |
| gen54 | 16.2% (192) | **18.8%** (188) | 83.0% (182) | 81.6% (186) | 17.5% | 82.3% |
| gen53 | 13.2% (202) | 14.5% (189) | **90.5%** (188) | **92.1%** (199) | 13.8% | **91.3%** |
| v28 | 5.7% (211) | 6.6% (197) | 71.4% (188) | 86.6% (200) | 6.2% | 79.0% |
| gen49 | 2.5% (209) | 6.8% (207) | 85.7% (189) | 67.7% (208) | 4.6% | 76.7% |

Brackets: distinct games out of 280, unless marked otherwise.

**Paired difference from v29**, in pp:

- Position-weighted over the 14 positions.
- SE in brackets: within-position binomial, because the book is fixed.
- Repeated games make the true SE somewhat larger.

| Model | White vs gen55 | White vs v29 | Black vs gen55 | Black vs v29 |
|---|---:|---:|---:|---:|
| gen55 | +10.3 (2.4) | +5.8 (2.3) | **−6.2 (2.5)** | −1.7 (2.1) |
| gen54 | +4.8 (2.6) | +9.0 (2.3) | −1.4 (2.8) | **−8.7 (2.3)** |
| gen53 | +1.8 (2.5) | +4.7 (2.1) | +6.1 (2.6) | +1.9 (1.9) |
| v28 | −5.7 (2.2) | −3.1 (1.8) | −13.0 (2.7) | −3.7 (2.0) |
| gen49 | −8.9 (2.0) | −2.9 (1.8) | +1.2 (2.8) | −22.6 (2.6) |

## Against the normal-start Elo and gates

The normal-start order is gen55 > gen54 > gen53 > v29 > v28 > gen49:

- gen55 is unrated. It scored 69–71% against gen54 in gate v4.
- Site Elo, joint fit: gen54 2641 (provisional), gen53 2555, v29 2450,
  v28 2326.
- gen49 is about 30 Elo below v28 in the round robin.

**White on the book follows that order:**

- gen55 18.6% ≈ gen54 17.5% > gen53 13.8% > v29 10.6% > v28 6.2% > gen49 4.6%.
- This **contradicts gate v4's White floor.** gen54 and gen55 failed gate v4
  against v29 on the White floor, by −19 to −21 pp at the normal start.
  Against v29's Black on the book, they are the two best Whites, at +9.0 and
  +5.8 pp.
- Their normal-start failures are the openings they choose. For example,
  gen55's d4 + f4 went 0/0/35 against v29. On the book, that position scores 0%
  against v29's Black for every model except gen53 and gen49, at 17.5% each.
- The gate's White floor measures repertoire, not White strength. This is
  what the census predicted.

**Black on the book contradicts the Elo:**

- gen53 91.3% > v29 87.4% > gen55 83.4% ≈ gen54 82.3% > v28 79.0% ≈ gen49 76.7%.
- At the normal start, gen55 won all 480 of its Black games against v29.
  From the book positions it is 6.2 pp below v29's Black against gen55's
  White.
- The shortfall is mostly two lines where gen55's own White beats gen55's
  Black:

  | Line | gen55's White vs its own Black (W/D/L) | vs v29's Black |
  |---|---|---|
  | e1e2 e2e3 | 21/0/19 (17 distinct) | 1/0/19 |
  | e2e3 f2f4 | 18/0/22 (21 distinct) | 0/0/20 |

- In both lines, gen53's, gen54's and v29's Blacks score 95–100% against
  gen55's White.
- gen54's Black shortfall is against v29's White (−8.7 pp).
- **gen53 has the best Black of the six**, ahead of v29 against both
  references (+6.1 and +1.9 pp).
- Each line rests on 20–40 games, and some on only a few distinct games, so
  quote the totals, not the lines (`book-line results are n = 1`).

## What it means for gen56

**1. The proposed promotion rule:**

- The rule (`GEN56_PLAN.md`) requires each of the four cells to be within
  3 pp of v29's own.
- **Of the six models, only gen53 passes it.**
- gen55 fails on Black vs gen55's White (−6.2 pp).
- gen54 fails on Black vs v29's White (−8.7 pp).

**2. gen53 is still ineligible:**

- Re-measured on the repetition-aware engine on October 13: held-out mean
  88.0% against a bar of 91.8% (v29's 92.8% minus 1 pp).
- The gap is gen53's White against v27: 41/1/38 W/D/L, from the normal start.
- See `docs/plans/GEN56_PLAN.md`, option c.

**3. Selection's Black reference is weak:**

- gen56's colour check keeps candidates whose Black is within 2 pp of the
  teacher's.
- The teacher is gen55, whose Black is the second weakest of the recent
  models on this book.
- So a candidate can pass selection with gen55's Black and still fail the
  proposed promotion rule on the cell gen55 fails.

**4. Teachers:**

- gen54 is the plan's second extra teacher. It has the weakest Black of
  gen53, gen54 and gen55 on the book.
- gen53 has the strongest Black of all six.
- gen53 is not held out, so it can be a teacher.

The resulting options are written into `docs/plans/GEN56_PLAN.md` for the
owner. The recipes and the decision file are unchanged.
