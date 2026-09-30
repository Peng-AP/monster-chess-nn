# Elo round robin: September 29, 2026

Owner request: *"an evaluatory round robin tournament aimed at finding out elo
values. Set the weakest version that beats human to be 1600 elo … a large pool
of players, but don't make it go too long."*

Driver `tools/elo_tournament.py`. Evidence is in `benchmarks/elo_rr_20260929/`:
the manifest with model hashes, one report and game journal per pairing, and
`ratings.json`. The run took 5 h 20 min (16:20 → 21:40) as one managed job,
with no failures.

## Setup

- **16 players, 120 pairings, 40 games each (20 per colour): 4,800 games.**
- **Instrument:** free play from sampled openings (16 temperature plies),
  3,200 simulations for both sides. That is the website's and the gates'
  operating point.
- **Anchor: v21 = 1600.** It is the weakest version on record that beats the
  owner:
  - v19's White was "easy to beat";
  - the owner called v21 "a very strong player";
  - the September 4 round robin recorded human play as below v21.

  Moving the anchor is a constant shift and needs no replay.
- **Fit:** Bradley–Terry maximum likelihood on points (a draw counts half),
  with one virtual draw per pairing so 40–0 sweeps stay finite. The 95%
  intervals come from a 1,000-rep parametric bootstrap that resamples every
  pairing's games.

## Ratings

| # | Player | Elo | 95% interval | Score (600 g) | as White | as Black |
|---:|---|---:|---|---:|---:|---:|
| 1 | **v29** (release) | **2592** | 2534–2657 | 86.9% | 81.8% | 92.0% |
| 2 | gen52 Arm C | 2576 | 2519–2641 | 85.9% | 76.8% | 95.0% |
| 3 | gen52 Arm B | 2570 | 2517–2635 | 85.5% | 79.8% | 91.2% |
| 4 | v28 | 2497 | 2443–2561 | 80.5% | 75.8% | 85.2% |
| 5 | gen49 | 2466 | 2414–2525 | 78.2% | 75.5% | 81.0% |
| 6 | B2 (held-out attention model) | 2326 | 2280–2379 | 67.7% | 66.7% | 68.8% |
| 7 | v27 | 2199 | 2154–2248 | 58.2% | 55.5% | 60.8% |
| 8 | v26 | 2132 | 2086–2179 | 53.2% | 50.7% | 55.8% |
| 9 | v25 | 2078 | 2035–2126 | 49.4% | 49.3% | 49.5% |
| 10 | v24 | 2071 | 2027–2120 | 48.9% | 45.2% | 52.7% |
| 11 | v23 | 1774 | 1724–1825 | 29.9% | 33.2% | 26.7% |
| 12 | v22 | 1690 | 1642–1743 | 24.9% | 27.0% | 22.8% |
| 13 | **v21 (anchor)** | **1600** | — | 19.7% | 25.0% | 14.3% |
| 14 | v20 | 1554 | 1503–1605 | 17.1% | 19.3% | 14.8% |
| 15 | v19 | 1446 | 1384–1504 | 11.6% | 14.7% | 8.5% |
| 16 | v17 | 1198 | 1132–1258 | 2.2% | 3.8% | 0.7% |

The colour columns are scores against the field. They carry colour bias and
must not be compared to 50%.

## What the numbers say

- **v29 is about 990 Elo above the anchor.** At that gap the expected score of
  v21, and so roughly of the owner, is about 0.3%. v29 is ~1,390 Elo above v17.
- **The top three are not separated.** v29, gen52 Arm C and Arm B lie within
  22 Elo, and their intervals overlap almost entirely. The gen52 arms failed
  gate v4 on their White floors, not because they are weaker overall.
- **Two large steps:**
  - v23 → v24: +297, the September bootstrap free-play era;
  - v27 → v28: +298.

  v28 → v29 adds +95. v24 and v25 are level (7 Elo apart).
- **Cross-check against the September 4 round robin** (free play, v21 as base):

  | Step above v21 | Now | September 4 |
  |---|---:|---:|
  | v22 | +90 | +88 |
  | v23 | +174 | +254 |
  | v24 | +471 | +564 |

  The two agree on order and rough size, though the September 4 run used a
  different instrument.
- **The game is decisive at these depths:** only 10.0% of the 4,800 games were
  drawn.

## Where the ratings mislead (non-transitivity)

The RMS gap between observed and predicted scores is 0.072 per pairing. The
biggest real surprises, all among the strong models:

| Pairing | Observed | Rating predicts |
|---|---:|---:|
| v28 vs gen49 | 77.5% | 54.5% |
| v29 vs gen52 Arm C | 33.75% | 52.2% |
| v28 vs v29 | 15.0% | 36.7% |
| v27 vs gen49 | 2.5% | 17.7% |
| v27 vs B2 | 15.0% | 32.5% |

- **Arm C beats v29 head-to-head, 66% over 40 games.** This matches its gate
  legs (65.3% over 800), but it is rated below v29 because v29 does better
  against everyone else.
- Several mismatched pairings fit badly only because the weak side scraped a
  few points: v17 got 1 point off gen49 where the ratings predict almost none.
  That is noise at the extremes, not structure.
- **Read head-to-head results for direct questions** ("is A better than B?").
  Read the ratings for placement across the whole pool. Never chain the
  ratings into a claim a direct match contradicts.

## Caveats

- These are ratings within this model pool, at 3,200 simulations, from sampled
  openings. They are not calibrated to any human rating list, and "1600" is a
  label pinned to v21, not a measurement of the owner.
- A different simulation count changes values (see "Depth changes values" in
  the measurement rules).
- The virtual draw per pairing pulls the extremes slightly toward the middle.
  v17's rating depends most on it, because it scored 13 points in 600 games.
