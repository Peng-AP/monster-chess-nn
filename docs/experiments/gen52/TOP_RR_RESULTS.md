# Top-group round robin (October 2, 2026)

Step 1 of `docs/plans/OVERNIGHT_20261002_PLAN.md`; descriptive, decides
nothing. Driver `tools/top_round_robin.py`; evidence
`benchmarks/top_rr_20261002/`.

- **Format:** 8 models, 28 pairings × 100 games (50 per colour) = 2,800
  games, at 3,200 simulations from sampled openings.
- **Time:** 05:36 → about 10:15. The chain was restarted at 05:36 (verdict
  mapping fix); pairing 1 resumed from its journal.
- **Ratings:** Bradley–Terry Elo relative to v29 = 0, with 1,000-rep
  bootstrap intervals.

## Ratings against the group

| Model | Elo vs v29 | 95% interval | Score | as White | as Black |
|---|---:|---|---:|---:|---:|
| **gen52 Arm R** | **+40** | +18 to +63 | 61.8% | **44.6%** | 79.0% |
| gen52 Arm L | +16 | −7 to +38 | 58.1% | 37.6% | 78.6% |
| v29 (release) | 0 | — | 55.5% | 35.4% | 75.6% |
| gen52 Arm C | −11 | −34 to +14 | 53.8% | 32.6% | 75.0% |
| gen52 Arm B | −24 | −43 to −1 | 51.6% | 37.0% | 66.3% |
| gen52 Arm A | −36 | −59 to −9 | 49.7% | 25.1% | 74.3% |
| v28 | −105 | −128 to −82 | 38.7% | 17.6% | 59.9% |
| gen49 | −158 | −183 to −134 | 30.8% | 11.1% | 50.4% |

## Head-to-head (row player's score, %)

| | R | L | v29 | C | B | A | v28 | gen49 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **R** | – | 51.0 | **42.5** | 66.0 | 62.0 | 67.5 | 74.5 | 69.0 |
| **L** | 49.0 | – | 48.5 | 56.0 | 55.0 | 65.0 | 70.0 | 63.0 |
| **v29** | **57.5** | 51.5 | – | **37.0** | 53.0 | **37.5** | 75.5 | 76.5 |
| **C** | 34.0 | 44.0 | **63.0** | – | 45.5 | 50.5 | 69.5 | 70.0 |
| **B** | 38.0 | 45.0 | 47.0 | 54.5 | – | 49.0 | 65.5 | 62.5 |
| **A** | 32.5 | 35.0 | **62.5** | 49.5 | 51.0 | – | 56.0 | 61.5 |

The v28 and gen49 rows are the complements of their columns.

## Is anything stronger than v29?

- **Against the group, yes: Arm R.**
  - It rates +40 Elo, and the interval excludes zero.
  - It has the best overall score (61.8%).
  - It has by far the best White (44.6% against v29's 35.4%). That is the
    colour the label fix targeted.
- **Head-to-head, no.** v29 beats Arm R 57.5–42.5 (100 games), consistent
  with Arm R's gate v4 legs against v29 (38.7% game-weighted).
- **The group is strongly non-transitive:**
  - v29 beats R but **loses to Arm C (37.0%) and Arm A (37.5%)**;
  - R beats C and A by about 2:1.
  - No model beats every other model.
- Arm L (the wide tower) is level with v29 and R head-to-head (48.5–51.5,
  49–51).

**What this means for promotion.** Under `PROMOTION_RULE.md` the binding
test is gate v4 against the release. Arm R fails it, so it is not eligible
however well it does against the group. Whether a +40 group rating with a
head-to-head loss to the release should count is a question for the owner,
not something this run decides.
