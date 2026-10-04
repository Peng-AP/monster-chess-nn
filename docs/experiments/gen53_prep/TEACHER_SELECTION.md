# Teacher selection for gen53: results (October 3–4, 2026)

Plan and rule: `docs/plans/TEACHER_SELECTION_PLAN.md` (declared before any
game). Chain `tools/teacher_chain_20261003.py`, October 3 15:00 → October 4
00:57, with no failures. Evidence is in
`benchmarks/teacher_selection_20261003/` (`selection.json`,
`rr_{1600,6400,12800}/`). Evaluation only; nothing trained or promoted.

## Recommendation: **gen52 Arm R** (clear winner, no tie)

| Candidate | Eligible (bias ≤ ±0.03) | Score: mean Elo vs v29 over 1,600 / 6,400 / 12,800 | 95% interval |
|---|---|---:|---|
| **gen52 Arm R** | yes (−0.005) | **+137.5** | +114 to +159 |
| gen52 Arm LR | yes (−0.007) | +88.6 | +67 to +110 |
| gen52 Arm L | no (+0.071) | +56.5 | +37 to +77 |
| gen52 Arm C | no (+0.049) | +20.8 | +2 to +42 |
| v29 | yes (+0.017) | 0 | — |

- **R minus LR:** +48.9 (95% interval +26.3 to +69.3). It excludes 0, so it
  is **not a tie**, and by the pre-declared rule **Arm R is the recommended
  teacher**.
- Every gen52 candidate outscores v29 at the teaching depths, so v29 is the
  weakest of the five for teaching.

## Elo by depth (5-player round robins, relative to v29 = 0)

| Depth (role) | Arm R | Arm LR | Arm L | Arm C |
|---|---:|---:|---:|---:|
| 1,600 (self-play, 100 g/pair) | +100 (68–137) | +75 (43–110) | **+121** (90–153) | +97 (63–134) |
| 6,400 (continuations, 100 g/pair) | **+130** (95–169) | +62 (28–94) | −16 (−46 to +13) | −21 (−53 to +14) |
| 12,800 (reanalysis, 60 g/pair) | **+183** (138–238) | +129 (86–176) | +65 (25–108) | −14 (−51 to +23) |

**Arm R's advantage grows with depth.** It is second to the ineligible Arm L
at 1,600 and first by a wide margin at 6,400 and 12,800, where the
deep-value and reanalysis data are generated. This repeats the October 3
ladder finding (+149 from 3,200 to 12,800) in an independent design.

## Head-to-head (row player's score, %)

At 1,600:

| | v29 | R | LR | L | C |
|---|---:|---:|---:|---:|---:|
| v29 | – | 34.0 | 37.0 | 34.0 | 39.5 |
| R | 66.0 | – | 46.0 | 47.5 | 55.5 |
| LR | 63.0 | 54.0 | – | 39.5 | 41.0 |
| L | 66.0 | 52.5 | 60.5 | – | 51.0 |
| C | 60.5 | 44.5 | 59.0 | 49.0 | – |

At 6,400:

| | v29 | R | LR | L | C |
|---|---:|---:|---:|---:|---:|
| v29 | – | 47.0 | 25.0 | 47.0 | 59.5 |
| R | 53.0 | – | **62.0** | **75.0** | **78.5** |
| LR | 75.0 | 38.0 | – | 55.0 | 54.0 |
| L | 53.0 | 25.0 | 45.0 | – | 44.0 |
| C | 40.5 | 21.5 | 46.0 | 56.0 | – |

At 12,800:

| | v29 | R | LR | L | C |
|---|---:|---:|---:|---:|---:|
| v29 | – | 50.0 | 22.5 | 25.8 | 51.7 |
| R | 50.0 | – | **65.8** | **74.2** | **85.0** |
| LR | 77.5 | 34.2 | – | 63.3 | 64.2 |
| L | 74.2 | 25.8 | 36.7 | – | 58.3 |
| C | 48.3 | 15.0 | 35.8 | 41.7 | – |

## Caveats

1. **Arm R does not beat v29 head-to-head at depth.** It scores 53% at 6,400
   and 50% at 12,800, where LR scores 75% and 77.5%. R's rating comes from
   dominating the other gen52 models (62–85%).
   - v29 has an unusual match-up advantage against R, as it did at 3,200
     (57.5%).
   - As a teacher, R's self-play would not face v29, so this matters less
     than it would for a release.
2. **12,800 used 60 games per pairing**, so its intervals are the widest
   (±45). The 6,400 result alone already separates R from LR.
3. Colour scores against the field carry block bias, so they are not
   compared with 50%.

## What this decides

By the pre-declared rule, **gen53's teacher should be gen52 Arm R**
(`models/candidates/bootstrap_main_gen_0052_ramp/arena_selected.pt`). For
gen53, R's recipe also carries forward: deep-value labels ramped like every
other source. Launching gen53 remains the owner's decision; its rule 1 (a
gen52 teacher must pass gate v4 against v29) is not met by R and would need
to be waived or revised.
