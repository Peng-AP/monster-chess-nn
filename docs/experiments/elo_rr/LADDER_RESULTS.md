# Strength ladder: search scaling and playable levels (September 30, 2026)

This was an overnight run on the owner's instruction ("run something useful
overnight"). Driver `tools/elo_ladder.py`; evidence in
`benchmarks/elo_ladder_20260930/`. The run took 3.4 h (01:35 → 04:58) as one
managed job, with no failures.

## Design

Nine (model, simulations) settings each played the round robin's 16 models,
which stayed at 3,200 simulations:
- **Stage 1:** 8 games against every model (4 per colour).
- **Stage 2:** 32 more against the 6 models rated closest to the stage-1
  estimate.

That is 320 games per setting and 2,880 in all. Ratings come from **one joint
Bradley–Terry fit** over these games plus the round robin's 4,800 (7,680
games), anchored v21 = 1600, with 1,000-rep bootstrap intervals.

## Finding 1: search stops paying above about 3,200 simulations

v29 at every depth, joint fit:

| Simulations | Elo | 95% interval | Gain per doubling |
|---:|---:|---|---:|
| 12 | 2015 | 1969–2064 | — |
| 50 | 2079 | 2030–2130 | ~32 |
| 200 | 2201 | 2157–2254 | ~61 |
| 800 | 2325 | 2282–2376 | ~62 |
| **3,200** | **2450** | 2411–2496 | ~62 |
| 6,400 | 2469 | 2427–2512 | ~19 |
| 12,800 | 2478 | 2438–2527 | ~9 |

**Direct matches against v29 at 3,200** (40 games each, SE about ±7.5 pp):

| v29 setting | Score vs v29 at 3,200 |
|---|---:|
| 6,400 | 51.25% |
| 12,800 | 55.0% |
| 800 | 30.0% |

What this means:

- **v29 gains about 62 Elo per doubling from 50 to 3,200, then almost
  nothing.** Four times the search (12,800) is worth about +28 Elo, which is
  not significant.
- **More search will not make v29 materially stronger.** Further strength has
  to come from the network. The site's 3,200 sits at the knee of the curve.
- **The 12,800 depth guard in gate v4 does not test a weaker engine.** It tests
  about the same strength at depth.
- **The network alone is strong.** v29 with 12 simulations (almost no search)
  rates 2015, level with v24–v26 at full search.
- **v17 barely responds to search.** It rates 1128 at 8 simulations, 1099 at
  50, 1124 at 400 and 1255 at 3,200. This is consistent with an uninformative
  value head, but that is not tested here.

## Finding 2: the joint fit compresses the scale

Adding the ladder's games moves the round-robin ratings:

| | Round robin alone | Joint fit (with ladder) |
|---|---:|---:|
| v29 | 2592 | 2450 |
| v24 | 2071 | 1983 |
| v17 | 1198 | 1255 |

Order is unchanged, but the v21 → v29 gap shrinks from 992 to 850 Elo. RMS
residual rises from 0.072 to 0.085.

- The ladder settings score relatively better against distant opponents than
  the round-robin fit predicts. That is non-transitivity again.
- The size of Elo gaps across 1,000 points depends on who played whom. Treat
  gaps that large as ±15%.
- The website's labels use the **joint fit**, the larger body of evidence.

## Finding 3: a colour specialist (from existing round-robin games, no new play)

- **v29 as White with gen52 Arm C as Black** rates about **2632** on the
  round-robin scale, against v29's 2592 (+40, inside noise). It scores 91.2%
  against its 14 opponents; v29 scored 86.9% against its 15.
- It is **not promotable**. Arm C's Black fails gate v4's 12,800 guard against
  v29 (Black 51.25% vs deep par 81.25%).
- It is an option only if the owner wants a stronger engine at the site's
  3,200 setting.

## Website difficulty levels (deployed September 30)

The site now offers ten opponents labelled with measured Elo; the default is
still v29 at 3,200:

| Elo | Opponent | Simulations |
|---:|---|---:|
| 2450 | v29 (default) | 3,200 |
| 2326 | v28 | 3,200 |
| 2325 | v29 | 800 |
| 2201 | v29 | 200 |
| 2015 | v29 | 12 |
| 1748 | v23 | 3,200 |
| 1600 | v21 | 3,200 |
| 1477 | v19 | 3,200 |
| 1255 | v17 | 3,200 |
| 1128 | v17 (easiest) | 8 |

**Not yet available:** anything below about 1,100. v17 is the weakest
loadable network, and cutting its search does not weaken it further.
Beginner levels would need a deliberately noisier player (for example
sampling moves at temperature), which would have to be measured the same way.

## Suggested next steps (none started)

1. **Strength must come from the network, not search.** This supports the
   training-side work in GEN52/POOLCAP_RESULTS (the teacher-pessimism
   hypothesis).
2. If a human rating is wanted, the owner can now play the levels. A few games
   at a level where he wins about half the time would place him on this scale
   with a measurement instead of an assumption.
