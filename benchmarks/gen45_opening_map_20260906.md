# Gen45 observed opening repertoire — September 6, 2026

All sources are completed free-opening games at 3,200 simulations per move: 800 against gen44, 200 against gen42, and 200 self-play games.

These are observed frequencies under temperature 0.5 for the first 16 search plies, not zero-temperature rankings or a uniform opening book. Self-play contributes to both color maps; opponent moves are conditional context, not gen45 choices.

Notation: `e4/e5` means White uses both moves to advance e2–e4–e5. Piece letters/captures are shown; check markers and SAN disambiguation are omitted.

## Zero-temperature preference check

An additional 21 fresh-tree probes used 3,200 simulations, seven starting cases,
three seeds each and eight searched continuation plies. Only the standard initial
position (row zero) of the human-log file was used; no human continuation was
consulted or added to training. All three seeds agreed within each case. These
repeats are a reproducibility check, not independent proof of opening quality.
Artifact: `benchmarks/gen45_opening_preferences_20260906.json`.

White's default continuation is `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5`.
From the separately forced `1. e4/e5` position, gen45 Black continues with
`...f6`, then `...fxe5`, then `...Nc6` against that White line.

| White opening | Gen45 Black's top response at temperature zero |
|---|---|
| e4/e5 | f6 |
| d4/d5 | c5 |
| d4/e4 | e5 |
| c4/c5 | d6 |
| f4/f5 | d6 |
| e4/f4 | f5 |

Pooled gen45-White first turns: e4/e5 665/700 (95.0%); d4/d5 29/700
(4.14%); d4/e4 5/700 (0.71%); e4/d4 1/700 (0.14%).
Across 470 recorded gen45-Black replies to e4/e5, f6 occurred 307 times
(65.3%), d6 118 (25.1%), e6 30 (6.4%), Nh6 12 (2.6%), d5 3 (0.6%).
These pooled frequencies weight the particular opponent mix, not opponents equally.

## Gen45 as White (700 games)

### Against self (200 games)

First White turn: e4/e5 195/200 (97.5%); d4/e4 2/200 (1.0%); d4/d5 2/200 (1.0%); e4/d4 1/200 (0.5%)

| Position reached | Games | Gen45 response frequencies within that branch |
|---|---:|---|
| 1. e4/e5 f6 | 134 | d4/d5: 133/134 (99.25%); f4/f5: 1/134 (0.75%) |
| 1. e4/e5 d6 | 42 | d4/d5: 42/42 (100.0%) |
| 1. e4/e5 e6 | 12 | f4/f5: 11/12 (91.67%); d4/d5: 1/12 (8.33%) |
| 1. e4/e5 Nh6 | 7 | d4/d5: 7/7 (100.0%) |
| 1. d4/e4 e5 | 2 | f4/c4: 2/2 (100.0%) |
| 1. d4/d5 c5 | 2 | e4/Ke2: 2/2 (100.0%) |
| 1. e4/d4 e5 | 1 | f4/c4: 1/1 (100.0%) |

Most frequent complete three-turn prefixes:

- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 Nc6` — 94/200 (47.0%).
- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 Na6` — 36/200 (18.0%).
- `1. e4/e5 d6 2. d4/d5 Na6 3. c4/Ke2 Nc5` — 13/200 (6.5%).
- `1. e4/e5 d6 2. d4/d5 Nd7 3. exd6/dxc7 Qxc7` — 13/200 (6.5%).
- `1. e4/e5 e6 2. f4/f5 Nh6 3. d4/d5 exf5` — 11/200 (5.5%).
- `1. e4/e5 Nh6 2. d4/d5 e6 3. f4/f5 exf5` — 7/200 (3.5%).
- `1. e4/e5 d6 2. d4/d5 Nd7 3. f4/c4 Nc5` — 4/200 (2.0%).
- `1. e4/e5 d6 2. d4/d5 Nd7 3. c4/f4 Nc5` — 4/200 (2.0%).

### Against gen44 (400 games)

First White turn: e4/e5 374/400 (93.5%); d4/d5 24/400 (6.0%); d4/e4 2/400 (0.5%)

| Position reached | Games | Gen45 response frequencies within that branch |
|---|---:|---|
| 1. e4/e5 f6 | 348 | d4/d5: 346/348 (99.43%); f4/f5: 2/348 (0.57%) |
| 1. d4/d5 c5 | 24 | e4/Ke2: 18/24 (75.0%); Kd2/Ke3: 3/24 (12.5%); e4/Kd2: 2/24 (8.33%); Kd2/e4: 1/24 (4.17%) |
| 1. e4/e5 d5 | 19 | Ke2/Ke3: 18/19 (94.74%); c4/Ke2: 1/19 (5.26%) |
| 1. e4/e5 e6 | 7 | f4/f5: 5/7 (71.43%); d4/d5: 2/7 (28.57%) |
| 1. d4/e4 e5 | 2 | f4/d5: 1/2 (50.0%); f4/c4: 1/2 (50.0%) |

Most frequent complete three-turn prefixes:

- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 Na6` — 297/400 (74.25%).
- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 d6` — 33/400 (8.25%).
- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 Nc6` — 12/400 (3.0%).
- `1. d4/d5 c5 2. e4/Ke2 d6 3. Kf3/Kf4 f6` — 10/400 (2.5%).
- `1. e4/e5 e6 2. f4/f5 Nh6 3. d4/d5 exf5` — 5/400 (1.25%).
- `1. e4/e5 d5 2. Ke2/Ke3 f5 3. c4/cxd5 c6` — 5/400 (1.25%).
- `1. e4/e5 d5 2. Ke2/Ke3 f5 3. e6/Kd4 Na6` — 5/400 (1.25%).
- `1. e4/e5 d5 2. Ke2/Ke3 f6 3. c4/cxd5 fxe5` — 4/400 (1.0%).

### Against gen42 (100 games)

First White turn: e4/e5 96/100 (96.0%); d4/d5 3/100 (3.0%); d4/e4 1/100 (1.0%)

| Position reached | Games | Gen45 response frequencies within that branch |
|---|---:|---|
| 1. e4/e5 f6 | 90 | d4/d5: 89/90 (98.89%); f4/f5: 1/90 (1.11%) |
| 1. e4/e5 e6 | 5 | f4/f5: 3/5 (60.0%); d4/d5: 2/5 (40.0%) |
| 1. d4/d5 c5 | 3 | e4/Ke2: 3/3 (100.0%) |
| 1. e4/e5 d5 | 1 | Ke2/Ke3: 1/1 (100.0%) |
| 1. d4/e4 d5 | 1 | Ke2/e5: 1/1 (100.0%) |

Most frequent complete three-turn prefixes:

- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 Na6` — 48/100 (48.0%).
- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 Nc6` — 30/100 (30.0%).
- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 d6` — 10/100 (10.0%).
- `1. e4/e5 e6 2. d4/d5 Nh6 3. f4/f5 exf5` — 2/100 (2.0%).
- `1. e4/e5 e6 2. f4/f5 Nh6 3. d4/d5 exf5` — 2/100 (2.0%).
- `1. e4/e5 f6 2. d4/d5 fxe5 3. f4/f5 d6` — 1/100 (1.0%).
- `1. e4/e5 e6 2. f4/f5 exf5 3. d4/d5 Na6` — 1/100 (1.0%).
- `1. d4/d5 c5 2. e4/Ke2 d6 3. Ke3/Kf4 e5` — 1/100 (1.0%).

## Gen45 as Black (700 games)

### Against self (200 games)

First White turn: e4/e5 195/200 (97.5%); d4/e4 2/200 (1.0%); d4/d5 2/200 (1.0%); e4/d4 1/200 (0.5%)

| Position reached | Games | Gen45 response frequencies within that branch |
|---|---:|---|
| 1. e4/e5 | 195 | f6: 134/195 (68.72%); d6: 42/195 (21.54%); e6: 12/195 (6.15%); Nh6: 7/195 (3.59%) |
| 1. d4/e4 | 2 | e5: 2/2 (100.0%) |
| 1. d4/d5 | 2 | c5: 2/2 (100.0%) |
| 1. e4/d4 | 1 | e5: 1/1 (100.0%) |

Most frequent complete three-turn prefixes:

- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 Nc6` — 94/200 (47.0%).
- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 Na6` — 36/200 (18.0%).
- `1. e4/e5 d6 2. d4/d5 Na6 3. c4/Ke2 Nc5` — 13/200 (6.5%).
- `1. e4/e5 d6 2. d4/d5 Nd7 3. exd6/dxc7 Qxc7` — 13/200 (6.5%).
- `1. e4/e5 e6 2. f4/f5 Nh6 3. d4/d5 exf5` — 11/200 (5.5%).
- `1. e4/e5 Nh6 2. d4/d5 e6 3. f4/f5 exf5` — 7/200 (3.5%).
- `1. e4/e5 d6 2. d4/d5 Nd7 3. f4/c4 Nc5` — 4/200 (2.0%).
- `1. e4/e5 d6 2. d4/d5 Nd7 3. c4/f4 Nc5` — 4/200 (2.0%).

### Against gen44 (400 games)

First White turn: e4/e5 269/400 (67.25%); d4/d5 131/400 (32.75%)

| Position reached | Games | Gen45 response frequencies within that branch |
|---|---:|---|
| 1. e4/e5 | 269 | f6: 169/269 (62.83%); d6: 75/269 (27.88%); e6: 18/269 (6.69%); Nh6: 5/269 (1.86%); d5: 2/269 (0.74%) |
| 1. d4/d5 | 131 | c5: 130/131 (99.24%); c6: 1/131 (0.76%) |

Most frequent complete three-turn prefixes:

- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 Nc6` — 82/400 (20.5%).
- `1. e4/e5 f6 2. d4/d5 fxe5 3. f4/f5 Na6` — 41/400 (10.25%).
- `1. e4/e5 d6 2. d4/d5 Na6 3. c4/Ke2 Nc5` — 31/400 (7.75%).
- `1. e4/e5 f6 2. d4/d5 fxe5 3. c4/c5 Na6` — 29/400 (7.25%).
- `1. e4/e5 e6 2. f4/f5 Nh6 3. d4/d5 exf5` — 16/400 (4.0%).
- `1. d4/d5 c5 2. Kd2/Kd3 d6 3. e4/Kc4 Nf6` — 11/400 (2.75%).
- `1. e4/e5 f6 2. f4/f5 Nh6 3. d4/d5 fxe5` — 10/400 (2.5%).
- `1. d4/d5 c5 2. e4/Ke2 d6 3. Kd3/Kc4 Qb6` — 9/400 (2.25%).

### Against gen42 (100 games)

First White turn: d4/d5 94/100 (94.0%); e4/e5 6/100 (6.0%)

| Position reached | Games | Gen45 response frequencies within that branch |
|---|---:|---|
| 1. d4/d5 | 94 | c5: 94/94 (100.0%) |
| 1. e4/e5 | 6 | f6: 4/6 (66.67%); d5: 1/6 (16.67%); d6: 1/6 (16.67%) |

Most frequent complete three-turn prefixes:

- `1. d4/d5 c5 2. e4/Ke2 d6 3. e5/Ke3 f6` — 6/100 (6.0%).
- `1. d4/d5 c5 2. e4/e5 d6 3. Kd2/Ke3 f6` — 6/100 (6.0%).
- `1. d4/d5 c5 2. e4/Kd2 d6 3. f4/e5 f6` — 5/100 (5.0%).
- `1. d4/d5 c5 2. Kd2/Kd3 d6 3. e4/Kc4 Qb6` — 4/100 (4.0%).
- `1. d4/d5 c5 2. e4/e5 d6 3. Kd2/Kd3 f6` — 4/100 (4.0%).
- `1. d4/d5 c5 2. e4/Ke2 d6 3. e5/Kd3 f6` — 4/100 (4.0%).
- `1. d4/d5 c5 2. e4/Ke2 d6 3. Kd3/e5 f6` — 4/100 (4.0%).
- `1. d4/d5 c5 2. e4/e5 d6 3. Ke2/Kd3 f6` — 4/100 (4.0%).

