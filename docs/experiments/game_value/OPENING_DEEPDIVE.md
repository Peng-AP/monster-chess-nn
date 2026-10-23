# Opening deep dive: six first turns, two turns deep, October 21–23, 2026

**Owner, October 21:**

- "Afterwards, we can hone in on those 6 first turns, and their only
  reply(s)."
- "Just make sure a certain line isn't judged on, say, the 3 best replies
  after it if 2 are substantially worse than the best response. Go ahead."

**Run:**

- Driver `tools/opening_deepdive.py`; rules declared before any result.
- Rehearsal, then production: October 21 21:38 → October 23 06:11,
  about 32.5 h. No failures.
- Evidence: `benchmarks/opening_deepdive_20261021/`, with `result.json` and
  each stage's `report.json`.
- Players, cells and scoring are the atlas's (`OPENING_ATLAS.md`):
  - White gen56 and gen55; Black gen56 and gen53.
  - White's score, with draws counted as half.

## How lines were judged (the owner's rule)

- **Strict minimax, never an average:**
  - A position where Black moves is worth Black's best reply.
  - A position where White moves is worth White's best move.
- **Candidates:** a move more than 10 pp worse than the best for the mover is
  dropped. Moves within 10 pp are kept only as candidates for the best, at
  most 4. They get more games, and the value is still the single best one.
- **Fresh games:** every value is measured on games that played no part in
  choosing the move.

| Stage | What | Games |
|---|---|---:|
| Lines | The six first turns; Black's candidate replies by refined atlas value | 14 lines |
| **A** | Every White second turn on every line (776), 3,200 sims | 12,416 |
| **B** | The 4 best per line by A, fresh games at 12,800. W2* = the best by B | 896 |
| **C** | Every Black reply to W2* (18–30 per line), 3,200 sims | 2,704 |
| **CR** | Black's candidate replies (within 10 pp of the best by C), fresh games at 3,200 | 528 |

- **A line's value** is the worst CR value among Black's candidate answers.
  That is Black's best answer to White's best try.
- **A first turn's value** is the worst of its lines.

## Result

| First turn | Value | Deciding line: Black reply → White's best try → Black's best answer | W/D/L | Atlas (12,800 check) |
|---|---:|---|---|---:|
| **e4 + d4** | **25.0%** | …f5 → d5 + c4 → …e6 | 0/9/6 | 25% |
| **e4–e5** | **20.8%** | …Nf6 → d4 + c4 → **…c5** | 3/0/12 | 25% |
| c4 + d4 | 12.5% | …d6 → d5 + c5 → …Na6 | 0/3/12 | 37.5% |
| f4–f5 | 8.3% | …Nh6 → d4–d5 → …Nf5 | 0/3/12 | 12.5% |
| d4 + f4 | 2.1% | …d6 → c4–c5 → …Na6 | 0/1/14 | 25% |
| c4 + f3 | 0.0% | …d5 → Kd1 + c5 → …f5 | 0/0/15 | 31% |

Each value rests on 15 target games at the deciding line, so about
±10–12 pp.

**Main finding: with best play two turns deep, no first turn holds.**

- e4 + d4 and e4–e5 lead.
- The two the atlas ranked first at depth, c4 + d4 and c4 + f3, fall to
  12.5% and 0%. Black's third-move answers (stage C) refute White's best
  second turns there.
- **e4 + d4 …f5 d5 + c4 …e6 is the only line that is mostly draws** (9 of 15).

**Caveat on e4 + d4.** The owner's rule left only …f5 as a candidate reply,
because …d5 was 17.5 pp worse for Black at the first-reply level. The atlas
depth check of …d5 gave its best second turn, Ke2 + e5, 25% at 12,800, the
same value. So the result stands, but …d5 was not dived.

## The 14 lines

**Columns:**

- **W2\* (A → B):** White's best second turn, with its 3,200 value in A and
  its 12,800 value on fresh games in B.
- **A's pick at depth:** the 12,800 value of the move A ranked first. It is
  the unbiased depth check.
- **Black's best answer by C:** with the C value, and the next-best reply for
  contrast.
- **CR:** the fresh-game value of each Black candidate, with W/D/L. The
  lowest decides the line.

| Line | W2\* (A → B) | A's pick at depth | Black's best answers by C (next-best) | CR | Line value |
|---|---|---:|---|---|---:|
| e4–e5 …Nf6 | d4 + c4 (56% → 75%) | 75% (same move) | **…c5 0%** (next …e6 81%) | …c5 20.8% (3/0/12) | **20.8%** |
| f4–f5 …Nf6 | d3 + c4 (38% → 44%) | 28% | …e5 0%, …g5 0% (next …Na6 12%) | …e5 12.5% (2/0/13); …g5 33% | 12.5% |
| f4–f5 …Nh6 | d4–d5 (31% → 25%) | 0% | …Nf5 6% (next …f6 25%) | …Nf5 8.3% (0/3/12) | **8.3%** |
| c4 + f3 …c6 | Kf2, Kg3 (50% → 56%) | 6% | …Na6 12% (next …Nh6 25%) | …Na6 12.5% (1/1/13) | 12.5% |
| c4 + f3 …Na6 | Kf2 + f4 (50% → 62%) | 28% | …d6 0% (next …e6 12%) | …d6 25.0% (4/0/11) | 25.0% |
| c4 + f3 …c5 | Kd1 + f4 (50% → 25%) | 25% (same move) | …f5 12%, …a6 19% | …a6 10.4% (0/5/10); …f5 29% | 10.4% |
| c4 + f3 …d5 | Kd1 + c5 (25% → 50%) | 0% | …a5, …b5, …f5 0% | …f5 0.0% (0/0/15); …b5 15%; …a5 17% | **0.0%** |
| c4 + d4 …d5 | f4 + Kf2 (34% → 50%) | 25% | …f6 25% (next …dxc4 44%) | …f6 29.2% (5/2/8) | 29.2% |
| c4 + d4 …d6 | d5 + c5 (56% → 38%) | 38% (same move) | …Na6 12% (next …dxc5 62%) | …Na6 12.5% (0/3/12) | **12.5%** |
| c4 + d4 …e5 | Kd2 + d5 (62% → 56%) | 56% (same move) | …c5, …Ne7, …Nf6 25% | …c5 29.2% (3/1/11); others 33% | 29.2% |
| e4 + d4 …f5 | d5 + c4 (41% → 38%) | 19% | …e6 25% (next …fxe4 38%) | …e6 25.0% (0/9/6) | **25.0%** |
| d4 + f4 …e6 | Kf2 + f3 (47% → 56%) | 56% (same move) | …d5 0%, …Nh6 0% | …Nh6 4.2% (0/1/14); …d5 21% | 4.2% |
| d4 + f4 …d6 | c4–c5 (38% → 38%) | 12.5% | …Na6 0% (next …f5 25%) | …Na6 2.1% (0/1/14) | **2.1%** |
| d4 + f4 …d5 | Kf2 + c4 (22% → 50%) | 25% | …dxc4 31%, …f6 38% | …f6 20.8% (3/0/12); …dxc4 25% | 20.8% |

The lines that decide their first turn are in **bold**.

## What the dive shows

**1. White's best tries mostly fail one move later.**

- On fresh games at 12,800, White's best second turns score 25–75% (B).
- Black's best answer brings each line back to 0–29% (CR).
- Across the 14 lines:
  - mean B value of W2\*: 47.3%;
  - mean line value after Black's answer: 15.2%.

**2. Black often has a single only move.**

- After e4–e5 …Nf6 d4 + c4, …c5 holds White to 20.8%. Every other Black
  reply lets White score 81% or more in C.
- After c4 + d4 …d6 d5 + c5, …Na6 holds White to 12.5%; the next best,
  …dxc5, gives 62%.
- After d4 + f4 …d6 c4–c5, …Na6 holds White to 2.1%; the next best gives 25%.
- These are the positions where a Black model that has not learned the one
  move loses.

**3. Stage A's first choice is a reasonable guide at depth.**

- A's top-ranked second turn averaged 29.9% at 12,800, against 39.1% at
  3,200 for the top four. Part of that drop is selection, part depth.
- In 5 of 14 lines, A's pick was also the best at depth.

**4. No family of White second turns stands out.**

| Family | Positions | Mean | Share ≥30% |
|---|---:|---:|---:|
| Two pawns | 201 | 10.0% | 12.9% |
| One pawn twice | 94 | 8.6% | 12.8% |
| Pawn + king | 336 | 8.8% | 8.0% |
| King walk | 145 | 6.8% | 4.8% |

- The W2\* winners span every family: 6 pawn + king, 4 two pawns,
  2 one pawn twice, 2 king walks.

## Against the atlas

| First turn | Atlas minimax (3,200) | Atlas depth check (12,800) | Dive (two turns, Black's answer) |
|---|---:|---:|---:|
| e4–e5 | 25.0% | 25% | 20.8% |
| f4–f5 | 22.5% | 12.5% | 8.3% |
| c4 + f3 | 17.5% | 31% | 0.0% |
| c4 + d4 | 12.5% | 37.5% | 12.5% |
| e4 + d4 | 0.0% | 25% | 25.0% |
| d4 + f4 | 5.0% | 25% | 2.1% |

- The atlas depth check stopped at White's second turn.
- Adding Black's answer lowers every first turn except e4 + d4, where …e6
  holds White to a 25% value.

## Conclusions

**For the game's value:**

- Against today's best models, White's best first turn (e4 + d4 or e4–e5) is
  worth about 20–25% two turns deep. Every other first turn is worth ≤12.5%.
- Black's practical win stands, now including Black's answers to White's best
  second turns.
- The best White line, e4 + d4 …f5 d5 + c4 …e6, is a draw more often than a
  loss (9/15 draws).
- This is still the models' best play, not a proof.

**For training, the book lines that matter:**

| First turn | Line | Why |
|---|---|---|
| e4 + d4 | …f5 d5 + c4 …e6 | White's drawing line |
| e4–e5 | …Nf6 d4 + c4 …c5 | Black's only move |
| c4 + d4 | …d6 d5 + c5 …Na6 | Black's only move |
| c4 + d4 | …d5 f4 + Kf2 …f6; …e5 Kd2 + d5 …c5 | Lines near 29% |
| d4 + f4 | …d6 c4–c5 …Na6 | Black's only move |
| d4 + f4 | …e6 Kf2 + f3 …Nh6 | |
| f4–f5 | …Nh6 d4–d5 …Nf5; …Nf6 d3 + c4 …e5 | |
| c4 + f3 | …d5 Kd1 + c5 …f5 | |

- Teaching these as book starts gives White its best tries and Black its only
  moves in the positions that decide games.
- Black's answers were measured at 3,200. A 12,800 check of the deciding
  Black answers, the six bold lines, would cost about 2 h.
