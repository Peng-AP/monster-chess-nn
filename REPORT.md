# RUN REPORT — 2026-08-03: the v20 campaign closes, and the scoreboard was wrong

**Headline.** The v20 campaign is decided and it decided negatively: both arms
rejected, and the last two levers on the finishing problem tested dead. The
larger result is not about v20 at all — a scoring rule you called out by hand
turned out to have been **paying Black for shuffling and charging White for
surviving**, through the whole v19 ladder. No number measured before today is
comparable to one measured after it.

Every claim below cites its artifact in `benchmarks/`. Test suite: **298 pass**.

---

## 1. What you changed, and what it cost

> *"A win by time shouldn't be counted the same as win by capturing the king."*

Implemented in `src/benchmark.py::summarize_side`: only `result >= 1` is a win,
only `result <= -1` a loss, and the ±0.5 move-limit relabels score as **draws**
— symmetrically, so the rule cannot favour a side. `time_leaning_wins` /
`time_leaning_losses` now ride along in every summary so "ahead at the cap"
stays visible instead of vanishing into the draw count. Pinned by
`tests/test_win_requires_capture.py`. The ±0.5 **training label** is untouched
— it carries gradient and is deliberate (CONTEXT law 3). This is only about
what the *gate* calls a win.

It immediately reversed a verdict.

### v20w: PASS at 08:12, FAIL at 10:52, same 40 games, same seed

| leg vs v19 | old rule | new rule |
|---|---|---|
| as White | 0.80 — 16W / 4L / **0D** | **0.90** |
| as Black | 0.50 — 10W / 10L / **0D** | **0.30** ← floor is 0.40 |
| aggregate | 0.65 | 0.60 |
| verdict | PASS, confirmed | **FAIL** |

`gate_v20w_20260803_081212.json` → `gate_v20w_20260803_105202.json`. The ramp
leg fails too (Black 0.375). Across legs: White 0.92, Black 0.38.

Read the ply counts and the mechanism is plain. v20w's Black won at a mean of
**196.5 plies** and lost at 36.8 — it did not win those games, it declined to
lose them for two hundred plies. And its White *"losses"* came in at
`mean_plies_when_lost: 225.0` — **exactly** the 150-turn cap, all four of them.
In twenty games, v19's Black never once captured its king.

### The bias had a direction

That is why v20w's White leg went *up* under the stricter rule. Cap games in
this variant are overwhelmingly Black-ahead-but-stuck, the cap relabels by
heuristic sign, so the old rule systematically converted Black's failures to
finish into Black wins and White's successful survival into White losses.

This is the likely answer to a question that has been open all campaign: **why
your read at the board never showed up in the metrics.** You said "Black
conversion is a big problem." The metric was paying Black for exactly that
failure. It ran through the entire v19 ladder, v19's own promotion included.

**Debt incurred:** no pre-2026-08-03 gate or match number may be compared to a
post-change one. Recorded in `CONTEXT.md` (standing rules) and `DIRECTIVE.md`
§5a. The incumbent needs re-measuring under the new rule before any future
campaign can claim a gain against it — queued as the next item, ~1h, not yet
launched.

---

## 2. The v20 campaign: both arms rejected

L3's premise was that self-play records Black's failures as ground truth, so
generate games where Black *does* convert — asymmetric search, Black at 1600
sims against White at 400 — and train on those. 240 such games from both decks,
two corpora, two trainings, both gated:

| arm | Black leg vs v19 | verdict |
|---|---|---|
| `v20` | 0.10 | FAIL — `gate_v20_20260803_073324.json` |
| `v20w` | 0.30 | FAIL — `gate_v20w_20260803_105202.json` |

Both now in `models/rejected/`. The premise was sound, the machinery worked,
the games were what they were supposed to be, and neither arm produced a Black
that converts against v19. That is a clean negative, not an inconclusive one.

`v20w`'s conversion probe says what it actually learned
(`promotion_defense_outcomes_ppc_v20w_20260803_090152.json`, Black@400 vs v19's
White@400, against v19's own numbers at the identical setting):

| | v19 | v20w |
|---|---|---|
| "black win rate" | 0.28 | **0.46** |
| **true king captures** | 0.15 | **0.13** |
| dominant, unfinished | 13 | **33** |

It got substantially better at reaching and holding winning positions and
**very slightly worse at finishing them**. Under the old scoring that reads as
a large gain. Under yours it reads as the disease.

---

## 3. The finishing problem: search reaches, nothing cashes

The PPC curve (100 positions, Black's sims varied against v19's White@400):

| Black sims | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|
| true king captures | 0.09 | 0.15 | 0.20 | **0.30** |
| dominant, unfinished | 12 | 13 | 28 | **40** |
| White wins | 0.79 | 0.72 | 0.52 | **0.29** |

Search buys real conversions — captures triple — but look at the unfinished
column climbing just as fast. Deeper search finds the won position and then
cannot end the game.

**So I tested the last standing explanation: that those games just needed more
moves.** Same config as the 800 cell, cap raised 150 → 400
(`promotion_defense_outcomes_capraise_20260803_103416.json`):

| Black@800 vs White@400 | cap 150 | cap 400 |
|---|---|---|
| true king captures | 0.20 | **0.20** |
| dominant, unfinished | 28 | 27 |
| mean plies | 88.3 | **197.0** |

Given **2.7× the moves**, Black converts identically. The unfinished games are
not conversions awaiting time. Black genuinely cannot finish them.

*(Correction to what I told you verbally: this ran Black@800 vs White@400, not
both sides at 800. The baseline it is compared against is the same
configuration, so the comparison and the conclusion stand.)*

---

## 4. Where that leaves the lever board — empty but for you

| lever | status |
|---|---|
| **L1** extend the scripted oracle to king+pawns | **struck by you** — complexity skyrockets |
| **L2** mask the class's value labels | **retired** — the PPC curve showed the labels aren't the binding problem; they accurately describe 400-sim play |
| **L3** asymmetric generation | **rejected today** — both arms failed the gate |
| *turn cap* | **dead** — 2.7× the moves, identical captures |
| **L4** your own games | **open, and now the only one** |

The honest statement: Black's residual failure is a **technique** gap. More
search finds the won position; more time does not cash it; more
self-generated data does not teach it. The only two things that have ever
supplied that technique are the scripted oracle on the single class it already
solves — which is why the bare-king class sits at 88% Black in the corpus
against the king+pawns class's 36% — and your own play. You convert this class
100%. The model converts it 0.30 at 1600 sims.

L4 has the best per-record effect on record: your 27 uncorpused games moved the
deciding gate leg **+0.150**, the largest measured effect of anything tried
(CONTEXT law 11, "data beats architecture"). It is no longer the flywheel
running alongside the real work. It is the work.

This matters because of *what* class it is: **1,613 of 2,624 games (61.5%)** in
`combined_v19_K` pass through the post-promotion class, and Black wins 38.6% of
them. Nearly two-thirds of all games are decided in a class you win every time.

---

## 5. Recommended next moves

1. **Re-measure v19 and v19_B under captures-only** before anything else is
   built on top of them. ~1h, not launched — say the word.
2. **A play session as Black from `postpromo_starts_v1.jsonl`.** This is the
   highest-value thing available and it needs you specifically: it is a live
   demonstration of the exact technique L1 would have scripted and L3 failed to
   approximate. Harvest via `tools/add_owner_games.py`.
3. **Decide the bar** (`v19` vs `v19_B`) — open since 2026-08-02, and the
   re-measurement in (1) is the natural moment to settle it.

Still open and still yours: §7.4's hand-corrected label precedent, and the
`data/raw/combined_v16` copy from the CPU box.

---

## 6. Housekeeping

- `v20w` moved to `models/rejected/`. `v20` was already there.
- `tests/test_recovery_foundations.py` had three stale failures — it unpacked 5
  values from `_convert_games_to_arrays`, whose arity grew to 6 when D1 added
  value weights. Fixed to index rather than positionally unpack. **298 pass.**
- `src/play.ipynb` was swept into a commit by a `git add -A`; backed out. Your
  notebook is modified-and-uncommitted, as you keep it.
- Nothing pushed.
- The Gmail and Google Calendar connectors need authorizing from your claude.ai
  connector settings before I can use them; I can't run that flow from here.

*Commits: `2f2b472` (scoring rule), `03b03e8` (turn cap), `a7ad6e1` (v20w
rejected).*
