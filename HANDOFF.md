# HANDOFF — 2026-09-05

**For the next agent or developer.** Read this first, then `CONTEXT.md` §2
(durable reference) and §5 (the laws that will bite you). `REPORT.md` is the
evidence log; `DIRECTIVE.md` is a completed scope record, not a plan.

---

## 1. Where things stand

| | |
|---|---|
| **Release** | `models/bootstrap_v24/best_value_net.pt` — generation 30, promoted 2026-08-22 |
| **Strongest measured** | `gen42` (`models/candidates/bootstrap_main_gen_0042/screen_nominee.pt`) — leads **both** Elo ladders |
| **Newest model** | `gen44` (`models/candidates/bootstrap_main_gen_0044/best_value_net.pt`), trained 2026-09-05, gate in flight |
| **Gate instrument** | **free play, deduped** — `tools/gate_free.py`. The book gate (`tools/gate.py`) still exists and is unchanged |
| **`gate.BAR`** | `vs_v24`. Read it, never infer it |
| **Replay window** | **8** (`--replay-generations 8`) so replay spans gen36+ only |
| **Anchor corpus** | **dropped** (`--anchor-data none`) |
| **Suite** | 741 passing |
| **Branch** | `main`. Commits in the owner's name, **no `Co-Authored-By` trailer**. Never push unless asked |

Never stage `src/play.ipynb` — its outputs are session noise.

---

## 2. The result that reframes everything: the 2026-09-04 round robin

Ten models, 45 pairings, 90 legs, **36,000 games**, 46 hours. Every pairing
played **twice**: a book leg on one identical opening block, and a free leg
deduped on opening state. Ratings by weighted least squares on the logit scale,
verified against synthetic data before use (recovers known ratings to 0.85 Elo).

Anchored so **v21 = 1000** (v21→v22 is +106 book, +88 free, both measured):

| model | book | free |
|---|---:|---:|
| gen42 | **1353** | **1781** |
| gen41 | 1342 | 1750 |
| gen40 | 1325 | 1741 |
| gen36 | 1333 | 1713 |
| gen38 | 1343 | 1699 |
| v24 | 1325 | 1564 |
| gen33 | 1335 | 1558 |
| gen26 | 1314 | 1535 |
| v23 | 1141 | 1254 |
| v22 | 1106 | 1088 |
| v21 | 1000 | 1000 |

Human play is **below v21** by the owner's account, with no measured match, so
it cannot be placed on either scale — only bounded.

**The tier split is the finding.** On free, gen36/38/40/41/42 sit **135–246 Elo
above** v24/gen33/gen26 — more than 6 SE. Book compresses that same structure
into 8–28 Elo, inside its own noise. **Five consecutive generations were
recorded as failures by an instrument that could not see what they improved.**

gen36 is the sharpest case: it **failed** its book gate at 400 sims against
gen33, is **level** with gen33 at 3200 on a book, and beats it by **+164 Elo**
on free.

**Within a tier nothing is separated.** The top five span 20 book Elo against
8.7 SEs. "The top five are tied" is the conclusive answer, not a failed one.

---

## 3. Owner decisions now in force

1. **Free play decides.** Book is retained for continuity, not for verdicts.
2. **Gates run on free play**, deduped, stopping on unique games or a time
   budget. `tools/gate_free.py`.
3. **Corpus is generation self-play only.** No human games, no v19-era anchor,
   no outside data. Replay reaches back only to gen36.
4. Promotion remains the owner's call and wants a playtest.

---

## 4. Measurement rules that will bite you

These are not style preferences. Each cost real time to learn.

| rule | evidence |
|---|---|
| **Never compare a per-colour score to 0.50** | Block colour-bias is ±0.056. Read against a measured par |
| **Free-play par is not 0.50 and is model-specific** | v24 scores White **0.8717** against itself; gen33 0.7933; gen38 0.5833 |
| **A book match carries ~20 Elo of block noise the SE hides** | Same pairing, different blocks: 19.3 and 23.7 Elo apart |
| **Disjoint blocks for independent samples; MATCHED blocks for comparisons** | Giving every cell its own block breaks the comparison you built the run for. Cost this project two runs |
| **Free play must be deduped** | After the opening prefix play is deterministic; two games sharing an opening state *are* the same game. 40% dupes same-era, **68–76%** within the top cohort |
| **One book line is n=1** | ~45% of per-line verdicts flip on resampling. Use `--book-temp-plies`; quote aggregates, not cells |
| **Depth changes values** | Line values reproduce at r≈0.83–0.90 within a depth, r≈0.36–0.67 across. A score at one sim count is a statement about that sim count |
| **Free play is non-transitive** | Round-robin RMS residual **73.5 Elo** free vs 12.6 book. Good tier detector, poor ordering device |
| **Never chain Elo** | v24 and gen26 sit 55 Elo apart via v22 and are **level** head-to-head. Three anchored claims were overturned by direct play |
| **Existence is not completion** | Killed runs leave truncated artifacts that resume logic accepted as done. Validate, don't `stat` |

---

## 5. Tooling added since the last handoff

| tool / flag | what it does |
|---|---|
| `tools/gate_free.py` | free-play gate: par leg (bar vs itself, cached), bar leg, confirmation replay. Stops on unique games or budget |
| `tools/match.py --book-temp-plies N` | samples N plies after each book position so a repeated entry yields *different* games — the only way to error-bar a single line. Also splits the pair seed when sampling |
| `tools/match.py --game-log` | per-game JSONL incl. the opening record, which is the exact dedup key for free play |
| `src/iterate.py --anchor-data none` | drops the v19-era anchor so the corpus is generation self-play only |
| `tools/model_report.py` | self/anchor × book/free × named predecessors × sim levels |
| `tools/export_lines.py` | replays named lines to scrubbable HTML. **Two engines, one per colour** — native MCTS reuses its tree, and sharing one leaks White's tree into Black's search |

---

## 6. Open questions

- **gen44's gate** — in flight at time of writing. `benchmarks/gate_free_gen44.json`
- **Does dropping the anchor cause overfitting?** gen44's val loss rose from
  epoch ~15 while train fell. Plausible consequence of a smaller, more
  homogeneous corpus. The control is the same recipe with the anchor restored —
  one variable, already isolated. Not yet run
- **Top-five ordering** unresolved and probably unresolvable at practical
  sample sizes. Deciding among them may need a criterion other than strength
- **The book gate's blind spot** — gen36 was rejected by it and belongs to the
  stronger tier. Any candidate rejected by a 400-sim book gate since gen33
  deserves re-examination
- **`iterations/gen_0043`** is an empty stub from a killed run; `iterate.py`
  treats directory existence as "generation taken", which is why gen43 was
  skipped and the model is gen44

---

## 7. Operating rules

- Long jobs via `py -3 tools/runs.py start --name X -- ...`, never blocking
- **Never run concurrent worker jobs.** 3×8 workers froze this box; a
  DPC_WATCHDOG_VIOLATION (0x133) hit on 2026-08-31 during a 12-worker match.
  Match workers are now **8**
- Never build Python scripts in bash heredocs — `\n` mangling has cost time
  twice
- Never weaken a gate threshold to let a recipe through
- Estimate from the running job; never extrapolate a rate across workloads
