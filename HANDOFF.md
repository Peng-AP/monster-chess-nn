# HANDOFF — 2026-08-18

**For the next agent.** Read this, then `CONTEXT.md` §2 (durable reference,
current standing), then `REPORT.md` §§50–52 (the last day's evidence).
`DIRECTIVE.md` is a completed scope record, not a plan.

---

## 1. Where things stand

| | |
|---|---|
| **Release** | `models/bootstrap_v23/best_value_net.pt` (owner promoted 2026-08-17) |
| **Strongest on record** | `models/candidates/bootstrap_main_gen_0023/selected_epoch_007.pt` |
| **Working bar for gen25** | gen23 (above). Pass it as `--bar-model` / `--incumbent`. |
| **`gate.BAR`** | still `vs_v23` — the *release*. Read it, never infer it. |
| **Replay window** | **12** (`--replay-generations 12`). Not the default 4. |
| **Suite** | 741 passing |
| **Branch** | `main`. Commits are in the owner's name with **no `Co-Authored-By` trailer**. |

gen23 is roughly **+120 Elo above the v23 release** (+251 vs v22 directly,
where v23 is +130). By the project's own precedent — v21→v22 ≈ +139, v22→v23 =
+130 — that is a version's worth. **It has not been promoted; that is the
owner's call and it is open.**

## 2. What happened, in one paragraph

The chain flatlined at gen18 (an exact 327W/327L/146D tie). The cause was data:
the corpus had shrunk 936k → 445k rows because the 2026-08-16 threefold-
repetition rule cut records per game 93.3 → 59.8 and nothing compensated.
Widening the replay window 4 → 8 → 12 restarted it. gen19 +53, gen21 +14,
gen22 +50, gen23 +30; gen20 and gen24 failed. Validation improved six
generations running and broke below 2.0 for the first time.

## 3. Open owner decisions

1. **Promote gen23?** ~+120 over the release. If yes, follow
   `models/bootstrap_v23/promotion_manifest.json` as the template and move
   `gate.BAR`, `BAR_MODEL`, `NUMBERED_INCUMBENT`, `CONFIRM_LEG`, the leg names,
   `src/config.INCUMBENT_MODEL`, `src/iterate.DEFAULT_CHAMPION`,
   `tools/confirm_candidates.BAR`, `tools/tune_training.BAR`, and the two bar
   pins in `tests/`.
2. **Screen tie-breaks.** When finalists tie on aggregate within noise the
   ranking picks arbitrarily, and at gen22 it took a White-negative candidate
   over a balanced one on a 0.015 difference against 0.064 SE. Owner has seen
   this and chose to leave it; revisit only if asked.
3. **Strip the `Co-Authored-By` trailer from 45 older commits.** Owner approved
   stripping the ten from 2026-08-17 (done). The remaining 45 reach back to
   February and **are already in `origin/main`**, so removing them rewrites
   published history and needs a force-push. Not done deliberately.

## 4. What to do next — ranked

**a. Re-run the sims-regime question properly.** Gates run at 400 simulations,
where Black wins ~33% of self-play games. At the owner's 3,200 the same model
wins **6.2%** as Black. Models are being *selected* under conditions materially
different from the ones they are *played* under, and "better at 400" may not
mean "better at 3,200". A gate leg at 3,200 would cost roughly 8× a 400-sim
leg, so the cheap version is a one-off: take gen17, gen19 and gen23, play them
against each other at 3,200 on a shared block, and see whether the 400-sim
ordering survives. **If it does not, that invalidates the selection criterion,
not just one measurement.**

**b. Push the replay window further.** 4 → 8 gave +53, 8 → 12 gave +14 then
+50. Window 16 is untried; `autochain.py` escalates to it automatically on two
failures. The corpus stabilises near 950k at window 12 because the window
slides rather than accumulates.

**c. The 17-channel encoding is built and unused.** `config.TENSOR_SHAPE` is
`(8,8,17)` with `WHITE_PAWN_PROGRESS_LAYER` and `BLACK_PAWN_PROGRESS_LAYER`,
but every stored position is **15** channels — the legacy layout, whose channel
14 is a *White-only* pawn-advancement feature with no Black counterpart.
Requires re-encoding the corpus. Designed, built, never measured.

**d. Do not bother with capacity.** The network overfits from epoch 5 on the
current corpus (train 1.95 → 1.55 while val 2.05 → 2.21). It can already
memorise what it is given; more parameters would memorise sooner. Revisit only
if the corpus grows and that stops being true.

## 5. Traps that cost real time today

**Never quote a per-colour score without its block's baseline.** Block colour
bias runs to ±0.056. A model played against *itself* — true value 0.5000 by
construction — scored White **0.4437** on one block and **0.3000** on another
from the same book. Read against 0.4437 instead of 0.50, gen17's "alarming"
0.4338 was *at par*. This misled three separate readings in one afternoon. The
gate now plays a `bar_selfmatch` leg on the bar leg's own block and reports
each binding leg against it (added 2026-08-18, diagnostic only — it cannot
change a verdict). For cross-generation comparisons, put every model on the
**identical** block; `benchmarks/ctl_*_20260818.json` uses v27 offset 1620.

**Never quote a chained sum of gate passes as a strength claim.** Chaining
oversold by 15.7% on one five-step chain and **27%** on the next four-step one.
It is not a constant. Measure the endpoints directly.

**The screen shortlists, it never predicts.** At 200 games its calibrated delta
has SE ≈ 0.064. It has read high (gen18, gen20, gen22) and low (gen23). A
+0.04 or +0.08 delta is noise; treat any screen number as a shortlist and let
the 800-game gate decide.

**High-sim game counts are not independent trials.** At 3,200 sims a strong
net's visit counts are peaked enough that temperature sampling returns the top
move nearly always, so 48 games held five distinct openings and outcome tracked
the opening family almost perfectly. Deep search buys strength and spends
variety.

**Verify tools obey current rules before trusting their output.**
`tools/export_selfplay_replays.py` had no repetition tracking and was playing
under pre-2026-08-16 rules; the tell was every drawn game measuring exactly 225
plies. Fixed, and `tests/test_selfplay_export_rules.py` pins it. Both of the
day's measurement defects were surfaced by the owner asking a sceptical
question, not by review.

**Operational.** Run long jobs via `tools/runs.py start`; never block a call on
a match. **Never run concurrent worker jobs** — 3×8 workers froze this machine
on 2026-08-16. Never commit `src/play.ipynb` outputs. Estimate from the running
job's own progress line, never by extrapolating a rate across workloads.

## 6. Scripts worth reusing

Driver scripts live in the session scratchpad, not the repo. The durable ones:

- `tools/gate.py --protocol full` — the binding gate, now with the calibration leg
- `tools/checkpoint_screen.py` — shortlist 8 checkpoints to 4
- `tools/draw_anatomy.py` — outcome × game length, with the winning-window curve
- `tools/review_game.py` — per-move swing on a saved human game
- `tools/selfplay_examples.py` — N games of each outcome, exported to HTML
- `tools/make_book.py --allow-short --oversample 2.2` — books; raise oversample,
  duplicate rates climb with generation

An autonomous chain driver (`autochain.py`) ran gen20–gen24 unattended: ensure
book capacity, generate + train, screen, gate, advance the bar on PASS, escalate
the replay window on FAIL, stop after two failures at the widest. Its state file
pattern (resume rather than repeat) is worth keeping.

## 7. Artifact

Self-play showcase, nine real games at 3,200 sims with scrubbable boards:
<https://claude.ai/code/artifact/b4c03519-b5e7-4060-9e8c-fe5ea3e5c256>
