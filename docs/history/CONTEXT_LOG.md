# CONTEXT.md log sections (moved October 6, 2026)

Moved verbatim from the root `CONTEXT.md` during the October 6 cleanup
(`docs/history/CLEANUP_20261006.md`): the dated status preamble and §2
"Where the project stands". Every "current" statement here is historical;
the current state is in the root `HANDOFF.md`.

---

## Preamble (status notes through September 25)


**September 25, later: v28 promoted** (gen50 epoch14 + calibrated value head,
owner-approved); gate bar and champion pointer moved to v28. Root reorganized:
docs in `docs/`, finished drivers in `campaigns/`, snapshot commit `b46ce1c`.
Next step (owner focus, shippable engine): `docs/plans/SHIPPABLE_ENGINE_PLAN.md`.

**September 25 current:** see [HANDOFF_20260925.md](HANDOFF.md) for the
consolidated state and proposed next steps. Value calibration completed
September 17 at 09:16:27: 3,696 games, continuation nominee 58.4375%/800 vs
unchanged gen50 epoch14 at 3,200 simulations, but 49.375%/160 at 12,800.
Ordinary-budget gain, not a demonstrated general/all-budget upgrade. No promotion;
public v27 and gen50's epoch14 selection remain unchanged. No active managed
campaign or queued follow-up found on September 25. See
`VALUE_CALIBRATION_RESULTS.md`. Cleanup preserved models/data/evidence and
archived completed logs; `CLEANUP_20260925.md` records the exact scope.

**September17 01:16:** checkpoint recovery COMPLETE3,568games+72probes.
Epoch15nominee passesgen49 gate73.9375%,Black97.875%,White50%; at12,800
White26.25%,Black99.375%. No balanced recovery, no overwrite/promotion.
See `GEN50_RECOVERY_RESULTS.md`; currentgen50selected remainsepoch14.
Value/search crossover explains ordinary-budgetc5 attraction better than root
policy, but no global value-head transplant is justified. No further run queued.

**September16 evening:** gen50epoch14 completed, gatePASS/75%vsgen49 over800,
especiallyBlack93.25%; White regressions vsv27/B2 prevent an unqualified broad
upgrade claim. See `GEN50_RESULTS.md`. Owner authorized checkpoint recovery:
`GEN50_RECOVERY_PLAN.md`, `run_gen50_recovery.py`, rehearsal complete983tests+
3subtests,172games/24probes,26receipts/resume verified. Production launched19:07
as `gen50_checkpoint_recovery`;
six-model matched-root/screens plus isolatedpolicy/value crossover and fresh
confirmation. No new training, search-default changes, overwrite or promotion.

**September16:** corrected counterplay study plus extension completed1,999games/
108probes. Deeper gen49 vsgen48:78.4375%, vsB2:99.375%; deep gen49 self
14White/36Black/110draw. See latest HANDOFF. Owner authorized gen50 next:
same mainline recipe with gen49teacher,12,800fork/reanalysis targets, unchanged
architecture/training. `GEN50_PLAN.md`, `run_gen50.py`; full rehearsal passed
982tests+3subtests and complete tiny generation/training/36game checks.
Production launched00:50 as `gen50_deep_targets`;2,280independent postselection
games across3,200/12,800 are chained after selection. No promotion.
Older dated active-run claims below are historical.

**September15 02:57 correction:** original counterplay study stopped/preserved:
equal-model conditional games shared one tree across colors. Corrected to match
the normal harness's separate per-color trees; added regression coverage. New
fully rehearsed run will use `benchmarks/mainline_counterplay_20260915_v2` and
managed `mainline_counterplay_v2`. Corrected production launched02:57 after978tests+
3subtests,169-game/31-probe rehearsal, and exact117-ply parity against the normal
benchmark. No gen49 weights or ordinary match engine changed; prior gen49 results stand.
Original study evidence is not pooled into v2. See the latest HANDOFF entry.

**September15 01:41 current work:** launched the authorized eight-hour mainline
counterplay study, `MAINLINE_COUNTERPLAY_PLAN.md`. Gen49 has finished: epoch7
scored94%/800games vsgen48,83.75%vsB2,98.25%vsv27; self14White/136Black/50draw.
See `GEN49_RESULTS.md`: large measured improvement, concentrated opening
advantages; B2 Black-score decline is mainly a defense-selection shift, not
established conversion deterioration. Owner human playtest is positive.
New work freezes all weights and studies actual ...e5/...d5 counterplay, the
dominant drawing line and search scaling. No gen50, architecture/rule change,
promotion or destructive operation. Managed `mainline_counterplay`, log
`logs/mainline_counterplay.log`; evidence `benchmarks/mainline_counterplay_20260915`.
977tests+3subtests and169-game/31-probe rehearsal passed, including four full
204,800-simulation probes. Production1,423games+108probes chained, no score-based
stopping; maximum8game workers/4root-probe workers, one heavy stage at a time.
Earlier dated status below is history where it conflicts with this paragraph.

**September14 02:33 current work:** launched `GEN49_PLAN.md` / `tools/start_gen49.py`:
independent normal-start gen48 tests first, then a mainline-only new gen49 data
increment (2,800 selfplay + 400 deep continuations), unchanged CNN/training,
and independent free-play gates/held-out tests. Managed `gen49_mainline`, log
`logs/gen49_mainline.log`. 954 tests + 3 subtests and full 28-game/48-test-game
rehearsal passed; all branches continue after measured failure. Public v27
is unchanged. No forced-prefix expansion, automatic promotion or deletion.
**Evaluation correction:** the gen48 H2H/B2/self measurements below were ALL
book-based. Flat B2 book transfer is not evidence of flat normal-start transfer.
The standing owner policy makes normal-start free play primary.

**04:36 measured correction:** gen48 independent normal-start H2H vsgen47
89.25%/92.25%, combined90.75%; held-out B2 76.25% vsgen47's39.25%, with gains
on BOTH colors. Gen48 selfplay82White/67Black/51draw, White score53.75%.
All1,200games audited; `production/gen48_free_results.json` under the gen49
research directory contains evidence. Gen49 generation is underway. The
following old book-only result is historical, not the current strength verdict.

**Current result (September14):** gen48 GPU data-recipe experiment completed.
Fresh H2H vsgen47 scored54.69% and53.13%, but held-out B2 transfer was flat and
selfplay White score unchanged. No demonstrated broad both-color upgrade or
promotion. See `GPU48_RESULTS.md` and the top of `HANDOFF.md`; no follow-up is
queued. CPU strength work remains paused, not deleted. Public release remains
v27/gen46; gen47 is the frozen comparison bar.
The older directive/status prose below is historical where dates conflict.

**Durable rules, vocabulary, laws, and historical operational knowledge.**
Current state and next steps are consolidated in `HANDOFF_20260925.md`;
`HANDOFF.md` remains the chronological operational record. `DIRECTIVE.md` is
the completed August rewrite scope, not a live campaign. `README.md` describes
general usage. September research includes untracked source and evidence: do
not assume concluded drivers/reports are backed up in git history. Date-check
historical claims below against the relevant artifacts in `benchmarks/`.

---


---

## 2. Where the project stands

> **September 12 CPU-search experiment completed.** The optional incremental
> evaluator saved another6.04% fixed-depth elapsed time beyond existing search
> optimizations, but timed development tied baseline. Fresh CPU2s /CPU8s versus
> fixedGPUgen47@2s scored42.19% /46.88% (32games each); White28.13% unchanged,
> Black56.25% /65.63%. The paired gain is inconclusive.160 real games passed
> replay audit; no release promotion or new model training. No jobs remain
> queued. Details and the recommended next block: `SEARCH_CPU_SCALING_RESULTS.md`.

> **September 7 release update.** Owner authorized three milestone releases:
> gen42 = **v25**, gen45 = **v26**, gen46 = **v27**. Checkpoints are immutable
> copies with promotion manifests; v27 is selected by the bootstrap pointer
> and the legacy gate bar. Gen46 passed both sampled confirmations against
> gen45; broader diagnostics remain separate. Gen47 is queued with stateful
> continuation/reanalysis, family-isolated splits, and 20% older-opponent
> games. See `GEN47_RUN.md` and the top of `HANDOFF.md`; older dated standings
> below are historical, not current. No thresholds or old verdicts changed.

> **September 6 overnight update.** Gen44 epoch 9 completed its 3,200-simulation
> campaign: sampled H2H against gen42 was 67.39% then 67.23%, against gen41
> 77.22%, and against v24 71.69%. Its 562 self-games scored 56.41% White /
> 43.59% Black. Black improvement against gen42 is not established; the original
> v2 verdict stays INCONCLUSIVE for endpoint coverage. Release remains v24,
> with no working-pointer promotion. New generations default to the separately
> versioned fixed-sample v3 gate, with dedup diagnostics kept separate.
> See `REPORT.md` section 53, `SAMPLED_GATE_PROTOCOL.md`, and `HANDOFF.md`.
> The gen44 saved-epoch screen and next controlled teacher iteration follow;
> architecture, optimizer, training seed 3173, and generation-only learning
> recipe remain unchanged. Suite: 782 tests plus 3 subtests passed.

> **Prior standing, 2026-09-05.** Release is `models/bootstrap_v24`
> (generation 30, promoted 2026-08-22). Strongest measured is **gen42**, which
> leads both Elo ladders. Newest model is **gen44**. Gates now run on **free
> play** (`tools/gate_free.py`); the book gate is retained for continuity.
> Corpus is **generation self-play only** — the v19-era anchor is dropped
> (`--anchor-data none`) and replay reaches back only to gen36. Full picture in
> `HANDOFF.md`. September 5 implementation: `FREE_GATE_PROTOCOL.md` records the
> corrected equal-color gate, unseen-confirmation coverage, recoverable logs,
> and explicit generation-only pipeline defaults. Original gen44 evidence is
> promising but undercovered; no successor promotion follows from its old PASS.
>
> **The 2026-09-04 round robin is the reframing result.** 45 pairings, 90 legs,
> 36,000 games, both instruments on every pairing. Anchored at v21 = 1000:
>
> | model | book | free | | model | book | free |
> |---|---:|---:|---|---|---:|---:|
> | gen42 | 1353 | **1781** | | v24 | 1325 | 1564 |
> | gen41 | 1342 | 1750 | | gen33 | 1335 | 1558 |
> | gen40 | 1325 | 1741 | | gen26 | 1314 | 1535 |
> | gen36 | 1333 | 1713 | | v23 | 1141 | 1254 |
> | gen38 | 1343 | 1699 | | v22 | 1106 | 1088 |
>
> On free, gen36/38/40/41/42 sit **135–246 Elo above** v24/gen33/gen26 (6+ SE);
> book compresses that into 8–28 Elo, inside its own noise. **Five consecutive
> generations were recorded as failures by an instrument that could not see
> what they improved.** gen36 failed its book gate against gen33, is level with
> it at 3200 on a book, and beats it by +164 free Elo. Within a tier nothing is
> separated — the top five span 20 book Elo against 8.7 SEs.
>
> Human play is **below v21** and unmeasured, so it can be bounded but not
> placed on either scale.

### Historical assessment (2026-08-16, superseded above)

**The bootstrap loop works, and its defect was never the game.** Two apparatus
faults cost more than every recipe idea combined: fine-tuning from the champion
instead of training fresh (~31 Elo, 800 games, CI [0.5104, 0.5796]) and offline
checkpoint selection instead of play-based (~52 Elo, z=+4.17). A single 35-minute
from-scratch run beat the model five generations of fine-tuning had produced.

**The current bottleneck is CONVERSION, not strength and not the colour gap.**
Measured 2026-08-15 on full games from the true opening at 1600 sims: **every
draw is a cap draw** — all exactly 225 plies, while decisive games top out at
147, with nothing in between. In all 24 sampled capped games Black finished
ahead on material (mean **+26.4**, White on a **bare king in 21**) while both
sides shuffled: a mean of 12.6 distinct positions in the last 100 plies. The
exact solver proved **6 of those 24 games contained a forced king capture within
four Black moves** that search walked past. Complete negative searches prove
only the absence of a forced capture inside that horizon, not a fortress or
game-theoretical draw. These are unconverted wins, not fortresses. See
`REPORT.md` §30.

**Standing, 2026-08-22.** Release and bar are `models/bootstrap_v24`
(owner promoted generation 30). Release and strength bar are aligned. On the
full measurement standard at 1600 sims under a book -- the instrument to trust
-- v24 is **+191.6 Elo above v23** over 600 games (z=+18.6), +259.6 above v22,
and +20.9 above gen26. `benchmarks/tonight/report_gen30_1600.json`,
`models/bootstrap_v24/promotion_manifest.json`.

Two live caveats. **gen31 measured 16.8 Elo AHEAD of v24** on the same
instrument at 1.9 SE -- short of significance, ungated, and still outstanding;
1800 games would settle it. And the owner's playtest of gen30 was never
performed; the promotion rests on the measurements.

**Free play is not strength evidence.** gen30 against ITSELF splits -0.417
under a book and +0.698 free -- a 1.115 swing from the opening distribution
alone (effective_unique 1.00 vs 0.41). Between near-peers free play manufactures
a gap out of opening preference. Use the book instrument.

**Per-line book results are n=1.** Book play is deterministic, so every line in
the opening map, the Black-reply catalogue and the duo ladder came from one
game; resampling shows 42.7% of family lines and 46.0% of duo positions get a
different verdict. Use `tools/match.py --book-temp-plies`, quote aggregates not
cells. `benchmarks/robust/`.

**White recovered; §49's structural-ceiling conclusion was wrong.** Controlled
on identical openings, White went 0.6417 (gen17) -> 0.7017 (gen19) -> **0.7567
(gen23)**, **+0.1150 = 2.8 SE**, after being flat at +0.0034 through gen17. It
was starved like the rest of the loop and recovered on the same fix. The
owner's account of the *game* -- Black consolidates, White has no chances after,
so White lives on the fast attack -- stands and is reinforced by the self-play
result below.

**Self-play is White-dominated at the playing depth.** gen23 against itself:
at 400 sims Black wins **33%**; at 3,200 sims Black wins **6.2%** and White
takes **91.4%** of decisive games. Deeper search finds the attack before
consolidation. This explains the playtest saturation directly, and it means
**gates at 400 sims measure a different regime from the one the owner plays.**

**Two measurement defects, both surfaced by the owner asking rather than by
review.** `tools/export_selfplay_replays.py` played showcase games with no
repetition rule (every draw hit the 225-ply cap exactly) -- fixed, and the
corrected run happened to reproduce every number. And deep search destroys
opening variety: nine games held five openings, with outcome tracking the
opening family almost perfectly, because peaked visit counts make temperature
sampling return the top move. Treat high-sim game counts as far fewer
independent trials than they look.

**The loop was STARVED, and widening the replay window restarted it
(2026-08-18).** gen18 tied its bar exactly (0.5000 over 800 games) and failed.
The cause was data: the corpus had shrunk **936,324 -> 444,792** rows while
validation loss stayed flat and the net overfit from epoch 5 -- it could
already memorise what it was given. The trigger was our own 2026-08-16
repetition rule, which cut records per game 93.3 -> 59.8 with nothing added
back. gen19 reran gen18 with **one** change, `--replay-generations 4 -> 8`,
same seed and bar: 703,042 train rows, and **0.6038 / 0.5756 PASS confirmed**
against gen18's 0.5000 FAIL. That is **+53 Elo** (confirm leg) after four steps
of +30, +13, +13, 0. **White also moved for the first time** -- controlled, it
sits at **0.7017**, outside the 0.60-0.67 band that held for six generations.
`REPORT.md` §50. Book duplicate rates were rising too (v22 3-5%, gen17 39%), so
the games were getting self-similar as well as fewer.

**Superseded by the above, retained because the measurement was sound:** White
never improved at all through gen17; Black was the entire chain (2026-08-17).
Measured under control -- five generations, the identical 300 openings, the
identical opponent, 600 games each -- **White moved +0.0034 (0.1 SE) from gen11
to gen17 while Black moved +0.2167 (5.3 SE), monotone at every step.** White
wanders between 0.60 and 0.67 and ends where it began. The entire +139 Elo is
Black's, and the Elo steps are already bending (+35, +30, +13, +13): when Black
flattens the chain stops.

**Per-colour scores need a baseline, not just an opponent.** A block's colour
bias is +-0.056, larger than most effects being measured: gen16 against
*itself* scored White **0.4437** over 800 games on one block (true value 0.5000
by construction) and **0.3000** on a 40-game block of the same book. Read
against 0.4437 rather than 0.50, gen17's "alarming" gate White of 0.4338 is at
par. Either play the bar against itself on the same block, or do not quote a
per-colour number. Note also that on a Black-favouring block the absolute 0.40
White floor sits only 0.044 below neutral.

**White is unexploited headroom, not a diminishing return.** It is not weak
absolutely -- against v21b gen17 scores 0.7742 with White its *stronger* colour
-- it simply stopped improving around v22. Its failure mode is specific: White
wins by ply 30 or never, 95% of wins land by ply 60, and every failure is a
repetition after the attack stalls. White's draws are **not** material: it scores 0.6141 from
pawnless positions against 0.6293 with pawns, and 34 of gen16's 62 wins ended
with zero pawns, because the double-moving king hunts exactly as
`evaluation.py` always claimed. **White's ceiling is a search problem in the
attack.** `REPORT.md` §§47-48.

**The bootstrap chain now runs V22 -> gen11 -> gen12 -> gen13 -> gen14, and a
round robin proves it is a single strength scale.** Every step passed the full
high-power protocol -- an 800-game binding leg plus a confirm leg on a fresh
seed and a disjoint book block -- and all four are confirmed. The six pairs the
ladder never tested were then played at 600 games each: **every model beats
every predecessor and loses to every successor, with no cycle anywhere**. A
least-squares Elo fit over all ten pairs puts gen11 at +39, gen12 at +56,
gen13 at +89 and **gen14 at +119** above V22, largest residual 0.0243. The
apparent White erosion across three consecutive gate legs was an artifact of
comparing scores against three *different* bars: gen14 scores **0.6350 as White
against V22**, above gen11's 0.6075 on the same colour against the same
opponent. `REPORT.md` sections 41-44.

**The models in that chain are epoch snapshots, not `best_value_net.pt`.** The
gated checkpoints are `bootstrap_main_gen_0011/selected_epoch_008`,
`gen_0012/selected_epoch_007`, `gen_0013/selected_epoch_009` and
`gen_0014/selected_epoch_007`. `best_value_net.pt` in those directories is the
lowest-training-loss net and **no gate ever measured it**. Read the scored path
out of `benchmarks/gate_*.json`; never infer it from a filename.

**The first high-power gate rejected the first v23 attempt (2026-08-17), and
the run that produced it was starved.** The binding bar leg went 200 -> 800
games on the owner's instruction that a pass be high power and confirmed; no
threshold moved. `bootstrap_v23_gen_0001/selected_epoch_008` then failed both
legs on both conditions at 600 games (aggregate 0.4258 / 0.4375 against a >0.50
requirement; Black 0.3583 / 0.3800 against the 0.40 floor). The 200-game screen
had scored the same checkpoint at **0.515** -- optimistic by about **3 SE**.
**The screen must shortlist, never nominate** (`REPORT.md` §40).

**But that arm is not evidence about self-play.** It was built under
`--run-root iterations/v23`, a fresh lineage with nothing to accumulate, so it
trained on **331,844 rows** against Gen9's **936,324** -- 35% of the data that
produced V22. Its own generation was normal size; the entire shortfall was
missing replay. Any claim that self-play is exhausted does NOT follow from it.

**Replay provenance, measured 2026-08-17.** The Gen10 replay was 23% anchor
plus 36.5% generated by `fresh_start_v20` -- two versions stale -- because it
was built by the legacy `generation_driver.py`, which accumulates every
generation. The canonical `src/iterate.py` defaults to
`--replay-generations 4`, a sliding window, so a current run composes anchor +
the last four generations and carries no V20-era data at all.

The anchor itself is **not** old self-play: of ~111k raw rows it is roughly 35%
ps_monster (human, Elo >= 1600), 17% owner games, 29% v17-era self-play, the
rest curated focus sets. About half of it is human data, which is the one
source that historically moved BOTH colours, so it earns its place.

**Two rule/behaviour changes landed 2026-08-16 on the owner's instruction, and
they break comparability with every earlier number** — the same discontinuity
the 2026-08-03 captures-only correction caused. Nothing measured before today
is on the same footing as anything measured after.

- **The scripted oracle is dropped** (off by default; `MONSTER_SCRIPTED_MATE=1`
  reproduces the old behaviour). It is verified 11/12 on an authored
  K+Q+R+R-vs-bare-king deck, but in real generation it drove Black in 12 of 24
  games and converted **2**, drew 5, and **lost 5** — 17% against the deck's
  92% — while stamping every move at policy 1.0. Of 387 policy-1.0 records in
  that batch, **241 came from games that never converted**: about one record in
  nine teaching a shuffling move at certainty, which is a mechanical source of
  law 1a. A 120-game A/B without it ran **2.1x faster** with Black 0.317 ->
  0.375 (≈1 SE, so the case rests on cost and label quality, not that delta).
- **Threefold repetition is a draw** (on by default; `MONSTER_NO_REPETITION=1`
  disables, `MONSTER_REPETITION_N` retunes). **1.85x** on generation. It ends
  one game per 24 that would otherwise have been a Black king capture; that
  game was inspected and is not a conversion cut short — Black held eleven
  pieces against a bare king from record 40 and needed until record 129, so
  calling it drawn is a fair verdict. Match scoring barely moves because capped
  endings already scored as draws; what changes is the **training label**,
  -0.5 -> 0.0.

**Both changes together: 4.15x on generation** (4m13s -> 1m01s over 24 games,
seed 4242), mean records 93.3 -> 64.9, and **zero cap draws** — every
non-decisive game now ends by rule rather than by clock. They interact: with
the oracle's anti-repetition drift gone, repetition draws rose 6 -> 11. Outcomes
moved further than the clock did — Black wins 5 -> 6 and White wins 11 -> 7,
the four recovered games being ones the oracle had been losing. Black reads
0.375 -> 0.479, but that is descriptive only: n=24 and it is a different rule
set, so it is not comparable to any earlier figure.

**Tree reuse and the exact finisher are now on by default too** (2026-08-16;
`MONSTER_NO_REUSE=1` / `MONSTER_NO_FINISHER=1` disable). Reuse is -23% wall
clock where early stopping runs (matches, gates, screens) and -6% in
generation, at strength-null. The finisher was a null at 700 sims **only
because the oracle blocked its class**: with the oracle dropped it fires 18
times across 6 of 24 games and converts one extra win at **zero** time cost
(1m01s -> 1m02s), which is what the 145x native solver bought.

**Full stack against the pre-2026-08-16 baseline: generation 4.08x**
(4m13s -> 1m02s over 24 games), mean records 93.3 -> 59.8, Black wins 5 -> 7,
White wins 11 -> 7. A full generation cycle is estimated at **~79 min against
~156 min** (~2.0x), with reanalysis now the largest un-optimised block at
14.5 min -- and §33 found the teacher program it feeds to be a measured null,
so the open question there is whether to run it at all.

Deliberately still off, because each measured WORSE, not from oversight: the
inference server (1.30x slower search), the solver leaf probe (null at ~1900x
cost), `MONSTER_SOLVER`, moves-left utility, promotion-aware policy, SE blocks,
capture-WDL, side adapters, spatial value head, and `HybridEvaluator`.

**CUDA graphs were already on** (`MONSTER_CUDA_GRAPH`, default 1), captured one
graph per batch size because padding to a fixed width was 2.67x faster and
*changed move selection*.

**The exact solver is now native, 145x faster, and the finisher is free.**
`native/src/solver.rs` reimplements the AND/OR forced-capture search on the
bitboard engine with a transposition memo and resulting-position dedup. Against
the Python solver on 24 real capped-game positions it is **145x** at depth 4
with **zero disagreements**. Depth 5 fell from ~23 CPU-min/position (§30.1) to
**1.78s** (~775x) and depth 6 is reachable. Re-running §34's pilot, the
generation finisher went **44m30s -> 2m22s (18.8x)**, made the identical nine
decisions, and recovered the game §34 lost to its per-game deadline — so the
10.5x cost objection is retired even though the conversion result at 700 sims
is unchanged. `try_forced_capture_move` picks the native path automatically;
`MONSTER_SOLVER_PYTHON=1` forces the reference and a test pins that they agree.
Profiling drove this: 92% of the Python solver's time was `_get_white_actions`,
which is why Python-level memoisation bought only 1.8x (`REPORT.md` §35).

**The exact finisher now reaches data generation, and is a null at 700 sims.**
Opt-in via `MONSTER_FINISHER`, running only where the scripted oracle abstains.
A paired 24-game pilot converted **zero** cap draws: it fired 9 times across 3
games that were already Black wins, cost **10.5x** runtime (4m13s -> 44m30s),
and lost one game to its per-game deadline. The cause is not the mechanism and
not the budget — all 8 capped games end with White on a bare king with 46–55
gated Black moves each, and re-running §30.1's protocol at both 2M and 6M nodes
proved **0 forced wins with 0 exhaustion**. §30's 6-of-24 convertible games were
measured at **1600** sims; at the 700-sim generation setting Black never reaches
positions where a depth-4 capture exists. **Do not enable it for generation at
700 sims** (`REPORT.md` §34).

**Moves-left is now implemented in search, but the isolated result is null.** A
frozen Gen9 lift kept all 117 inherited tensors bit-identical and trained only
the 8,321-parameter head. It learned a measurable length signal (test MAE 26.54
versus 31.60 for the median constant; correlation 0.449), but default bounded
utility scored 0.5062 in an 80-game paired A/B and changed 0 of 16 real capped
conversion moves at 1600 sims, removing 0 of 4 reversals. Keep the mechanism
opt-in; it does not supersede Gen9 or solve conversion (`REPORT.md` §32).

**The colour gap is largely a search artifact.** On identical positions with
sims as the only variable, Gen9's self-play gap closes **0.250 -> 0.067** from
400 to 1600 sims and v21b's closes **0.133 -> 0.033** — both by about 74%. The
two models get there by opposite means: Gen9 converts White's lost wins into
Black **wins**, v21b into **draws** (§29.2). Deeper search benefits whichever
side plays Black; that is a fact about the game, not about a model.

| model | role |
|---|---|
| `models/bootstrap_v23` | **numbered release and formal bar** (owner, 2026-08-17). Byte-identical to `bootstrap_main_gen_0015/selected_epoch_016`. **First release of the BOOTSTRAP series**: the version number continues from v22 so the release ladder stays comparable, while the directory prefix changes from `fresh_start_*` because the lineage did. +130.4 Elo above v22 on a fit over all 15 pairs of the chain; 0.6567 against v22 directly over 600 games. Manifest: `models/bootstrap_v23/promotion_manifest.json`. |
| `models/fresh_start_v22` | prior numbered release and bar, retained unchanged. Last release of the `fresh_start` series. |
| `models/candidates/bootstrap_main_gen_0011..0014` | the rungs between v22 and v23. Each passed a confirmed 800-game gate against its predecessor; none is a release. The gated checkpoints are epoch snapshots (`selected_epoch_008/007/009/007`), never `best_value_net.pt`. |
| `models/fresh_start_v21b` | prior unnumbered bar, retained unchanged. |
| `models/fresh_start_v21` | prior numbered release, retained unchanged. |
| `models/candidates/gen7_scratch/screen_nominee.pt` | passed the gate against v21b (pooled 0.5600 over 400 games, z=+2.40); the working bar inside the bootstrap loop. |
| `models/candidates/gen8_scratch/screen_nominee.pt` | paired re-screen selected epoch 7; it passed the first Gen7 leg but failed fresh confirmation at Black 0.3500. No promotion. |
| `models/candidates/gen9_scratch/screen_nominee.pt` | source of V22. It passed the complete paired gate against Gen7 at 0.5950 (W 0.7200/B 0.4700), then 0.5575 (W 0.7050/B 0.4100), and directly beat v21b 0.5763 over 400 games. |
| `models/candidates/gen10_scratch/screen_nominee.pt` | rejected. It scored 0.5225 against Gen9 but Black 0.3850 failed the unchanged 0.40 floor; seed-43 also failed at 0.4925/W 0.6300/B 0.3550. No confirmation was earned. |

**The loop has now produced a clean successor.** Gen8 remained a useful data
increment but did not survive a fresh paired confirmation. Gen9 generated from
the unbeaten Gen7 bar, accumulated Gen7–Gen9 replay, trained fresh, selected
epoch 6 by worst-colour calibrated play, and cleared every binding leg twice.
The absolute 0.40 floor was not changed. See `REPORT.md` §§26–27.

**Opening diversity was measured directly, not inferred.** The historical
800-game sampled v21b–v21 re-anchor contained 650 unique states and 684 unique
state+model-colour games (85.5% effective), so its standard error needs only a
1.08× correction, not the feared ~2×. A paired-book re-anchor reproduced the
same result (0.5238 versus 0.5225 sampled). Current screens and gates use pinned,
mixed-provenance books with disjoint selection and test blocks (`REPORT.md`
§24.3). Gen10 then demonstrated that replay accumulation is not monotonically
improving: its best balanced checkpoint traded White strength for a small Black
gain and failed the Gen9 gate (`REPORT.md` §28).

**Gen9 post-gate diagnostics.** Its paired self-skew is W 0.6675/B 0.3325 over
400 games. Directly against v21b it scores 0.5763 overall, W 0.6975/B 0.4550
(400 games, paired SE 0.0157). Gen9 became V22 on 2026-08-16 after both Gen10
seed paths failed. New candidates must beat the byte-identical release model.

**The repeated confirmation Black drop is mostly measurement structure, not a
second-game model mutation.** Across 28 historical confirmed gates, Black's
raw score fell on 23 confirmations; all five modern 200-game confirmations
fell, by 6.2 points on average. This is partly selection regression (only a
lucky first leg earns confirmation) and partly the opening blocks themselves.
On the original p8 book, V22-versus-itself scored W 0.605/B 0.395 on the first
recovery gate block but W 0.695/B 0.305 on its confirmation block: a built-in
nine-point Black drop before changing either model. Checkpoint screens already
subtract an incumbent self-calibration separately by colour. Until the binding
gate does the same, interpret raw colour scores together with exact-block self
calibration and require positive calibrated deltas on both colours for a new
successor claim.

**Opening books start games already decided.** The p16 book used by screens
and gates has White down **1.3 of its 4 pawns** on average, with both armies
intact in only **3%** of its 800 entries. An 8-ply book has 78% intact. Pairing
controls for it, but the measurement begins from lopsided middlegames rather
than the opening. A shallow book needs **temperature 1.0**, not 0.5 — the
earlier 8-ply build failure (31 of 100 unique) was sampling temperature, not a
reachability ceiling; at 1.0 it produced 60 unique entries in 34 seconds with
zero duplicates (`REPORT.md` §31).

**Deep-teacher ingestion defect found 2026-08-16.** Gen7--Gen10 each generated
4,000 retained one-row teachers at 3,200 simulations, but the historical
`generation_driver.py` processing call inherited the four-ply non-human minimum
and dropped every teacher. Gen9's 102,906 rows and Gen10's 111,098 rows are
exactly twice their ordinary record counts; neither has a policy-only value
mask. This means the expensive reanalysis completed but supplied zero training
signal. The raw teacher files remain intact.

Gen7--Gen9 are now recovered into new, non-overwriting `*_teacher3200_fixed`
processed increments. Each has exactly 4,000 teachers / 8,000 mirrored rows,
2,400 Black and 1,600 White, zero teacher value weight, and verified source-game
split linkage. Their hashes are registered in `iterations/accepted_data.json`.
**The recovery campaign is closed: four arms, one trade, no successor
(2026-08-16).** Teacher policy weights 1x, 2x, and 4x on the historical
teachers, plus a fourth arm whose teachers were re-searched by V22 itself at a
balanced 50/50 side split, all move Black up and White down. None produced a
checkpoint positive on both colours at 200 games; the two that reached the
binding gate failed it (1x: Black 0.3850, aggregate 0.4925; balanced 4x:
Black 0.3800, aggregate 0.4825), as did both 4x checkpoints. No threshold
moved. See `REPORT.md` §33.

**The loop cannot currently resolve the effects it is chasing.** Eight
200-game self-calibrations of V22 *against itself* span Black 0.280–0.395
(mean 0.3425, sd 0.043) — ordinary sampling noise at 100 games per colour
leg, not block difficulty. A calibrated colour delta subtracts two such
measurements, so its SE is ≈ 0.064 and every effect this campaign measured
sits inside 2 SE of zero. The balanced arm proved it end to end: its nominee
screened at Black **+0.105**, projected to raw 0.495 on the gate block, and
scored **0.380** there against that block's own calibration of 0.390 — a
calibrated −0.010. Its raw Black barely moved (0.4050 → 0.380); the
*calibration* moved (0.300 → 0.390). The screen had selected the luckiest of
eight checkpoints. Resolving a 0.05 colour effect at 2 SE needs roughly 400
games per colour for both candidate and calibration, about 4x the current
screen cost. **Do not read a single-block 200-game colour delta as a real
effect, and do not gate on one.**

A corollary worth holding: the absolute 0.40 Black floor is being applied on
blocks where V22 scores 0.280–0.395 as Black, so an exact copy of V22 would
fail its own floor on five of the eight blocks measured. The floor has not
been changed and is not proposed to change — but a raw colour score is only
interpretable next to that block's self-calibration.

The canonical `src/iterate.py` path now uses the actual successful workload:
500 games at 700 simulations, 8,000/4,000 reanalysis at 3,200 simulations,
fresh LR-0.002/EMA training, value floor/horizon 0.5/60, mandatory teacher
census, sparse-policy-aware registry hashes, and automatically disjoint book
blocks. A new 1,200-entry p8/temperature-1.0 book draws exactly 240 positions
from each of v20, v21, v21b, Gen7, and Gen9. The recovered-teacher control is
now settled (closed, no successor), so hyperparameter tuning is no longer
blocked on it — but §33.4 applies to the tuner too: its trial ranking runs on
the same 200-game blocks and inherits the same ≈0.064 SE, so a trial winner
selected on one block is not evidence until it survives a fresh one.

### The v20-era ledger (2026-08-05, retained)

- **Incumbent and formal gate bar: `models/fresh_start_v20`.** The owner
  promoted the preserved `lc0b_attention_ema_wide64` checkpoint on 2026-08-05.
  Future candidates must clear v20 twice without moving the established
  0.40 per-color floor or 0.50 aggregate threshold.
- **v20 lineage.** The 32-channel `models/candidates/lc0b_attention_ema` keeps
  v19_B's exact data and training recipe, changing only to the compact
  attention policy head plus EMA (`0.999`) validation/checkpoint weights. It
  passed the unchanged binding gate: against v19_B it scored **0.7125 overall,
  0.925 as White, 0.500 as Black**, then **0.7375 / 0.925 / 0.550** on the
  automatic fresh confirmation. It also scored 0.825 against ramp and 1.000
  against the fixed heuristic anchor. This is the strongest automated result
  on record at that stage and passed the owner's playtest.
- **Owner gate passed 2026-08-05.** The owner found `lc0b_attention_ema` very
  strong on both sides, able to handle his White attack and vicious as White;
  only occasional spotty Black conversion moves remained. It is now the
  approved playing-strength bar for successor work (checkpoint preserved in
  place; no historical artifact was overwritten).
- **Promoted v20 source: `lc0b_attention_ema_wide64`.** The only change is widening
  the attention query/key channels from 32 to 64. Against the approved model
  it improved both colors twice: calibrated **+0.050 Black / +0.025 White**
  at 40 games, 400 sims, then **+0.1625 Black / +0.0625 White** at 80 games,
  800 sims. The campaign copy and all prior bars remain preserved.
- **v20 color-skew baseline.** An 80-game native self-match at 400 sims with
  identical weights/search on both sides produced **47 White wins, 16 Black
  wins, 17 draws**: White score **0.6938**, Black score **0.3063**. This is a
  color baseline, not a head-to-head model-strength result.
- **V21 screen closed without promotion (2026-08-05).** Attention widths 96
  and 128, a side-specific attention adapter, capture-only WDL, and scalar +
  capture-WDL auxiliary heads all failed repeatable positive deltas in both
  colors. The best Black-leaning arm (`v21_mixed_capture_wdl_w003`) produced
  +0.225 Black / +0.000 White once but repeated at +0.075 / **-0.300**.
  V20 remains incumbent. Capture terminal labels are useful pressure toward
  Black conversion but currently trade away White/general strength; see
  `REPORT.md` §12 and the `v21_*_20260805.json` artifacts.
- **Promotion-aware policy is implemented; first isolated candidate rejected
  (2026-08-05).** The legacy 4096 source/destination policy merged q/r/b/n
  promotions. The opt-in 4288 ABI adds 192 distinct promotion logits in both
  Python and native search while old checkpoints remain compatible. A frozen
  v20 lift trained only the 1,548-parameter promotion delta on 3,790 augmented
  rows. Held-out full-policy top-1 on promotion rows rose 7.8% -> 57.8% (val)
  and 4.1% -> 64.5% (test), but the binding gate was 0.5125 then 0.5000 versus
  v20; confirmation Black scored 0.325 below the 0.40 floor. Keep the
  representation fix and reject `v21_promotion_policy_exact` as v21. Do not
  turn obvious tactical givens into phase-specific gates or adapters; they
  must emerge from general training and search.
- **The iterative bootstrap pipeline is implemented (2026-08-05).**
  `src/iterate.py` is now a resumable manifest-driven state machine covering
  self-play, Black-weighted general deep-search reanalysis, isolated processing,
  replay composition on the exact v19_B/V20 anchor, conservative end-to-end
  fine-tuning, offline regression checks, the binding gate, self-color skew,
  and explicit promotion. It never overwrites numbered V20. A live smoke run
  completed generate -> reanalyze -> process with 2 games, 200 positions, one
  deep-search teacher, 402 augmented rows, and no illegal targets. The first
  real generation must hold architecture fixed; moves-left remains opt-in so a
  pipeline result is not confounded with a model change.
- **Bootstrap triage corrected (2026-08-06).** Held-out policy/value comparison
  is advisory by default; imperfect self-play targets are not a strength oracle,
  and only the unchanged binding game gate rejects a successfully trained
  candidate. The first production arm had Black sign accuracy +3.14 points but
  was denied games by a White policy top-1 drop of 1.52 points under the old
  rule. Its recovered game gate scored 0.650 overall / 0.725 White / 0.575
  Black in the first V20 leg, then failed the independent confirmation at
  0.3875 / 0.575 / 0.200. It was therefore rejected for unstable play rather
  than proxy metrics. The loop also honors the measured eight-worker GPU default
  (7.11 decisions/s versus 5.39 at four) instead of the obsolete hard
  four-worker NN cap.
- **Bootstrap production contracts added (2026-08-06).** Processed
  champion-generated data is accepted into a run-local immutable registry
  immediately after processing, independent of whether its candidate later
  passes. Recent accepted generations therefore accumulate across explicit
  `--continue-after-reject` runs. Replay keeps validation/test membership
  unchanged and deterministically smooths only the training indices across
  side, true capture outcome, and material-count phase quantiles. Checkpoint
  selection now compares every epoch against V20 on identical validation rows,
  optimizing worst-color policy/sign gains with fixed incumbent regression
  guards. If every epoch violates those guards, the generation ends as
  `rejected_training`; it cannot fall back to an unsafe or duplicate model.
- **Bootstrap checkpoint play-testing hardened (2026-08-06).** Four production
  generations all received real binding games. Generation four showed that the
  validation proxy chose the wrong stopping point: preserved epoch two passed
  a fresh full gate (V20 0.525 overall / 0.600 White / 0.450 Black, confirmation
  0.6875 / 0.800 / 0.575), while selected epoch four failed. The 80-game,
  800-simulation calibrated read still rejected epoch two: +0.275 White and
  +0.119 aggregate, but -0.0375 Black. Its pooled self-match White score was
  0.781 versus V20's 0.694, confirming increased White skew. The pipeline now
  play-tests every unique preserved epoch, nominates by worst calibrated color,
  and requires an independent calibrated 80x800 improvement in both colors
  after the binding gate. V20 remains champion.
- **First bootstrap successor cleared every automated gate (2026-08-06).**
  `models/candidates/bootstrap_gen5_teacher3200_full/selected_epoch_002.pt`
  keeps the V20 architecture and fine-tunes the full replay with 4x-weighted,
  policy-only teachers searched at 3200 simulations. Its calibrated 80-game,
  800-simulation read improved **+0.0125 Black / +0.1375 White / +0.075
  overall**. The independent full binding protocol then scored 0.650 / 0.825 /
  0.475 against V20 and 0.550 / 0.675 / 0.425 on the fresh V20 confirmation;
  it also retained 0.900 against ramp and 0.950 against the heuristic. Its
  equal-settings self-match reduced pooled White score from V20's 0.6938 to
  **0.5875**. This is the strongest bootstrap successor candidate, but it is
  not V21 until the owner playtests and promotes it.
- **Moves-left follow-up did not clear the Black stability bar (2026-08-06).**
  An otherwise identical auxiliary-head arm learned the remaining-length
  target and produced two positive checkpoints in the full A/B screen. Epoch
  six beat the fixed successor at 80x800 (+0.025 Black / +0.100 White), but
  failed the fresh V20 binding Black leg at 0.375 < 0.400. The earlier,
  Black-leaning epoch three independently read -0.050 Black / +0.1375 White
  against the fixed successor. Keep the head implemented and opt-in; do not
  promote this arm.
- **Bootstrap production audit completed 2026-08-06.** Deep-search teachers
  now inherit the source game's train/validation/test split (the pre-fix demo
  leak was 10/40 and 61/180; both are now zero). Worker phases fail after a
  bounded no-progress interval, generated batches have an explicit completion
  floor, reanalysis/replay publish atomically, replay artifacts are fully
  hashed, phase seeds do not overlap, concurrent loops are locked out, and
  pipeline training memory-maps the large replay arrays. Full discovery now
  passes 522 tests.
- Owner's read after playing the v19-era models: *"White isn't doing
  terribly — the play is coherent and attacking chances are taken; definitely
  improved. Black's defending play is a big improvement as well. **Black
  conversion is a big problem.**"*
- **The live problem is Black conversion**, but the next intervention is
  general iterative learning rather than a post-promotion rule patch. The
  pipeline reserves 60% of deep-search teachers for Black and retains explicit
  per-side promotion floors.
- **The earlier, unpromoted v20-named experiment closed 2026-08-03: both arms
  rejected** (`v20` Black 0.10
  vs v19; `v20w` PASS-then-FAIL — its gate pass was move-limit relabels, Black
  0.50 → **0.30** under captures-only scoring; both in `models/rejected/`).
  Asymmetric generation at this scale did not produce a Black that converts
  against v19; the turn cap tested dead (2.7× moves, identical captures). See
  `REPORT.md`. The engine rewrite and re-baseline are now complete; the active
  campaign is the post-E5 data ladder and separately measured E6 search work.
- **Post-E5 data ladder closed 2026-08-04: all three arms rejected.** Against
  `v19_B`, frozen-corpus base / +e1500 policy-only / +41 owner games scored
  0.250 / 0.100 / 0.075 as Black (overall 0.4750 / 0.3875 / 0.2125). The bar
  remains `v19_B`; the active work is now Black-first search and opt-in
  LC0-derived architecture screens.
- **The rewrite's E0-E5 phases are complete.** Rules, heuristic and encoding
  reached exact parity; native MCTS passed 400/400 move agreement and its
  200-game self-match; the full gate runs in 6.5 minutes instead of ~34. E5
  established the new captures-only/native zero point and moved the gate bar
  to v19_B. The first E6 solver measurement was a null (+0.02 true
  captures at both 1600 and 3200 sims), so it stays off by default.
