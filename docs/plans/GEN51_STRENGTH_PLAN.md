# Better play after v28 — proposed September 25, 2026

Status: **proposed, not started.** Owner direction on September 25: work toward
better play now; shipping (see `SHIPPABLE_ENGINE_PLAN.md`, deferred) comes
later. Nothing here is queued. Each stage below is a separate
plan → implement → wait cycle with its own rehearsal and owner go-ahead.

## 1. Goal and success criteria

Produce a successor to v28 that is **stronger at the playing budget without
giving anything back at depth or in either colour**. The last three
experiments each traded one colour for the other; this plan is built to stop
that.

A candidate counts as better only if it passes, against v28:

1. **3,200 simulations (binding, the existing sampled gate):** two 400-game
   legs, each above 50% overall, and each colour at least v28's same-colour
   self-par minus 5 pp.
2. **12,800 simulations (new binding guard):** 160 games. Point estimate at
   least 47.5%, and each colour at least v28's 12,800 self-par minus 10 pp.
   This is a non-inferiority check, not a second win condition; it exists
   because gen50 epoch15 passed at 3,200 while its White fell to 26% at 12,800.
3. **Diagnostics, always run and never binding:** gen49, v27 and B2 at 3,200;
   actual-colour self-play at both budgets; endpoint-unique scores.

Promotion still needs the owner's playtest and approval. A gate PASS never
promotes anything by itself.

## 2. What the evidence says

| # | Finding | Evidence | Implication |
|---|---|---|---|
| E1 | **The generation loop still compounds.** | gen48→gen49 94% over 800 games, gen49→gen50 75% over 800 at 3,200 | Keep the loop; don't redesign it |
| E2 | **Gains are colour-traded and depth-dependent, and the 3,200-only gate can't see it.** | gen50 epoch15: 73.9% gate PASS, but White 26.25% at 12,800. gen50 epoch14: 75% at 3,200, 54.7% at 12,800 | Add a 12,800 guard to selection and gating (§1) |
| E3 | **The value head is the live lever, and deeper outcomes move it.** | A value-head-only fit on 1,833 new distinct positions from 6,400-sim continuations gave +59 Elo at 3,200 (White +14 pp vs par). The replay-only arm with the same labels failed nomination (43% vs epoch14 in its screen). The c5 error was traced to value/search interaction | Deeper-play outcomes at disagreement positions are a proven value signal. Make them a standard data source |
| E4 | **Training data is narrowing.** | Distinct encoded inputs among value rows: gen49 62.4%, gen50 53.0% (both include mirror copies). In gen50 the 100 most repeated positions are 14.5% of value rows. Self-play already uses temperature 1.0 for 15 moves plus Dirichlet 0.3/0.25 | Each generation learns from fewer situations; opening positions dominate the value loss |
| E5 | **Value labels come from shallower play than the play we care about.** | Value targets: outcomes of 2,800 games at 1,600 sims (distance-tempered). Play at 3,200; guard at 12,800. Values correlate only r≈0.36–0.67 across 3,200→12,800 | Weight deeper outcomes more; consider strict capture labels |
| E6 | **Architecture and capacity are nulls.** | B2 state-CNN vs control on identical data: all arms within about ±3 pp of gen47, no paired interval excluding zero. v19-era 2.74× tower also null. Validation loss bottoms mid-run | Spend compute on data, not networks |
| E7 | **Search constants are old.** | c_puct 1.5, FPU 0.30 and policy temperature 1.0 were set in August on v19/v20-era networks (`docs/history/REPORT.md` §8) | Cheap to re-check with games only, no training |

## 3. Strategy

Keep the proven generation loop and v28 as teacher. Spend the added compute
on the two things the evidence points to, **value targets from deeper play
(E3, E5)** and **position diversity (E4)**, and make evaluation see depth
(E2). Run cheap, no-training checks first so that the expensive generation
inherits their results.

## 4. Stages

Durations are sized from measured runs: gen50's full chain took 17h18m
(launched 00:50, done 18:08 on Sept 16, including 2,280 post-selection games);
gen48 training took 3h15m; the calibration campaign played 3,696 games plus
two head fits in 6h57m. They are guides, not promises; re-estimate from each
running job's own progress.

### Stage 0 — audit and instrument (no GPU, 1–2 sessions)

1. **Diversity audit** (extends the September 25 count). Measure distinct
   positions after collapsing mirror pairs, by phase (Black, White first half,
   White second half), by game ply bucket and by opening family, for the gen50
   increment and the 8-generation replay composite. Count distinct positions
   with conflicting outcomes, and what share of the 12,800-sim reanalysis
   teachers fall on already-common positions. Output
   `benchmarks/diversity_audit_20260926/` plus a short note.
2. **Gate v4 spec.** Implement the §1 criteria as a new protocol version in
   `tools/gate_sampled.py`: the 3,200 legs unchanged plus a 12,800 guard leg
   with its own cached v28 self-par. Constants go in code with no CLI override,
   pinned by tests. Old reports keep their version; nothing is reinterpreted.
3. **Selection spec.** Saved-epoch screening adds a short 12,800 probe
   (80 games vs v28) for the top three 3,200-screen epochs, so the nominee is
   chosen knowing its deep behaviour.
4. **v28 self-par at both budgets**: 400 games at 3,200 and 160 at 12,800,
   cached once. This is the only GPU work in Stage 0, about an hour or two.

**Exit:** audit note; gate v4 and selection code with tests; cached v28 par.

### Stage 1 — search-constant check on v28 (GPU, games only, about 5 h)

One-factor changes against v28 at its current settings, same network, normal
start, sampled openings:

| Arm | Change |
|---|---|
| A | c_puct 1.0 |
| B | c_puct 2.0 |
| C | FPU reduction 0.20 |
| D | FPU reduction 0.40 |

- **Screen:** 400 games per arm vs the default at 3,200 (200 per colour).
- **Nomination (fixed now):** the best arm with overall at least 52.5% and
  both colours at least default self-par minus 5 pp. At most one nominee.
- **Confirmation:** a fresh 400-game block at 3,200 plus 160 at 12,800, judged
  by the gate v4 rules against the default.
- **If confirmed,** it becomes the engine default. That changes the runtime
  identity, so it gets a new version tag and is recorded in `CONTEXT.md`.
  Gen51 generation then uses it. **If not,** the defaults stand and the result
  is reported as a null.

Why before gen51: search settings shape every game gen51 generates and every
target its reanalysis produces, so settle them first rather than confound them
with data changes.

### Stage 2 — gen51 generation with a shared two-arm data pool (GPU, 1 overnight)

Teacher v28 (with Stage 1's search settings if they were confirmed). Same
recipe as gen50 unless listed:

| Source | gen50 | gen51 (both arms generate it once) |
|---|---|---|
| Normal-start self-play | 2,800 @1,600 | same |
| Parent-linked continuations | 400 @12,800 | same |
| Reanalysis (policy-only) | 24k sampled → 12k retained @12,800, 60% Black | same |
| **Disagreement continuations (new)** | none | **768 roots** from the self-play games, selected by the calibration's generic model/search-disagreement rule; phases balanced 50% Black / 25% each White half. **Two completed continuations per root at 6,400** (v28 vs v28 with separate per-colour trees), 1,536 games. Two runs per root give repeated labels, so their noise can be measured (handoff §12 item 2) |

Rehearse the whole chain at tiny scale first. Pilot 32 disagreement roots in
production to measure continuation cost, then size the stage from that pilot,
not from this document. Family-linked splits apply to everything: parent,
forks, continuations and reanalysis descendants stay in one split.

### Stage 3 — two training arms from the same data (GPU, about 1 overnight)

Both arms: scratch training, gen50 architecture/optimizer/seed recipe, the
8-generation rolling replay plus the gen51 increment.

| Arm | Value targets | What it tests |
|---|---|---|
| **Control** | gen50 recipe: distance-tempered outcomes; disagreement continuations excluded | Does v28-as-teacher alone keep compounding (E1)? |
| **Deep-value** | Control, **plus** disagreement-continuation positions as value rows with strict capture labels (draw = 0), value weight 4 (law 17: small high-quality sources vanish at 1×); repeated-label positions averaged | Does E3 generalize from a head-only fit to full training? |

Each arm: saved-epoch screen with the Stage 0 selection spec (3,200 screen,
then the 12,800 probe for the top three epochs), one nominee per arm, then the
full gate v4 against v28 and the §1 diagnostics. **Both arms get games**; offline
loss is advisory only. If both pass, the arms play each other: 400 games at
3,200 and 160 at 12,800. The better one is offered to the owner.

A third arm (duplicate-aware weighting: identical positions share one unit of
weight, their labels averaged) is **specified but not run** unless the Stage 0
audit finds most of the value loss concentrated in a small set of positions.
Adding it costs one training run and about 700 games.

### Stage 4 — decision and what follows

- **Deep-value passes and beats control:** the disagreement-continuation source
  becomes a standard part of the recipe. gen52 follows with no other change.
- **Control passes, deep-value does not:** keep the gen50 recipe. Revisit label
  noise with the repeated-continuation data, but not in the next generation.
- **Neither passes:** no promotion. Diagnose with the saved matched-root and
  crossover tools, as after gen50. Only then consider the diversity lever (§5).

## 5. Held back deliberately

- **More self-play exploration** (e.g. temperature for more moves, stronger
  root noise) to counter E4. It is a standard AlphaZero lever and changes only
  training data, not the evaluation instrument. But the owner stopped
  forced-variety expansion on September 14, so this is an **owner decision**,
  proposed only if Stage 0 shows the narrowing is concentrated rather than
  benign.
- **Architecture changes, larger networks, WDL or moves-left heads:** measured
  nulls (E6).
- **Opening bans, move patches, hard-coded tactical rules:** excluded by owner
  rule.
- **Promoting on a 3,200-only result, relaxing any threshold after seeing
  data, or treating deep-search outcomes as perfect-play truth.**

## 6. Resources and conduct

One heavy GPU stage at a time, at most 8 workers, about 12 GiB VRAM target,
nothing heavy while the owner plays. Every stage: frozen plan document, full
rehearsal, receipts and hashes, all post-selection branches run even after a
measured FAIL (an execution error stops the chain), long completion waits
instead of polling, and results written to `docs/experiments/gen51/`. New
drivers live under `tools/` in a new namespace. Old campaigns resume only
from their snapshot commit.

## 7. Owner decisions needed

1. Go ahead with Stage 0 (no GPU apart from the par games) and Stage 1.
2. Make the 12,800 guard binding for new gates (gate v4). It does not apply
   retroactively.
3. Gen51's two-arm design: one extra training run plus about 1,000 extra
   screening/gate games, in exchange for a causal answer about deep-value data.
4. Whether the more-exploration lever may be proposed later (§5).
