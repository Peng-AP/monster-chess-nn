# Gen54 plan (drafted October 5, 2026; owner decision the same day)

## Owner decision (October 5): "a mix of teachers as well as the other proposed changes. 1 arm only."

This replaces the draft's teacher and arms questions.

**One arm, all changes together.** That means no A/B, so the pool's own
effect is not separately measured.

**Teacher mix (self-play games, all at 1,600 simulations, 30-ply
exploration):**

| Teacher | Games | Role | Why |
|---|---:|---|---|
| **gen53** | 2,800 | Main teacher: canonical self-play, 400 forks, reanalysis at 12,800, deep-value continuations (reference v29) | Strongest model |
| gen52 Arm R | 700 | Extra increment (policy and value) | Best teacher at the deep teaching depths |
| gen52 Arm LR | 700 | Extra increment (policy and value) | Wide tower, a different architecture; beats v29 head-to-head |

The main teacher contributes two-thirds of the self-play. v29 is not a
teacher; it is the weakest at the teaching depths.

**Opponent pool (hole-scan rule):** v29 and gen52 Arms A, B, C, L, LR and R.

- 1,204 gen53-vs-pool games, 86 per opponent per colour.
- Only gen53's moves carry policy weight.

**Labels:** every source on the main ramped game results. The deep-value
data is gen53's (ramped) plus gen54's own (ramped).

**Unchanged:**

- 1.9M tower, scratch training, seed 3173, replay gen47–gen54;
- selection and gate v4 against the teacher (gen53);
- diagnostics against B2, v27, gen49, v28 and self-play.

**Follow-ups, as for gen53** (`tools/gen54_followup_chain.py`):

- gate v4 against the release (v29);
- the v27-position probe ("did gen54 fix gen53's endgame hole?");
- the value audit.

**Primary question, without an A/B:** is gen54 stronger than gen53, and is
the endgame hole closed? Evidence:

- gate v4 against gen53;
- the probe;
- held-out v27;
- promotion eligibility under `PROMOTION_RULE.md` (the owner decides).

Driver `tools/gen54_campaign.py`. Decision file
`docs/plans/gen54_teacher_decision.json`.

---

# Original draft (superseded where the decision above differs)

Status: **draft**. Nothing here runs until the owner approves it. Evidence:

- `docs/experiments/gen53/GEN53_RESULTS.md`;
- `docs/experiments/gen53/MORNING_20261005.md` (position probe, hole scan,
  top group);
- `docs/experiments/gen53_prep/TEACHER_SELECTION.md`.

## Proposal

### 1. Teacher: gen53 (to be confirmed by the teacher-selection rule)

- **For:**
  - +105 Elo vs v29 against the 9-model group (interval +87 to +124);
  - beats every model head-to-head;
  - passes gate v4 against both Arm R and v29;
  - value bias −0.006.
- **Against:** a White endgame hole that only the held-out v27 punishes.
  This is why its held-out mean is 88.3%.
- **To confirm:** re-run `TEACHER_SELECTION_PLAN.md`'s rule (round robins at
  1,600, 6,400 and 12,800) with gen53 added. That is about 10 h of
  evaluation. Or the owner may accept gen53 on current evidence.

### 2. Opponent pool: back, chosen by the hole scan

The owner's guess ("broader opponents") with the owner's limit ("not worse
play for the sake of another model"), made measurable.

**Proposed rule.** An opponent joins the pool if it is not held out (B2,
v27), and in a hole scan against the teacher (100 games, 3,200) it both:

- **exposes holes:** at least 2 distinct losing endpoints for the teacher;
- **is not an easy win:** the teacher scores at most 85% against it.

Applied to gen53's scan today, the pool would be **v29 and gen52 Arms A, B,
C, L, LR and R** (7 models).

- gen49 found 3 holes but scored 88.5%, so it is left out by the second
  condition.
- v24, v25, v26, v28 and gen48 found 0–1 holes, so they are left out.

This replaces gen52's pool (v28, gen49, gen48, v26), which by this scan
exposes almost nothing for the current teacher.

**Size and mechanics as in gen52:**

- 1,200 teacher-vs-pool games at 1,600 simulations, split evenly;
- only the teacher's moves carry policy weight;
- values are from game results, on the main ramp.

### 3. Measure the pool, don't assume it

Two arms, as gen52 did:

| Arm | Recipe |
|---|---|
| A | gen53's recipe (no pool) |
| B | A plus the new pool |

Both arms keep Arm R's label recipe: deep-value data ramped, and only the two
newest deep increments.

### 4. Targets fixed before results

| Target | Measure |
|---|---|
| Primary (pool effect) | Arm B vs Arm A head-to-head at 3,200 (400 games); same verdict rule as Arms L and R |
| The gen53 hole | Position probe: the new nominees as White from the v27 position (10 games); gen53 lost 10/10, R and v29 won 10/10 |
| Held-out | B2 and v27 at 3,200, 160 games each. **v27 stays held out**, and its endgame must not be fed into training directly. |
| Promotion | Unchanged (`PROMOTION_RULE.md`). Gate v4 against the current release plus held-out non-regression; the owner decides. |

### 5. Cost

| Work | Time |
|---|---|
| Teacher re-selection (optional) | ~10 h |
| gen54 two-arm campaign | ~40 h (gen52 took 39 h) |

Single arm with the pool: ~25 h.

## Decisions needed from the owner

1. Teacher: gen53 now, or after re-running the selection rule with gen53
   included?
2. Adopt the hole-scan pool rule (thresholds: ≥ 2 distinct losing endpoints,
   ≤ 85% score)?
3. Two arms (measure the pool) or one arm (pool on)?
