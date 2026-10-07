# Gen55 plan (October 7, 2026)

## Owner decision (October 7): "Add gen 54 to the website. Should add one more teacher to the pool. Then start 55"

The draft below stands, with these changes:

- **Teachers:** gen54 2,800 + 400 forks; extra teachers gen53, gen52 Arm R
  **and gen52 Arm LR**, 700 games each. LR was in gen54's mix and is the
  "one more teacher".
- **Pool:** gen54's hole-scan pool plus gen53. That is 8 opponents, 86 games
  per opponent per colour, 1,376 games.
- **v29's par is measured once at the start.** The selection's White check and
  the gate against v29 both use it.
- **Evaluation folded into production:**
  - gate v4 against gen54 and against v29;
  - the diagnostics;
  - the v27-position probe;
  - the value audit;
  - promotion eligibility in the summary.
- Driver `tools/gen55_campaign.py`. Decision
  `docs/plans/gen55_teacher_decision.json`. Recipes `tools/recipes/gen55*.json`.
- Seeds: 3.530e9 (production) and 3.565e9 (rehearsal), clear of every
  earlier block.

---

# Original draft (superseded where the decision above differs)

Goal: the **next release**. The release candidate must pass gate v4 against
v29 and keep the held-out mean at or above 91.7% (`PROMOTION_RULE.md`).

## Where we are

Evidence: `docs/experiments/gen54/GEN54_RESULTS.md` and
`docs/experiments/repetition_search/RESULTS.md`.

gen54 is the strongest model on most measures:

- 79% against gen53;
- it never loses as Black;
- it fixed gen53's v27 hole;
- held-out mean 92.2% (B2 98.4% with the new engine).

It fails gate v4 against v29 on one thing: **White at 3,200** (−19 / −21 pp).

**The White losses are concentrated.**

- Against v29, 60 of gen54's 141 games in the line e4 d4 …d5 **c4 Ke2** …dxe4
  were losses, as were 30 of 66 in its d4-first twin. gen53's main line,
  e4 d4 …d5 **Ke2 Ke3**, held 156 draws in 157 games.
- At 12,800, gen54 holds the c4 line (21 of 21 draws).
- gen54's raw network prefers c4 no more than gen53's does. The choice comes
  from its search's evaluations at 3,200.

**Repetition awareness is now in the engine** and is neutral in level play. All
gen55 data, selection and gates use it, and every par is measured fresh.

## Proposal: one arm, three changes from gen54

1. **Teacher: gen54**, with **gen53 as an extra teacher (700 games)** in place
   of one of the gen52 arms:
   - gen54 is the strongest, and its self-play and pool games contain its own
     White losses as training signal;
   - gen53 contributes the White repertoire that holds against v29.

   Proposed mix: gen54 2,800 + 400 forks, gen53 700, gen52 Arm R 700.
2. **Pool: unchanged hole-scan pool plus gen53.** That is v29, gen52 Arms
   A, B, C, L, LR and R, and gen53, with only the teacher's moves carrying
   policy weight. v29 stays in the pool, so gen54's White now meets v29's
   Black in training.
3. **Selection adds a White check against v29** (pre-declared):
   - Among the epochs that pass the deep probe against the teacher, keep only
     those whose White scores within 5 pp of v29's own White par in 80 games
     at 3,200 against v29.
   - Then take the best screen rank. If none qualifies, take the best on
     that White check and say so.

   This is the measure gen54 failed, applied before the gate rather than
   discovered after it.

Unchanged from gen54:

- ramped labels;
- the deep-value continuations (reference v29);
- the 1.9M tower, scratch training, seed 3173, replay of the last 8 generations;
- the diagnostics (B2, v27, gen49, v28, self-play).

The follow-up runs gate v4 against **both** gen54 (the teacher) and v29 (the
release), then the v27 probe and the value audit, and reports promotion
eligibility. Nothing is promoted automatically.

## Cost and risks

- About 30 hours, from gen54's measured stage times, plus about 1 hour for the
  White selection check.
- The White check uses v29, which is the release, not a held-out model. B2 and
  v27 stay held out.
- One arm again, so the three changes cannot be attributed separately.
- If gen55's White still fails against v29, the next lever is the opening
  itself (for example, training opponents that force the c4 line). That
  waits for this result.

## Owner decisions needed

1. Approve the plan as written, or change the teacher mix, the pool or the
   selection rule.
2. Approve launch: a rehearsal, then production, run as managed jobs as before.
