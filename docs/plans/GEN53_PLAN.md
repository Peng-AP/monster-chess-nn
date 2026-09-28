# Gen53 — decision rules declared before gen52 results (draft, September 28, 2026)

Status: **draft; rules fixed before any gen52 selection, gate or diagnostic
result existed** (gen52 was training Arm B when this was written). The owner
authorized continuing "overnight and into tomorrow" on September 28.
Implementation (`tools/gen53_campaign.py`) follows gen52's completion, reusing
the gen52 driver with only the declared changes.

## Primary evidence of general strength

The **held-out mean**: the average 3,200-simulation score against B2 and v27,
which were never training opponents (owner, September 26). gen51 showed that
beating the teacher can hide a real hole, and that older opponents find it.
Gate v4 against the teacher remains the binding gate.

## Rule 1 — whether gen53 launches

- **Launch** only if at least one gen52 arm's nominee passes gate v4 against
  its teacher (gen51 deep-value).
- **No launch** if neither passes: write up the results, notify the owner and
  wait.

## Rule 2 — gen53 teacher

Among gen52 nominees that passed gate v4, pick the one with the **higher
held-out mean**. If the two held-out means are within 1.0 percentage point,
the gen52 arm-vs-arm match at 3,200 decides (≥ 50% for Arm B → Arm B,
otherwise Arm A). The teacher is used unpromoted; promotion is the owner's.

## Rule 3 — whether the model pool continues

Pool games stay in gen53's recipe **if Arm B's held-out mean ≥ Arm A's held-out
mean − 1.0 pp, and Arm B's gen49 score ≥ Arm A's gen49 score**. The gen49
condition is there because the pool exists to fix the hole gen49 exposed.
Otherwise the pool is dropped and gen53 is a single-recipe generation.

- If kept: same design as gen52. Arm A is the base recipe, Arm B adds 1,200
  pool games, so the pool effect is measured again, not assumed.
- **Pool membership stays exactly as the owner set it:** v28, gen49, gen48,
  v26. **B2 and v27 stay held out.** Adding newer models (e.g. gen51's control
  nominee) is an owner decision, not made here.

## Everything else carries forward unchanged

- 2,800 self-play games at 1,600 simulations, 30-ply exploration; 400 forks
  at 12,800; reanalysis 24k → 12k at 12,800; 8-generation replay; scratch
  training, seed 3173.
- Deep-value source: 768 disagreement roots × 2 at 6,400. Strict labels,
  value weight 4, policy 0. Prior deep-value increments roll forward within
  the replay window.
- Selection and evaluation as gen52: gate v4 against the new teacher;
  diagnostics against B2, v27, gen49 and v28; self-play at both depths; the
  line breakdown after e4+d4 …d5.
- Fresh seed namespaces, full rehearsal before production, nothing promoted.
