# Gen52 Arm C — capped deep-value share (declared September 29, 2026)

Owner go-ahead on September 29 ("push, then do step 2"). A training-only
experiment: no new games are generated.

## Question

Did gen52 fail to improve on its teacher because the deep-value share of
value training nearly doubled? It rose from 13.0% in gen51's deep-value arm to
22.3% in gen52, because gen51's disagreement increment rolled forward next to
gen52's own, each at value weight 4, with about a third draw labels.

## Design

**Arm C = gen52 Arm B minus the rolled-forward gen51 deep-value source.**

- Keep: the canonical gen52 replay (gen45–gen52), gen52's own deep-value
  increment, and gen52's 1,200 pool games.
- Drop: `gen_0051_deepvalue`.
- Expected deep-value share: about 11–12% of value weight, close to gen51's
  13.0%. Measured and reported before judging.
- Training: exactly gen51/gen52's train command (scratch, seed 3173,
  30 epochs / patience 10); only the data and model directories differ.
- Selection and evaluation identical to gen52's arms, **with the same seeds**
  (matched openings for screens, probes and gates):
  - `checkpoint_screen` against the teacher (gen51 deep-value), 12,800 probes
    of the top three epochs, the gen51 nominee rule;
  - gate v4 against the teacher, reusing gen52's teacher par;
  - diagnostics against B2 and v27 (held out), gen49 and v28 (pool members),
    and self-play at 3,200 and 12,800.
- **Arm C vs Arm B head-to-head:** 400 games at 3,200 plus 160 at 12,800,
  always run (the only direct measurement of the change).

## Criteria, fixed now

The hypothesis is **supported** if all three hold:

1. Arm C's held-out mean (B2, v27) ≥ Arm B's 91.4%.
2. Arm C's combined gate-v4 score against the teacher at 3,200 > Arm B's 47.4%.
3. Arm C scores ≥ 50% against Arm B at 3,200.

It is **contradicted** if Arm C is worse than Arm B on criteria 1 and 2.
Anything else is **inconclusive**.

Separately, a gate-v4 PASS against the teacher would make Arm C a teacher
candidate for gen53 under `docs/plans/GEN53_PLAN.md`. That rule's pool
condition compares against Arm A; it does not auto-launch gen53 here.

## Conduct

- New driver `tools/gen52_poolcap_campaign.py`; full tiny rehearsal first.
- One managed job; the web server keeps running (visitors may see slower
  engine replies).
- Nothing promoted; results in `docs/experiments/gen52/POOLCAP_RESULTS.md`.
- Guide: about 13–14 hours (training ~5.5 h, selection ~2.5 h, gate ~2 h,
  diagnostics ~2 h, head-to-head ~1 h). Re-estimate from the running job.
