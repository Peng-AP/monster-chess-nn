# Gen52 Arm L: a 2× wider network on Arm B's data (declared September 30, 2026)

Owner go-ahead on September 30 ("Let's do it"). A training-only experiment:
no new games are generated.

## Question

Is network size now a limit? The August capacity null (v19_C, 2.74× tower)
was measured when the corpus had shrunk to 445k–936k rows and the network
already overfit by epoch 5. Arm B's replay is about 5.1M rows.

The strength ladder (`docs/experiments/elo_rr/LADDER_RESULTS.md`) supports
the question: v29 gains almost nothing from search beyond 3,200 simulations,
so further strength has to come from the network.

## Design: one change

**Arm L = gen52 Arm B with the residual tower widened from 128 to 192 channels.**

- Tower `64,64,192,192,192,192,192,192` instead of `64,64,128,…`. That is
  4.00M parameters instead of 1.91M (2.09×).
- Measured on the RTX 5060 Ti with random weights:
  - inference at batch 16 (the search batch) takes 3.08 ms vs 3.55 ms, so
    search speed is unchanged;
  - a training step at batch 256 takes 2.0× as long.
- Wider rather than deeper, because a 12-block tower of the same size was
  1.35× slower per search evaluation.
- **Everything else is identical to Arm B:**
  - the same composed replay (`bootstrap_replay_main_gen_0052_armB`,
    manifest hash checked);
  - Arm B's exact train command (scratch, seed 3173, 30 epochs, patience 10,
    same learning rate, weight decay and EMA);
  - the same selection and gate code and seeds as gen52's arms, so openings
    match Arm B's.
- The teacher and gate bar is v29. The gate reuses gen52's teacher par.

## Evaluation

1. **Selection:** `checkpoint_screen` against v29, then 12,800 probes of the
   top three epochs under the gen51 nominee rule. Seeds are the same as
   Arms A–C.
2. **Gate v4 against v29.**
3. **Diagnostics:** B2 and v27 (held out), gen49 and v28, and self-play at
   3,200 and 12,800.
4. **Arm L vs Arm B head-to-head:** 400 games at 3,200 and 160 at 12,800.
5. **Elo placement** on the joint scale (v21 = 1600), with the same
   two-stage design as the ladder: 320 games against the 16 rated models.

## Criteria, fixed now

Primary measurement: Arm L vs Arm B at 3,200 (400 games). Let *s* be L's
score, *SE* its per-game standard error, and *u* the endpoint-unique score.

| Verdict | Condition |
|---|---|
| **Capacity helps** | *s* − 1.96·*SE* > 50% **and** *u* > 50% **and** L's held-out mean (B2, v27) ≥ Arm B's 91.4% − 1.0 pp |
| **Capacity harmful** | *s* + 1.96·*SE* < 50% |
| **Null** | anything else |

- The unique-score condition guards against a win driven by a few repeated
  openings.
- Secondary results are reported but decide nothing: gate v4, the 12,800
  head-to-head, per-colour scores and the Elo placement.
- **Promotion is separate.** A gate v4 PASS against v29, together with the
  held-out non-regression in `docs/protocols/PROMOTION_RULE.md`, makes Arm L
  eligible. The owner decides. Nothing is promoted automatically.

## Conduct

- New driver `tools/gen52_large_campaign.py`, modelled on Arm C's; a full tiny
  rehearsal runs first.
- One managed job. The web server keeps running, so visitors may see slower
  replies.
- Results go in `docs/experiments/gen52/LARGE_RESULTS.md`.
- **Time guide, not a promise:**
  - training about 11 h (Arm B's 5.5 h × the measured 2.0× step cost);
  - evaluation about 8.5 h (Arm C's, since search speed is unchanged);
  - Elo placement about 0.5 h.

  About 20 h in all, re-estimated from the running job.
