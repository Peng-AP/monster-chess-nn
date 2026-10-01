# Gen52 Arm R: deep-value data on the main game-result labels (declared October 1, 2026)

Owner, October 1: "retrain with game result labels and retest." The owner
also noted that White's lower score may simply be how the game is, so this is
a comparison, not a "fix".

## Question

The value audit (`docs/experiments/gen52/VALUE_AUDIT.md`) found that the
White-pessimism shift tracks the deep-value share. Deep-value rows are the
only source trained on strict, **undiscounted** capture results; every other
source trains on the processor's ramped game results (floor 0.5, horizon 60).

Does putting the deep-value rows on the same labels change the model?

## Design: one change

**Arm R = gen52 Arm B with both deep-value increments rebuilt on the main
corpus's labels** (`tools/process_linked_extra_ramped.py`):

- Same raw continuation games, parent splits, seed, positions (hash-checked)
  and value weight ×4, with policy weight 0.
- Only `game_results.npy` differs: ramped processor results instead of strict
  undiscounted captures.
- Composition: Arm B's receipted compose command with the two deep-value
  sources swapped for the rebuilt ones.
- Training: Arm B's receipted train command, with only the data and model
  directories changed. That is the 1.9M tower, seed 3173.
- Selection, gate and diagnostic code and seeds match Arms A–C and L.

## Evaluation

1. Value audit of the trained model (`tools/value_colour_audit.py`, same
   games). This is diagnostic and gates nothing.
2. Checkpoint screen against v29, then 12,800 probes, under the gen51 nominee
   rule.
3. Gate v4 against v29.
4. Diagnostics: B2 and v27 (held out), gen49, v28, and self-play at 3,200
   and 12,800.
5. **Arm R vs Arm B:** 400 games at 3,200 and 160 at 12,800.
6. Elo placement on the joint scale.

## Criteria, fixed now (identical to Arm L's)

Let *s* be R's score against Arm B at 3,200, *SE* its per-game standard
error, and *u* the endpoint-unique score.

| Verdict | Condition |
|---|---|
| **Labels help** | *s* − 1.96·*SE* > 50% **and** *u* > 50% **and** held-out mean ≥ 91.4% − 1.0 pp |
| **Labels hurt** | *s* + 1.96·*SE* < 50% |
| **Null** | anything else |

Reported but not deciding: the audit bias (expected to fall toward the gen51
control's), gate v4, per-colour scores and Elo.

Promotion is separate: a gate v4 PASS against v29 plus held-out
non-regression makes Arm R eligible, and the owner decides. Nothing is
promoted automatically.

## Conduct

- Driver `tools/gen52_ramp_campaign.py`; a tiny rehearsal runs first.
- One managed job; the web server keeps running.
- Results go in `docs/experiments/gen52/RAMP_RESULTS.md`.
- **Time guide:** relabelling and composing about 1 h; training about 5.5 h
  (Arm B's); evaluation about 9 h (Arm L's). About 15–16 h in all,
  re-estimated from the running job.
