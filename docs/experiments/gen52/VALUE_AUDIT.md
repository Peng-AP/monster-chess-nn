# Value colour audit (October 1, 2026)

Owner request: a diagnostic before any retraining on "final results instead
of evaluations". The value head already trains on game results; the open
question was whether **v29's own self-play outcomes taught it White
pessimism** that it passes to its students.

Tool `tools/value_colour_audit.py`; evidence
`benchmarks/value_colour_audit_20261001/report.json`.

## Method

- **Ground truth:** 8,427 rating games, from the round robin and both
  ladders, played by 16+ different models with whole trajectories saved.
  That gives 368,107 positions after the 16 sampled opening plies.
- Each model's raw value output (from White's point of view) is compared with
  the target it was trained to predict: the game result discounted as
  training discounts it, result × 0.5^(min(plies to end, 60)/60).
- **Bias** = mean(target − prediction). Positive means White did better than
  the model expected. Intervals come from a bootstrap over games.

## Results

Overall bias:

| Model | Bias (95%) | Without v29's games | MSE |
|---|---|---:|---:|
| v29 (teacher) | +0.036 (0.030–0.041) | +0.036 | 0.091 |
| gen52 Arm B | +0.059 (0.053–0.065) | +0.054 | 0.098 |
| gen52 Arm L | +0.064 (0.059–0.070) | +0.060 | 0.101 |
| v28 | +0.138 (0.132–0.144) | +0.149 | 0.136 |
| gen49 | −0.034 (−0.040 to −0.029) | −0.027 | 0.092 |
| B2 | +0.005 (−0.002 to +0.011) | +0.015 | 0.104 |

By distance to the end of the game:

| Model | 0–10 | 10–30 | 30–60 | 60+ |
|---|---:|---:|---:|---:|
| v29 | +0.077 | +0.102 | −0.011 | −0.118 |
| gen52 Arm B | +0.082 | +0.118 | +0.024 | −0.074 |
| gen52 Arm L | +0.084 | +0.121 | +0.033 | −0.062 |
| B2 | +0.084 | +0.079 | −0.064 | −0.173 |
| gen49 | +0.059 | +0.045 | −0.112 | −0.224 |

v29 calibration (predicted vs realised):

| Predicted | Realised | Positions |
|---:|---:|---:|
| −0.90 | −0.80 | 110k |
| −0.72 | −0.58 | 66k |
| −0.10 | −0.16 | 68k |
| +0.29 | +0.15 | 6k |
| +0.90 | +0.87 | 24k |

## Reading

1. **There is no large uniform White pessimism.** v29's overall bias is
   +0.036 in value units, about 1.8 points of expected score.
2. **The dominant pattern is shared by every model, B2 included:**
   - Within 30 plies of the end, they predict Black's win as more certain
     than it turns out (+0.08 to +0.12).
   - More than 60 plies out, they are *more optimistic for White* than the
     outcomes (−0.06 to −0.22).
   - This is overconfidence near the end, not a v29-specific colour bias.
3. **Students are a little more pessimistic than their teacher.** gen52
   Arms B and L sit about +0.025 above v29 overall and about +0.04 in the
   30–60 band. The direction matches the hypothesis, but the size is small.
4. **v28 is markedly pessimistic about White (+0.14).** v28 is the
   "calibrated value head" release. Its calibration evidently did not
   transfer to games played by other models.

**Verdict:** weak support for the teacher-pessimism hypothesis. The effect
that passes from teacher to students is real but small (~0.02–0.04). The
larger miscalibration is overconfidence late in games, which every model
shares.
