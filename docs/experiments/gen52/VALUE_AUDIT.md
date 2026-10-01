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

## Strong-only check (confound control)

Weak players (v17–v23) fail to convert won positions, which alone makes a
strong model look overconfident. Restricting to games where both players
rate 2150+ (v27 and up, B2, the gen49/gen52 models, v29 at 200+ sims) leaves
2,728 games and 147,726 positions:

| Model | Bias (95%) | 0–10 | 10–30 | 30–60 | 60+ |
|---|---|---:|---:|---:|---:|
| v29 (teacher) | **+0.017** (0.009–0.024) | +0.089 | +0.102 | −0.031 | −0.168 |
| gen52 Arm B | **+0.063** (0.054–0.072) | +0.109 | +0.129 | +0.033 | −0.085 |
| gen52 Arm L | **+0.071** (0.062–0.079) | +0.114 | +0.135 | +0.042 | −0.077 |
| v28 | +0.097 (0.089–0.105) | +0.117 | +0.161 | +0.083 | −0.061 |
| gen49 | −0.080 (−0.088 to −0.072) | +0.054 | +0.023 | −0.156 | −0.308 |
| B2 | −0.044 (−0.052 to −0.035) | +0.081 | +0.072 | −0.124 | −0.281 |

## Verdict (revised by the strong-only check)

1. **The teacher is not the pessimist; its students are.** Against strong
   games, v29 is nearly unbiased (+0.017). Both gen52 students sit about
   +0.05 more pessimistic about White, and the intervals do not overlap.
   That is roughly 2.5 points of expected score, in the direction of every
   gen52 arm's White failure.
2. ~~This fits the outcome-label pathway.~~ **Withdrawn the same day.** That
   claim compared a win rate with a score. gen52's self-play White *score* is
   33.6% (21.3% wins + 24.6% draws ÷ 2). The qualifying strong rating games
   (no B2/v27, not v29 against itself; 1,744 games) score **32.3%** for
   White. The teacher's self-play outcomes are not more Black-favoured, so
   training on rating-game outcomes is not expected to help, and it was not
   run.
3. **Late-game overconfidence is separate and shared** (+0.08 to +0.14
   within 30 plies of the end, every model, strong games included). It is a
   real calibration issue, but it is not the gen52 regression.

## Where the bias comes from: the deep-value label scale

All arms were audited on the same strong games:

| Model | Deep-value share of value weight | Bias (95%) | 60+ plies from end |
|---|---:|---|---:|
| gen51 control | 0% | −0.040 (−0.047 to −0.032) | −0.231 |
| v29 (gen51 deep-value) | 13.0% | +0.017 (0.009–0.024) | −0.168 |
| gen52 Arm C | 12.4% | +0.049 (0.039–0.058) | −0.119 |
| gen52 Arm B (+ pool) | 22.3% | +0.063 (0.054–0.072) | −0.085 |
| gen52 Arm A | 22.3% | +0.075 (0.067–0.083) | −0.073 |

- **The White pessimism tracks the deep-value share.** Within gen51, adding
  the deep-value data moved the bias by +0.056. Within gen52, the pool games
  dilute it (B below A).
- **Mechanism:** deep-value rows carry strict capture-only results with **no
  distance discount** (`process_linked_extra.py`, value floor 1.0), at value
  weight 4. Everywhere else, a result 60+ plies from the end trains toward
  half strength.
  - Most results are Black wins, so the undiscounted rows pull predictions
    toward "Black wins", most of all far from the end.
  - That is exactly where the shift is largest, and the 60+ column falls in
    the same order as the deep-value share.
  - It is a label-scale mismatch between two sources, not learned pessimism
    from self-play.
- **It does not explain the gen52 White gate failures on its own.** Arm C has
  less bias than Arm B but the same White deficit against v29 (−15.6 vs
  −13.4 pp). The same data also produced v29's +95 Elo gain.

**Candidate next experiment (not started; owner's call):** Arm B with the
deep-value increments relabelled on the main ramp (floor 0.5, horizon 60)
instead of strict undiscounted captures. It is training-only and tests
whether consistent label scales keep the deep-value gain without the shift.
