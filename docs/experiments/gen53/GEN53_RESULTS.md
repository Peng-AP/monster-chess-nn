# Gen53 results (teacher: gen52 Arm R), completed October 5, 2026

Plan `docs/plans/GEN53_PLAN.md` (owner revision of October 4: teacher by
`TEACHER_SELECTION_PLAN.md`; rule 1 relaxed). Driver `tools/gen53_campaign.py`.
Production ran October 4 01:13 → October 5 00:30 as one managed job, no
failures. The release gate (chain `tools/gen53_release_gate_chain.py`)
followed, 00:31 → 05:14. Evidence:
`benchmarks/gen53_program/gen53_20260928/{production,vs_release}/`.
**Nothing was promoted.**

## Recipe

- **Teacher:** gen52 Arm R.
- **Self-play:** 2,800 games at 1,600 simulations with 30-ply exploration;
  400 forks; reanalysis 24k → 12k at 12,800.
- **Deep-value data:** 768 roots × 2 at 6,400, reference v29.
  - Relabelled to the main ramped game results (Arm R's recipe).
  - Only the gen52 and gen53 deep increments are in the replay, giving
    **21.9%** of value weight (Arm R: 21.6%).
- **Arms:** single arm, because the pool was dropped by rule 3.
- **Training:** replay gen46–gen53, scratch, seed 3173, the 1.9M tower.
- **Nominee:** epoch 15. All three finalists beat Arm R with both colours on
  the 3,200 screen. The nominee's 12,800 probe against Arm R was 51.25%;
  epochs 11 and 13 probed 66.25% and 64.4%.

## Gate v4 against its teacher (Arm R): PASS

| Leg | Score | White W/D/L | Black W/D/L | Distinct games only |
|---|---:|---|---|---:|
| vs_bar @3,200 (400) | 66.1% | 26/156/18 | 123/75/2 | 59.8% |
| vs_bar_confirm @3,200 (400) | 67.1% | 36/150/14 | 119/77/4 | 61.5% |
| deep guard @12,800 (160) | 51.9% | 3/76/1 | 4/76/0 | 60.8% |

Against Arm R's self-par, White is **+18.75 / +22.25 pp** and Black +13.5 /
+12.0 pp. This is the first student this cycle that improved its teacher's
White.

## Gate v4 against the release (v29): PASS

The reused v29 par from gen52 was rejected by `gate_depth`'s runtime-identity
check (the code or native build changed since September 27), so v29's par was
measured afresh: White 48.9% at 3,200 and 20.0% at 12,800.

| Leg | Score | White W/D/L | Black W/D/L | Distinct games only |
|---|---:|---|---|---:|
| vs_bar @3,200 (400) | **53.25%** | 3/182/15 | 38/162/0 | 62.9% |
| vs_bar_confirm @3,200 (400) | **52.6%** | 4/181/15 | 32/168/0 | 69.8% |
| deep guard @12,800 (160) | **62.5%** | 0/76/4 | 44/36/0 | 57.5% |

Deltas against v29's par:

| Leg | White | Black |
|---|---:|---:|
| vs_bar | −1.9 pp | +8.4 pp |
| vs_bar_confirm | −1.6 pp | +6.9 pp |
| deep guard | +27.5 pp | −2.5 pp |

All within the floors. Combined score 52.9%, or 64.2% counting distinct
games once.

**gen53 is the first candidate to pass gate v4 against v29.**

## Diagnostics (3,200 simulations, 160 games)

| Opponent | gen53 | Arm R (teacher) | v29 |
|---|---:|---:|---:|
| B2 (held out) | **98.4%** | 90.0% | 99.06% |
| v27 (held out) | **78.1%** | 93.75% | 86.25% |
| **Held-out mean** | **88.3%** | 91.9% | **92.7%** |
| gen49 | **89.7%** | 66.6% | 75.6% |
| v28 | **84.7%** | 74.1% | 74.9% |

Self-play is very drawish:

| Simulations | White W/D/L | White score |
|---:|---|---:|
| 3,200 | 0/181/19 | 45.25% |
| 12,800 | 0/155/5 | 48.4% |

**Value audit** (strong games): gen53 −0.006 (−0.014 to +0.002), Arm R
−0.005, v29 +0.017. gen53 is unbiased.

## The v27 dip is one endgame gen53 misplays

| | Score vs v27 |
|---|---:|
| Counting every game | 78.1% |
| Counting each distinct game once (28 games) | **90.9%** |

- **All 33 of gen53's losses came as White, 32 of them from one position:**
  `rnbqkb1r/ppp2npp/4P3/8/8/4K3/2P5/8 w kq - 0 6`.
- gen53's White opening funnels into it by six move orders (e4+d4 …d5,
  Ke2/Kd2–e3, …dxe4 f3, d5 …e6, e5 …Nh6, dxe6, exf7 Nxf7, e5–e6).
- From the same position gen53 drew v28 22/22 and gen49 7/7.

**Position probe** (`tools/position_probe.py`; v27 as Black, 10 games per
side with 4 sampled plies):

| White | Result as White | As Black (v27 White) |
|---|---|---|
| **Arm R** | **10 W** | 10 D |
| **v29** | **10 W** | 10 D |
| Arm LR | 10 D | 10 D |
| Arm L | 10 D | 10 D |
| v28 | 10 D | 10 D |
| **gen53** | **10 L** | 10 D |

**The position is a win for White with good play.** gen53's opening choice
is sound; its **endgame play from this position is broken**, while its own
teacher wins it. That points at a narrow gap in gen53's training data (an
endgame type its self-play rarely reached or mislabelled), not at the
opening. v27 is the only opponent that punishes it.

## Promotion eligibility (`docs/protocols/PROMOTION_RULE.md`)

| Criterion | gen53 | Required | Met |
|---|---:|---|---|
| Gate v4 PASS vs release v29 | PASS | PASS | **yes** |
| Held-out mean (B2, v27) | 88.3% | ≥ 92.7% − 1.0 = 91.7% | **no** |

**Not eligible for promotion** under the current rule, solely because of the
v27 endgame hole. Nothing was promoted. The owner said on October 5 that
promotion is not required.
