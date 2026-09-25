# Corpus diversity audit — September 25, 2026

Stage 0 item 1 of `docs/plans/GEN51_STRENGTH_PLAN.md`. Read-only;
`tools/diversity_audit.py`, report
`benchmarks/gen51_program/diversity_audit_20260925/report.json` (about 2 minutes
of CPU for the gen49 and gen50 increments plus the 3.64M-row gen50 replay).

A position is keyed by the smaller hash of its 15-plane input and the
file-mirrored input, so a row and its mirror augmentation count once. With
augmentation, **2.0 value rows per distinct position means no repeats.**
Encoded inputs omit repetition history and turn count, so a few distinct game
states share a key. Material buckets count all pieces on the board; the start
has 21.

## Findings

**1. Repetition grows every generation, and fastest recently.**

| Source | Value rows | Distinct positions | Rows per position | Top-100 share | Top-1000 share |
|---|---:|---:|---:|---:|---:|
| gen42 | 132,482 | 59,351 | 2.23 | 5.7% | 10.0% |
| gen44 | 190,340 | 83,319 | 2.28 | 6.0% | 10.6% |
| gen45 | 394,106 | 165,669 | 2.38 | 6.4% | 10.6% |
| gen46 | 776,448 | 315,908 | 2.46 | 6.9% | 11.5% |
| gen47 | 783,298 | 289,681 | 2.70 | 8.1% | 13.5% |
| gen48 | 463,358 | 147,088 | 3.15 | 12.4% | 22.5% |
| gen49 | 445,516 | 138,969 | 3.21 | 13.2% | 23.8% |
| gen50 | 456,932 | 121,075 | 3.77 | 20.4% | 35.2% |

gen50 produced about 10% more value rows than gen49 from 13% fewer distinct
positions.

**2. It has spread from the opening into the middlegame.** Top-100 share by
material bucket:

| Pieces on board | gen49 | gen50 |
|---|---:|---:|
| 21–18 (opening) | 45.1% | 52.5% |
| 17–12 | 11.8% | 19.3% |
| 11–7 | 4.7% | **23.6%** |
| 6–2 | 5.9% | 10.8% |

Opening concentration is expected, since every game starts from one position.
A fivefold jump in the 11–7 bucket means gen50's games keep reaching the
*same middlegames*. That matches the narrow e4+d4 …d5 repertoire and the
low endpoint-unique gate scores (51.2% unique vs 58.4% sampled for v28).

**3. All three phases narrowed similarly** (gen50 rows per position: Black
3.78, White first half 3.91, White second half 3.64). It is not a one-colour
effect.

**4. New generations still reach new ground.** Only 8.4% of gen50's distinct
positions (10,202 of 121,075) already appear in gen42–gen49. Replay across
generations adds diversity even as each generation narrows.

**5. Conflicting outcomes are concentrated where expected.** 35% of gen50
value rows sit on positions with more than one recorded outcome, mostly
openings (77,744 of 119,724 opening rows). That is the normal cost of scoring
a shared opening by many games' results, not a label defect.

**6. Reanalysis adds no new positions.** All 24,000 gen50 policy-only teacher
rows sit on positions already present as value rows, by construction.
Reanalysis improves targets, not coverage.

## Decisions this triggers (predeclared in the plan)

- **Third training arm (duplicate-aware weighting): not triggered.** The
  predeclared condition was "most of the value loss concentrated in a small set
  of positions". In the 8-generation replay that training actually uses, the
  top 1,000 positions hold 13.9% of value rows and the top 100 hold 7.6%.
  Concentration is real but not "most". The arm stays specified and unrun.
- **Exploration lever: proposable.** Plan §5 allowed proposing it if the
  narrowing proved concentrated rather than benign. Findings 1–2 show it is
  concentrated, accelerating, and reaching the middlegame. It stays an
  **owner decision** and is not bundled into gen51 without approval. If
  approved, the minimal form would change only the gen51 self-play sampler
  (e.g. temperature for more moves, or a larger share of noisy decisions). It
  would not touch evaluation instruments or add forced openings, and would be
  measured by this same audit on the gen51 increment.
