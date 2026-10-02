# Overnight chain, October 2, 2026 (declared before Arm R's verdict)

Owner, 01:30: "make sure the next at least 8, preferably 16 hours are
utilized." Everything below runs sequentially from one chain
(`one heavy GPU job at a time`) after gen52 Arm R completes. Nothing is
promoted.

## Step 1: top-group round robin (evaluation only)

`tools/top_round_robin.py`

- **Players:** v29, v28, gen49 and gen52 Arms A, B, C, L, R.
- **Format:** 28 pairings × 100 games (50 per colour) at 3,200 simulations,
  sampled openings, in a shuffled order. Seeds start at 3.90e9.
- **Output:** Bradley–Terry Elo relative to v29 = 0, with 1,000-rep
  bootstrap intervals, plus the full head-to-head matrix.
- **Question:** is any candidate stronger than v29 against this group, and
  by how much? It is descriptive and decides nothing.

## Step 2: one training experiment, chosen by Arm R's pre-declared verdict

| Arm R verdict (`GEN52_RAMP_PLAN.md`) | Step 2 |
|---|---|
| `labels_help` | **Arm LR**: wide tower (Arm L) on Arm R's relabelled replay |
| `null` or `labels_hurt` | **Arm L2**: Arm L repeated with training seed 3174 instead of 3173 |

Driver `tools/gen52_variant_campaign.py --variant lr|l2`.

- **One change each.** Each variant is the receipted train command of its
  parent arm with only these flags changed: LR changes data and model
  directories (parent: Arm L); L2 changes seed and model directory (parent:
  Arm L).
- **Evaluation is identical to Arms L and R:**
  - checkpoint screen against v29 and 12,800 probes (gen52 seeds);
  - gate v4 against v29;
  - diagnostics (B2 and v27 held out, gen49, v28, self-play);
  - Elo placement.
- **Head-to-head:**
  - Primary, 400 games at 3,200 plus 160 at 12,800:
    - LR against Arm L ("do consistent labels help the wide net?");
    - L2 against Arm B ("does the capacity result replicate?").
  - Secondary, 400 games at 3,200: LR against Arm R; L2 against Arm L.

**Verdict (same rule as Arms L and R).** Let *s* be the primary score,
*SE* its per-game standard error, and *u* the endpoint-unique score. The
comparator is the arm the primary head-to-head plays against.

- **Helps:** *s* − 1.96·*SE* > 50%, *u* > 50%, and held-out mean ≥ the
  comparator's held-out mean − 1.0 pp.
- **Hurts:** *s* + 1.96·*SE* < 50%.
- **Null:** anything else.

Comparator held-out means (B2, v27): Arm B 91.4%, Arm L 92.5%.

## Expected time

| Step | Time |
|---|---|
| Arm R remainder | until about 05:30–06:30 |
| Round robin | about 4 h |
| Variant | training ~6 h + evaluation ~9.5 h |
| **Chain end** | about 01:00 on October 3 |

Estimates are re-checked from the running jobs.
