# Promotion rule (adopted September 29, 2026)

**Why:** on September 29 the owner reported that the models now exceed his
ability to judge strength by playtest. The earlier requirement of "automated
evidence plus the owner's playtest" therefore becomes "automated evidence
meeting this fixed standard, plus the owner's approval of a short evidence
summary". v29 is the first release under it.

## Standard a candidate must meet against the current release

1. **Gate v4 PASS** (`tools/gate_depth.py`, `free_sampled_depth_guard_v4`),
   with the release as the bar:
   - two fresh 400-game legs at 3,200, each above 50%;
   - each colour at least the release's self-par − 5 pp;
   - the 12,800 guard: at least 47.5%, and each colour at least the release's
     deep self-par − 10 pp.
2. **Held-out non-regression:** the candidate's mean 3,200-simulation score
   against the held-out opponents is at least the release's mean − 1.0 pp,
   measured the same way (160 games each, normal start). Currently the
   held-out opponents are **B2 and v27**. Models used as training opponents
   are never held-out opponents.
3. **Independent evidence only:** selection and screen games never count
   toward 1 or 2.

Then the owner approves from a one-page summary: both gates, per-colour
deltas, held-out and pool-opponent results, and known weaknesses (e.g.
specific opening lines).

## What does not change

- Nothing is promoted automatically. A PASS makes a candidate eligible; it
  does not promote it.
- Thresholds are never relaxed after seeing results.
- Releases are immutable copies with hash-pinned manifests; nothing is
  overwritten. The champion pointer and the gate bar move together.
- Head-to-head scores are non-transitive; no Elo ladder is inferred from
  chained matches.

## v29 against this rule

| Criterion | v29 (gen51 deep-value) | Release v28 | Met |
|---|---|---|---|
| Gate v4 vs v28 | PASS (74.1% / 75.75%; 85.0% at 12,800) | — | yes |
| Held-out mean (B2, v27) | 92.7% (99.06 / 86.25) | 82.3% (73.75 / 90.94) | yes |
