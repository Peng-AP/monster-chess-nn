# v28 search-constant check — completed September 25, 2026

Stage 1 of `docs/plans/GEN51_STRENGTH_PLAN.md`. Driver
`tools/search_constants_campaign.py`, evidence
`benchmarks/gen51_program/search_constants_20260925/production/`
(`summary.json`, `nominee.json`, per-arm journals). The full rehearsal passed
first. Production ran 14:51–18:14 Eastern (203 minutes), 2,160 games, all
replay-audited.

## Result: null. The defaults stand.

Same v28 network on both sides; only the candidate's constants move. 400 games
per arm at 3,200 simulations, sampled normal start. Colour deltas are against
v28's own actual-colour self-par at 3,200 (400 games: White 31.75%, Black
68.25%), with nominal 95% half-widths.

| Arm | c_puct | FPU | Overall | Endpoint-unique | White vs par | Black vs par |
|---|---:|---:|---:|---:|---:|---:|
| A | 1.0 | 0.30 | 49.50% | 50.00% | +2.0 ± 6.5 pp | −3.0 ± 6.5 pp |
| B | 2.0 | 0.30 | 46.00% | 45.09% | −2.8 ± 6.2 pp | −5.2 ± 6.6 pp |
| C | 1.5 | 0.20 | 47.13% | 48.85% | +0.5 ± 6.4 pp | −6.2 ± 6.5 pp |
| D | 1.5 | 0.40 | 48.38% | 51.69% | +1.0 ± 6.2 pp | −4.2 ± 6.8 pp |

The fixed nomination rule required overall ≥ 52.5% and both colours ≥ par
− 5 pp. **No arm qualified, so no confirmation ran** (as predeclared). Every
arm scored below 50% against the defaults; c_puct 2.0 was the weakest, about
two nominal SE below even. Screens were not designed to prove the defaults
optimal, but no one-factor move in either direction helped. The engine
defaults (c_puct 1.5, FPU 0.30) are unchanged, and so is the runtime identity.

## By-products

- **v28 self-par, reusable:** at 3,200, White 31.75% (63W/128D/209L of
  400); at 12,800, White 45.62% (14W/118D/28L of 160). Gen51 reuses this par
  through gate v4's `--par-dir` check, which accepted it because the runtime
  did not change.
- **Throughput at 3,200 simulations:** 400 games in about 31–37 minutes on
  8 workers (for sizing later stages).
