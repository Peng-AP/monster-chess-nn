# Gen48 GPU data-recipe experiment — completed September 14, 2026

## Conclusion

**A repeatable, modest advantage over gen47, but no demonstrated broad,
both-color strength improvement. No promotion.**

Gen48 won both independently reserved, fresh-book matches against gen47.
Across their 512 games it scored 53.90625%, with a paired-opening bootstrap
95% interval of 52.1484–55.7617%. This is useful evidence of a head-to-head
advantage under this protocol. It does not establish better play against all
opponents, or proximity to perfect play.

Against the held-out B2 model, gen48 was essentially level with gen47's
performance: identical aggregate White score and a small, inconclusive Black
decline. Common-opening selfplay had exactly the same White score. Thus the
larger general-strength/Black-conversion objective remains unproven. This is
not grounds to claim that gen48 *only* exploits gen47: one held-out opponent
and 64 pairs also cannot establish that stronger negative claim.

Public release remains **v27/gen46**. Gen47 remains the frozen strength bar.
Gen48 is retained as a research candidate; CPU code/models remain intact and
CPU strength development remains paused. No deletion, commit, push, promotion,
or subsequent experiment was performed.

## Locked models and evidence

- Candidate: `models/candidates/bootstrap_main_gen_0048/arena_selected.pt`,
  epoch 17; SHA256 `a8c074390c93390ac974b1f58076a86aa0525d9a34f7cff66e12442bb7e07722`.
- Reference: `models/candidates/bootstrap_main_gen_0047/arena_selected.pt`,
  SHA256 `810297fe98807eb56cfe4184c3210048fed7dc93110c47dfb18c0564fb3865fb`.
- Held-out opponent: `models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt`,
  SHA256 `fc23076a9f5c7f237785f27cb1a665c10588ea8e8916cd743016d19a96999d15`.
- Authoritative results: `benchmarks/gpu48_20260913/production/summary.json`;
  SHA256 `5119c412816af00950c1af1c8ca67e2b9832bec2dde39eb912c6a6c2f379fbe6`.
- Preregistered plan: `GPU48_PLAN.md`; launcher: `tools/start_gpu48.py`.
- Production manifest SHA256: `ddc2d1f6fe3b5aa207a6b87bdbb4d9846ed5b607bcca877ce28ec45fbce899de`.
- Test book SHA256: `48388ed50fcd87b2f3074f11de2dbf67ef6cd3dafdb00163db42466f3520ff79`.

All 896 post-selection games ran at **3,200 simulations per player**, not an
equal-wall-time protocol. Book starts were generated once from gen47/v27/gen44
at 700 simulations, temperature 0.8, 16 half-plies, with new RNG namespaces.
The candidate did not generate its book. H2H legs use disjoint starts 0–127
and 128–255. Transfer and selfplay deliberately share starts 256–319; do not
pool those as independent tests. Book evaluation begins with fresh driver
history, unlike full-history training generation/reanalysis.

## Independent matches

Score is wins plus half the draws, divided by games. Each color has half the
listed game count. W/D/L columns are explicit to avoid the historical mixed
W-L-D/W-D-L notation.

| Match | Games | Wins | Draws | Losses | Score | As White | As Black |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gen48 vs gen47, first leg | 256 | 110 | 60 | 86 | 54.69% | 43.36% | 66.02% |
| Gen48 vs gen47, fresh confirmation | 256 | 112 | 48 | 96 | 53.13% | 35.55% | 70.70% |
| Gen48 vs B2 | 128 | 50 | 33 | 45 | 51.95% | 37.50% | 66.41% |
| Gen47 vs B2, same starts | 128 | 49 | 37 | 42 | 52.73% | 37.50% | 67.97% |

The first H2H leg's score interval is 52.15–57.42%; confirmation is
50.78–55.47%. Combined H2H is descriptive across the two disjoint planned
blocks; both separate results remain visible, not replaced by the aggregate.

**Do not judge individual colors against 50%.** Monster Chess has a large
color imbalance. The 50% H2H null applies to the balanced aggregate; the
summary's mechanical per-color `minus_50percent` fields are not tests of
color-specific improvement. Use the matched common-opponent comparison below.

| Gen48 minus gen47, versus B2 | Score difference | Paired 95% interval |
| --- | ---: | ---: |
| White | 0.00 pp | −6.25 to +6.25 pp |
| Black | −1.56 pp | −10.16 to +7.03 pp |
| Overall | −0.78 pp | −6.25 to +4.69 pp |

These are exploratory 10,000-resample, whole-opening-pair intervals, not
multiplicity-adjusted promotion tests. Neither a transfer improvement nor a
definite Black regression is established.

Post-hoc description of the saved games: on B2 White, 8 starts improved, 7
regressed and 49 stayed equal; on Black, 9 improved, 8 regressed and 47 stayed
equal. Improvement counts alone conceal different win/draw/loss magnitudes.
Gen48 Black gained one capture win but also three capture losses and four
fewer draws, leaving its Black score one point lower across 64 games.

The modest H2H gain was not confined to gen47-generated openings: across both
legs, gen48 scored 55.06% on v27 starts (168 games), 53.24% on gen44 starts
(170 games), and 53.45% on gen47 starts (174 games). These are descriptive
subgroups, not separately preregistered significance tests.

## Common-opening selfplay

One game per start, not duplicated color-swapped copies of the same self-game.

| Model | Games | White wins | Black wins | Draws | White score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gen48 | 64 | 11 | 33 | 20 | 32.8125% |
| Gen47 | 64 | 13 | 35 | 16 | 32.8125% |

Same color score, four more draws for gen48. This is a skew diagnostic, not a
strength proof or evidence of better Black conversion. Across all 896 final
games there were 207 White captures, 475 Black captures and 214 repetition
draws; **zero turn-cap or ply-cap endings**. Repetition draws are outcomes of
the existing rule, not proof that those positions are immutable fortresses.

## What was changed and trained

No architecture or playing-search change. Same 15-plane residual CNN and
attention policy as gen47, with its scalar value head; no new moves-left head,
SE blocks, CPU evaluator or inference heuristic.

- 64-game full-depth cost pilot: 64 saved, zero failures, 358.05 seconds.
  The declared sizing rule selected 3,200 production games rather than 4,800.
  Pilot games were not imported into training.
- Production: 1,600 free selfplay +800 stochastic-prefix games +400 balanced
  league games versus v25/v26/v27, all at 1,600 simulations; 400 completed
  full-history forks at 6,400 simulations. All 3,200 saved without failures.
  Forks share their original parent family; opponent policies remain masked.
- Reanalysis: 24,000 roots at 6,400 simulations; deterministic family-balanced
  sampling with a cap of 16 per transitive family. All 2,800 original families
  were represented; actual maximum 10, from 231,679 eligible rows.
- Sampling was 14,400 Black /4,800 White-first /4,800 White-second. Existing
  disagreement ranking retained 12,000 policy-only teachers from 2,783 families:
  7,200 Black /3,225 White-first /1,575 White-second. Actual outcomes, not deep
  search values, remain the value targets. B2 was absent from the new data recipe.
- Processed new data: 487,358 rows **including augmentation and 24,000 augmented
  teacher rows**, not that many independent positions. Data audit PASS; all
  12,000 teacher-family split links checked.
- Replay: generations 40,41,42,44,45,46,47 plus 48; no human anchor. Scratch
  seed 3173, LR 0.002, batch 256, warmup 3, EMA 0.999, value ramp 0.5/60,
  teacher policy multiplier 4. Maximum 30 epochs /patience 10; stopped at 25.
- Eight of 25 saved checkpoints received 40-game probes; three finalists
  received 200-game screens, plus their calibration games: 1,160 selection
  games total. Finalists were epochs 17,9,15. Offline-best was epoch 15, but
  epoch 17 won play-based nomination.

Selection deltas versus calibration were +40.5 pp for epoch 17, +26.25 pp for
epoch 9 and +13.0 pp for epoch 15. These model-influenced sampled-opening
selection scores did **not** translate into comparably large fresh-book gains.
The locked candidate was never replaced based on confirmation or B2 results.

This bundled data-recipe experiment cannot isolate the causal contribution of
more search, family coverage, changed teacher/mixture, or sample size.

## Execution and verification

Production ran September 13 12:42:42 to September 14 00:05:03 Eastern:
**11 hours 22 minutes**, without an execution failure or restart.

| Stage | Wall time |
| --- | ---: |
| Generate | 3h 46m |
| Reanalyze | 1h 16m |
| Process and compose | 3m |
| Train | 3h 15m |
| Checkpoint selection | 1h 49m |
| Fresh book +896 final games | about 1h 13m |

931 Python tests plus 3 subtests passed before production, with 172 existing
PyTorch deprecation warnings. Real rehearsal completed 28 generated games,
training/selection, and all six final-test branches (20 post-selection games).
Every one of the 896 production post-selection games passed a move/state/clock/
termination/outcome replay audit. At completion, the entire pinned provenance
still matched and all outputs in all eight stage receipts matched their hashes.

The existing common-selfplay reporter was corrected to score ±0.5 cap labels
as draws, matching match rules. The September 10 B2 confirmation's 13 logs
contained zero such outcomes; its historical scores were unaffected. No old
artifacts were rewritten.

Only one campaign GPU job ran, with at most eight workers. Task allocation
stayed within the 12 GiB target in observed samples. Total device usage later
rose to about 13 GiB with a separate graphics workload; campaign worker
counters totaled about 7.2 GiB then. Other applications were left alone.
Simulation budgets and recipes were unchanged. Contention affects wall-time
comparisons; these results are fixed-simulation comparisons.

Canonical `iterations/gen_0048/state.json` is deliberately **partial** after
checkpoint selection. The research driver is **complete**. This distinction
must not be relabeled as an official binding-gate pass. Managed
`gpu48_campaign` exited; no follow-up experiment is queued.

## September14 evaluation qualification and revised direction

All independent comparisons in this report used fixed book starts, including
B2 and self-play. These establish the reported book-conditioned scores, NOT
failure to transfer from the normal initial position. Normal-start free play
is the owner's primary instrument; the large free checkpoint-screen result
needs independent free confirmation, rather than dismissal based on books.

The owner has requested a mainline-focused follow-up. `GEN49_PLAN.md` supersedes
the proposed variety expansion below: finish gen48 free tests, then replace
fresh-prefix/older-league games with the new teacher's normal-start self-play,
holding the architecture and training settings fixed. Historical measurements
are not rewritten and gen48 has not been promoted.

## Prior next-step proposal — superseded above

1. Keep gen48 available for playtesting, but retain gen47 as the comparison bar
   and v27 as the release. There is no evidence here for a broad promotion.
2. Use the saved matched B2 regressions/improvements to investigate why gains
   fail to transfer, particularly the Black win-to-loss changes. This can
   start from existing logs, not another night of generation or a pawn rule.
3. Before another training block, define a genuinely reserved opponent/test
   panel. Test a less gen47-centric teacher/opponent mixture while holding
   architecture, search depths and training fixed. If B2 enters that training
   mixture, it must stop being called an untouched holdout; reserve a different
   model lineage and fresh evaluation starts first.
4. Do not respond to this result with another CPU rewrite or merely higher
   simulation counts. The next question is transfer of learned play, not
   whether still more work can improve the same selection matchup.
