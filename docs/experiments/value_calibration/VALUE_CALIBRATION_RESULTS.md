# Frozen-policy value calibration: completed results

Experiment completed **September 17, 2026, at 09:16:27 Eastern**. This report
was assembled from the preserved artifacts and checked on September 25.

## Conclusion

The continuation-calibrated value head produced a repeatable improvement over
its unchanged gen50 epoch14 initialization at the ordinary 3,200-simulation
budget: **58.4375% over 800 independent head-to-head games**, with both planned
legs above 50%. That improvement did **not** carry through at 12,800 simulations:
the deeper comparison scored **49.375% over 160 games**. Results against older
opponents also do not establish a general upgrade.

This is a useful value-calibration result, not a demonstrated all-budget,
both-color successor. Nothing was promoted or copied over an existing model.
Gen50's `arena_selected.pt` remains epoch14; public release remains v27/gen46.
The campaign is complete, not waiting for more games. It queued no subsequent
experiment.

Primary evidence:

- `benchmarks/value_calibration_20260917/production/summary.json`
- `benchmarks/value_calibration_20260917/production/status.json`
- `benchmarks/value_calibration_20260917/production/nominee.json`
- `benchmarks/value_calibration_20260917/production/confirmation/play/`
- Frozen design: `VALUE_CALIBRATION_PLAN.md`
- Implementation: `run_value_calibration.py`, `value_calibration.py`,
  `test_value_calibration.py`
- Archived launcher log after the September 25 cleanup:
  `logs/archive/2026-09-17/value_calibration.log`

Production launched September 17 at approximately 02:19:51 and took about
6 hours 57 minutes. It completed all **3,696 scheduled games**: 576 fresh
data-generation games, 960 screening games, and 2,160 confirmation games.
Rehearsal games are separate and are not pooled into those results.

## What was changed, and what was held fixed

The preceding checkpoint-recovery experiment localized an ordinary-budget
opening problem to an interaction between value estimates and search. It did
not establish that a particular move should be banned, that gen49 values were
globally correct, or that more search always corrected the behavior. See
`GEN50_RECOVERY_RESULTS.md` for the policy/value crossover evidence.

This experiment tested three arms:

1. **Baseline:** unchanged gen50 epoch14.
2. **Replay:** the same checkpoint, fitting only its existing scalar value head
   on sampled replay outcomes.
3. **Continuation:** the same fitting procedure, with half of each batch drawn
   from fresh completed deeper continuations and half from replay.

Architecture, search settings, policy head, backbone, and every non-value
parameter/buffer were held fixed. The backbone ran in evaluation mode; its
global-average-pooled features were cached for inexpensive head-only fitting.
Exported candidates are ordinary loadable checkpoints, not mixed search engines.
The fit artifacts record exact non-value parameter/buffer equality and exact
raw-policy equality for both arms. The prelaunch rehearsal included native
loading, cached/full-forward parity, and frozen-tensor checks.

Both trained arms also used strict capture-outcome replay labels instead of
the original distance-tempered targets. This shared label change is controlled
between the two fitted arms, but prevents attributing all change from the
unchanged baseline solely to the new continuation data.

### Frozen references and candidates

| Role | Checkpoint | SHA256 |
| --- | --- | --- |
| Initialization / unchanged baseline | `models/candidates/bootstrap_main_gen_0050/arena_selected.pt` | `51b5ddb01db51ae9023eaaf8ccbd896b48a805a52b2707633dc1d7e3f8067f25` |
| Continuation opponent / gen49 reference | `models/candidates/bootstrap_main_gen_0049/arena_selected.pt` | `4bcc68a0219acf8c3dc53326d738789e6bd767fccba88fd4f471345567b4a647` |
| Older B2 diagnostic opponent | `models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt` | `fc23076a9f5c7f237785f27cb1a665c10588ea8e8916cd743016d19a96999d15` |
| Public v27 opponent | `models/bootstrap_v27/best_value_net.pt` | `976294daf7e3d6f0c51c358dd602f11997c7fdf2dc4b255b810b588c253e5459` |
| Replay-only fit, selected epoch10 | `benchmarks/value_calibration_20260917/production/fits/replay/candidate.pt` | `98a68b982a0a30f1dc928fdb40944e6d4b799bba9f3dce2024fefe6dab6f916b` |
| Continuation fit, selected epoch12 | `benchmarks/value_calibration_20260917/production/fits/continuation/candidate.pt` | `b651e7405afe4e5672676c6fb13bb5bc6e1f5ea0071fd5ce4f4d5318e229e35a` |

The new candidates live under **benchmarks**, not `models`. Do not delete that
directory as disposable benchmark output. Current notebook model discovery does
not automatically expose these research paths; preservation is not promotion.

## Data and fitting recipe

The experiment generated 192 fresh normal-start epoch14 selfplay games at
3,200 simulations. From each parent it selected one eligible root between
primitive plies 4 and 120, using generic model/search disagreement rather than
named moves or favorable outcomes. Root phases were balanced as 96 Black,
48 White-first-half, and 48 White-second-half positions.

Every root received two completed continuations at 6,400 simulations:
epoch14 White versus gen49 Black, then gen49 White versus epoch14 Black.
These supplied 384 continuation games, for 576 fresh games in total. Separate
per-color trees and full driver history were restored. This does not eliminate
the native search's existing limited-history representation.

Labels were completed outcomes: actual king captures were +1/-1 and
repetition/turn-cap draws were 0, converted to the actual side-to-move
perspective. In particular, White's second half-turn remains White's
perspective. Outcomes describe these frozen players; they are not proofs of
optimal play, and a drawn continuation is not a proven fortress.

The 192 parent families were split 144/24/24 into training/validation/test.
A parent and its two children stayed in the same family. At most eight evenly
spaced positions per phase per game were extracted. Existing gen50 replay
contributed 32,768 sampled training rows and 4,096 sampled validation rows from
the original split, with positive value weight and strict `capture_results`
labels.

Exact encoded inputs were isolated across fitting and held-out pools, with
priority `new_test > new_val > old_val > train`. Within each pool, duplicates
were merged and conflicting outcomes averaged. Old and new training pools
could share inputs because both are training inputs. This separation applies
to the incremental fit: it cannot make the held-out positions historically
unseen by the already pretrained epoch14 model.

Effective data after those exclusions and duplicate merges:

| Pool | Input rows | Rows excluded by higher-priority pools | Retained unique inputs | Conflicting-outcome inputs |
| --- | ---: | ---: | ---: | ---: |
| New training | 9,872 | 6,424 | 1,833 | 24 |
| Old training | 32,768 | 2,961 | 27,995 | 346 |
| New validation | 1,617 | 848 | 511 | 10 |
| Old validation | 4,096 | 89 | 3,844 | 42 |
| New test | 1,691 | 0 | 1,069 | 19 |

Source: `production/data/complete.json`. The large reduction in usable new
training inputs is important: 576 generated games did not provide 576
independent strategic families or thousands of wholly novel positions.

Both fits used AdamW, learning rate 1e-4, weight decay 1e-4, seed26017,
12 epochs of 128 updates, and batch size512. Sampling was 50% Black and 25%
each White half. The continuation arm used 256 replay and 256 new examples per
batch. Loss was outcome MSE plus a 0.1 MSE anchor to the original prediction;
gradient norm was clipped to1. Lowest mean phase-balanced validation MSE across
old and new validation pools selected the checkpoint. No per-epoch game gate
or offline-only arm rejection was used.

Held-out new-test MSE, measured after checkpoint selection:

| Arm | Selected epoch | Overall MSE | Black | White first half | White second half |
| --- | ---: | ---: | ---: | ---: | ---: |
| Replay | 10 | 0.1812203 | 0.1888464 | 0.1659065 | 0.1885652 |
| Continuation | 12 | 0.1620679 | 0.1696841 | 0.1467911 | 0.1693865 |

Source: `production/fits/{replay,continuation}/complete.json`. Lower MSE here
means better prediction of these completed-game labels, not demonstrated
proximity to perfect-play values or guaranteed improvement at every search
budget.

## Screening: all arms received games

Each cell was an 80-game normal-start match at 3,200 simulations, 40 per color.
Opponent seed blocks were shared across arms; changed policies/values do not
guarantee the resulting trajectories were identical. Scores count draws as half.

| Arm | Opponent | Overall | White | Black |
| --- | --- | ---: | ---: | ---: |
| Baseline | gen49 | 78.75% | 61.25% | 96.25% |
| Baseline | epoch14 | 46.25% | 32.5% | 60% |
| Baseline | v27 | 93.125% | 86.25% | 100% |
| Baseline | B2 | 76.875% | 83.75% | 70% |
| Replay | gen49 | 78.75% | 70% | 87.5% |
| Replay | epoch14 | 43.125% | 31.25% | 55% |
| Replay | v27 | 91.25% | 82.5% | 100% |
| Replay | B2 | 85% | 97.5% | 72.5% |
| Continuation | gen49 | 85.625% | 73.75% | 97.5% |
| Continuation | epoch14 | 51.25% | 36.25% | 66.25% |
| Continuation | v27 | 95% | 91.25% | 98.75% |
| Continuation | B2 | 76.875% | 87.5% | 66.25% |

The baseline-versus-epoch14 row is an arbitrary model-A-role split of equal
weights, not an actual-color self-skew measurement.

Predeclared eligibility required overall score against epoch14 of at least50%,
Black score against gen49 no more than five percentage points below baseline,
and neither older-opponent White score below baseline. Only **continuation**
qualified. The replay arm's strong B2 White screen did not override its other
failures. Thresholds were not changed after observing scores.

Screen results select a nominee; they are not independent confirmation.

B2 and v27 were used during this screening and had been studied previously.
Their later matches use independent seed blocks, but they are not blind or
previously unseen opponents.
Source: `production/screen/play/*.json` and `production/nominee.json`.

## Independent confirmation

All rows below concern the continuation nominee. Normal-start sampling used
temperature0.5 for the first16 primitive plies, then0, with no book starts or
new search-default changes. Both players used the listed simulation budget.

| Opponent / budget | Games | Overall | White | Black |
| --- | ---: | ---: | ---: | ---: |
| Epoch14 / 3,200, first leg | 400 | 56.5% | 42.75% | 70.25% |
| Epoch14 / 3,200, fresh confirmation | 400 | 60.375% | 49.5% | 71.25% |
| Epoch14 / 3,200, combined | 800 | 58.4375% | 46.125% | 70.75% |
| Gen49 / 3,200 | 160 | 83.125% | 70% | 96.25% |
| v27 / 3,200 | 160 | 90.9375% | 81.875% | 100% |
| B2 / 3,200 | 160 | 73.75% | 83.75% | 63.75% |
| Gen49 / 12,800 | 160 | 56.875% | 48.75% | 65% |
| Epoch14 / 12,800 | 160 | 49.375% | 41.25% | 57.5% |

The sampled epoch14 gate **passed**. Its separate 400-game epoch14 selfplay
calibration is part of the 2,160 confirmation games, not part of the nominee's
800-game H2H denominator. A PASS is the declared operational screen, not proof
of both-color statistical non-inferiority against every opponent.

Combined ordinary-budget epoch14 W/D/L:

- As White: 128 wins, 113 draws, 159 losses.
- As Black: 213 wins, 140 draws, 47 losses.

Deeper epoch14 W/D/L:

- As White: 2 wins, 62 draws, 16 losses.
- As Black: 18 wins, 56 draws, 6 losses.

Deeper gen49 W/D/L:

- As White: 1 win, 76 draws, 3 losses.
- As Black: 24 wins, 56 draws, 0 losses.

Repeated opening trajectories are part of the declared frequency-weighted
instrument. They are not independent strategic structures. Do not convert the
nominal H2H Elo estimate into a universal strength or perfect-play claim.

### Actual-color selfplay

At 3,200 simulations, 160 nominee self-games produced **37 White wins,
73 Black wins, and 50 draws**. Actual White score was **38.75%**, actual Black
score61.25%. The report's arbitrary model-A score53.75% is not color skew.

The earlier unchanged epoch14 selfplay sample had White score34% over200 games.
These different samples are descriptive, not a controlled proof of changed
intrinsic color balance or an objective target of50:50.

## Interpretation and boundaries

The 51.25% screening result against epoch14 was small, but the reserved
independent legs confirmed a larger ordinary-budget advantage. Conversely,
the older-opponent White improvements suggested by screening did not reliably
transfer to the independent block. Earlier epoch14 results were White84.5%
against v27 and91.5% against B2; the new samples were81.875% and83.75%.
Those comparisons use different seeds and sample sizes, not matched causal
deltas. They nevertheless prevent an unqualified broad-strength claim.

The 12,800-simulation epoch14 result is essentially even. Against deep gen49,
the nominee did not show epoch15's extreme White collapse, but that is not the
same as establishing a deep-search upgrade over epoch14. Value calibration can
alter search choices substantially while leaving raw policy identical; it does
not make imperfect rollout outcomes ground truth.

A future proposal could test a more conservative value update or more reliable
repeated continuation labels, with unchanged-policy controls and independent
older-opponent/deeper-budget checks. That is a possible next study, **not an
authorized or queued continuation of this completed campaign**. Nothing here
supports hard-coded opening moves, calling repeated draws fortresses, or an
architecture rewrite by itself.

## Verification and preservation

Recorded prelaunch rehearsal: 983 tests plus3 subtests, 120 real rehearsal
games, both tiny fits, all26 rehearsal receipts, clean resume, native loading,
and frozen-policy verification. Those are historical recorded checks, not tests
rerun on September25.

Fresh September25 read-only PowerShell SHA256 verification found **zero missing
files or mismatches** across:

- 25 production receipts covering82 output hashes.
- Four frozen reference model files.
- 32 runtime files, including `native/monster_native.pyd`.
- 286 pinned implementation files, with no additional `tools/*.py` or
  `tests/*.py` absent from the manifest.
- The three smaller replay inputs: `capture_results.npy`, `value_weights.npy`,
  and `splits.npz`.

The 15,169,843,328-byte replay `positions.npy` was deliberately **not rehashed**
in this bounded audit. No fresh full test suite, game replay legality audit,
GPU inference run, or independent checkpoint tensor comparison was performed
on September25. The fit receipts' frozen-tensor/policy assertions were verified
as unchanged artifacts, not recomputed. Do not describe this bounded audit as a
new complete runtime/data verification.

Completed stages, evidence journals, candidate weights, prepared data, and
source pins remain valuable. `run_value_calibration.py` hashes all top-level
Python files under `tools/` and `tests/`, the three calibration scripts and
frozen plan, runtime files, reference checkpoints, and original replay inputs.
Moving or editing those pinned files can invalidate exact resume, even if the
experiment itself is complete. This new results document is not a frozen input.
