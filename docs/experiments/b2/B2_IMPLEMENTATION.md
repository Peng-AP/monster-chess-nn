# B2 implementation — September 8

Implemented the prerequisites for the three-arm generation/training experiment.
This is not a claim of a stronger model or completion of the full B2 evaluation.

## Runtime

- Control remains the original15-channel,1,909,699-parameter CNN.
- State CNN adds the24-channel encoding,1,914,883 parameters.
- Hybrid adds two four-head64-square attention blocks with pre-LayerNorm and
  two-times-width MLPs after the CNN tower,2,179,843 parameters. The policy/value
  ABI stays unchanged. No WDL, moves-left utility, tactical rule or search-budget
  change is included.
-24 channels: original17-channel layout plus signed file coordinate, cleaned
  K/Q/k/q rights, raw en-passant square and remaining turn budget /150. Requires
  explicit turn_count. Python inference, processing and Rust leaves propagate it.
  Repetition/history are deliberately not encoded. Both processed arms disable
  mirror augmentation. Old15/17-channel encodings remain supported unchanged.
- New state models carry a persistent encoding/architecture schema buffer; the
  loader rejects missing or unsupported schemas rather than guessing.
- Rust rebuilt successfully. Windows held the prior native library open, so it
  was renamed to native/monster_native.pre_b2_20260908.pyd and preserved before
  installing the new binary. Existing processes retain their loaded version;
  restart a notebook kernel before testing B2 checkpoints. No process was killed.

## Data/training launcher

tools/b2_campaign.py generates two equally sized teacher contributions: v27 and
the broadly selected gen47 checkpoint. Total12,000 completed games:6000free,
3000league,1500fresh-prefix,1500fork. League teacher colors balanced, fork roots
60%Black; one fork per source family. Fork selection mixes uniform sampling with
value/outcome surprise, a generic disagreement proxy, not a tactical heuristic.
Ordinary700/forks3200 sims. Each teacher reanalyzes40k positions and retains20k.

tools/reanalyze_b2.py retains the state snapshot in deep teacher rows. The merge
namespaces both source games and parent pointers. tools/b2_prepare.py audits
every intermediate position by replay and splits complete transitive families.
Streaming conversion holds dense policies for only one game, writes tensor
memmaps, and emits the existing sparse policy format. It preserves the existing
target conversion, near-mate floor.5/horizon60 and unchanged masks/weights.

All three arms train from scratch on identical source families/splits, seed3173,
batch256,LR.002,EMA.999,warmup3,max30epochs/patience10. Every epoch is saved for a
bounded later shortlist, not play-tested every epoch. Checkpoints land under
models/candidates/b2_001_control, b2_001_state_cnn and b2_001_hybrid when the run
root is iterations/b2_001. There is no automatic release promotion.

The launcher checks immutable source/runtime/tool/model identities, stops on
failed subprocesses and hashes stage artifacts. Completed stages can be reused;
it does not promise mid-epoch optimizer resume. An interrupted unreceipted tensor
directory is rejected for inspection rather than silently overwritten. The old
rehearsal manifest intentionally reflects the code used then; don't rewrite it
to force a resume after implementation changes.

## Verification

- Full suite after architecture/encoding and fork option changes:825 passed,
 172 existing deprecation warnings,3 subtests.
- Latest targeted B2 suite:16 passed, including exact streamed/reference tensor,
 policy and weight equality, source-family namespacing and failed-stage guards.
- iterations/b2_rehearsal_20260908:28 generated games,16 reanalyzed positions,
 8 retained teacher rows; all three arms completed one training epoch and passed
 checkpoint/CUDA-graph/native-search smoke checks. These are not strength models.
- Hybrid warm callbacks were~12–25%slower than control across batch1/4/8/16.
  State-CNN cost was near control. These single-worker measurements do not replace
 production throughput or equal-time strength testing.
- Optimization validation:128 generation games at3200,exact full-record parity,
 11.6%less elapsed time.128 persistent-engine games,exact game-record parity,
 10.6%less time. Campaign subprocesses opt into pinned inputs. Global default
 remains off; graph-cache optimization remains off due negligible measured gain.

## Teacher selection and subsequent work

b2_teacher_screen is the active broad selection job. It freezes gen47 epochs6,
11,17,and best-validation (duplicate hashes removed), opponentsv24/v25/v26/v27,
seeds and a new52-entry sampled book. Each checkpoint gets200 games:24free and
26matched per opponent, balanced colors.24/26 is the nearest balanced allocation
to half free/half matched within each50-game opponent cell. Ranking prioritizes
minimum color score, then overall. This selects a data teacher, not a release.

Production command, once selection is complete:

    py -3 tools/b2_campaign.py --root iterations/b2_001 --teacher-evidence benchmarks/b2_teacher_screen_20260908/selection.json

Queued as b2_production after b2_teacher_screen; b2_production_smoke follows and
requires a complete, hash-matching training receipt. No GPU stages overlap. A
failed screen leaves no selection receipt, so production fails closed rather
than picking a teacher arbitrarily. These background jobs run the written code;
they do not summon an assistant to repair a failure automatically.

The teacher is read from the verified selection report. Production refuses to
start without it. Full B2 checkpoint screens, second-seed comparison, equal-time
confirmation and owner playtest remain later work per B2_EXPERIMENT_PLAN.md.
Existing gen47 stateful data were exercised in profiling, but no old corpus is
silently mixed into the shared B2 recipe; reuse beyond this recipe remains an
explicit audit decision, not a fabricated state reconstruction.
