# Processed data

The active bootstrap corpus is assembled from immutable anchors plus small
per-generation increments:

- `combined_v19_B_r50h60_capture` — current immutable v19_B/V20 anchor.
- `bootstrap_new_main_gen_0002` through `...0005_teacher3200`, plus
  `...0007`, `...0008`, `...0009`, and `...0010` — accepted/informative
  replay increments.
- `bootstrap_replay_main_gen_0010` — latest complete accumulated corpus:
  1,047,422 rows. Gen10 failed its playing-strength gate, so this snapshot is
  informative replay rather than proof of a stronger model.

## Deep-teacher recovery (2026-08-16)

The original Gen7--Gen10 processing commands retained only non-human files
with at least four rows. Each 3200-simulation reanalysis teacher is a deliberate
one-row policy-only file, so all 4,000 retained teachers per generation were
silently dropped. The ordinary self-play rows were unaffected.

Gen7--Gen9 have been reprocessed without overwriting their historical outputs:

- `bootstrap_new_main_gen_0007_teacher3200_fixed`
- `bootstrap_new_main_gen_0008_teacher3200_fixed`
- `bootstrap_new_main_gen_0009_teacher3200_fixed`

Each corrected increment has a passing `generation_audit.json`: exactly 4,000
teacher files, 8,000 rows after mirroring, 60% Black-to-move, zero value weight,
and source-linked split membership. The controlled recovery corpus is
`bootstrap_replay_main_gen_0009_teacher_recovery4x` (960,324 rows). Its 32,000
deep-teacher rows from Gen5/7/8/9 have effective policy weight 4; no teacher is
used as a value target. This corpus is an experiment input until its fresh
seed-42 model passes playing-strength gates against Gen9.

The `bootstrap_new_*` directories are the compact reusable increments;
`bootstrap_replay_*` directories are reproducible accumulated snapshots. The
remaining conversion, promotion, demo, and earlier generation directories are
retained experiment inputs, not silently mixed into current training.

Rejected-campaign processed copies were sent to the Windows Recycle Bin on
2026-08-05. Their much smaller raw sources remain under `data/raw/`, and their
processing recipes remain in benchmark/training metadata, so they can be
regenerated if an old experiment must be reproduced. See
`logs/archive/cleanup_20260805_manifest.md` for the exact cleanup set.
