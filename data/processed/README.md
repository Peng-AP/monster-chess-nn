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
and source-linked split membership. Four controlled recovery corpora were
composed from them at teacher policy weights 1x/2x/4x plus a V22-searched
50/50 arm. All four arms were rejected on playing strength (`REPORT.md` §33),
and the corpora themselves have since been retired (below).

The `bootstrap_new_*` directories are the compact reusable increments;
`bootstrap_replay_*` directories are reproducible accumulated snapshots. The
remaining conversion, promotion, demo, and earlier generation directories are
retained experiment inputs, not silently mixed into current training.

## Replay retirement (2026-08-16)

`data/processed` had grown to 217 GB, almost all of it accumulated
`bootstrap_replay_*` snapshots. Nineteen of them were deleted on the owner's
instruction, freeing **172.9 GB** (now ~45 GB): the two `demo_hour` replays,
`replay_gen5_plus_conversions_w1/w4`, main generations 0001--0005, 0007, 0008
with the four `0005_*` variants, and the four closed teacher-recovery corpora.

Only replay *snapshots* were removed. Every `bootstrap_new_*` increment, both
`combined_v19_B_r50h60*` anchors, `bootstrap_replay_main_gen_0009`, and
`bootstrap_replay_main_gen_0010` (the documented `tools/tune_training.py`
input) are untouched, as is all of `data/raw` and `iterations/`.

**These were deleted permanently, not to the Recycle Bin** — at this volume it
could not hold them. They are recoverable only by recomposition, so each
directory's `replay_manifest.json` was preserved first in
`retired_replay_manifests/<dirname>.json`. Every manifest records the full
recipe: each source path with its row count and `policy_only_multiplier`, plus
the training-balance alpha and seed. Recompose with
`tools/compose_processed.py`. Two caveats: a rebuild emits the current
**sparse** policy format, so it will be roughly 4x smaller than the deleted
dense original rather than byte-identical; and `tools/overnight_sweep.py` and
`tools/average_checkpoints.py` hardcode paths from this set, so those
concluded-campaign drivers need their corpora recomposed before they will run.

Rejected-campaign processed copies were sent to the Windows Recycle Bin on
2026-08-05. Their much smaller raw sources remain under `data/raw/`, and their
processing recipes remain in benchmark/training metadata, so they can be
regenerated if an old experiment must be reproduced. See
`logs/archive/cleanup_20260805_manifest.md` for the exact cleanup set.
