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

The `bootstrap_new_*` directories are the compact reusable increments;
`bootstrap_replay_*` directories are reproducible accumulated snapshots. The
remaining conversion, promotion, demo, and earlier generation directories are
retained experiment inputs, not silently mixed into current training.

Rejected-campaign processed copies were sent to the Windows Recycle Bin on
2026-08-05. Their much smaller raw sources remain under `data/raw/`, and their
processing recipes remain in benchmark/training metadata, so they can be
regenerated if an old experiment must be reproduced. See
`logs/archive/cleanup_20260805_manifest.md` for the exact cleanup set.
