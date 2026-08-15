# Processed data

The active bootstrap corpus is assembled from immutable anchors plus small
per-generation increments:

- `combined_v19_B_r50h60_capture` — current immutable v19_B/V20 anchor.
- `bootstrap_new_main_gen_0002` through `...0005_teacher3200`, plus
  `...0007`, `...0008`, and `...0009` — accepted/informative replay increments.
- `bootstrap_replay_main_gen_0009` — latest complete accumulated corpus:
  936,366 rows. Generation 10 is composed beside it while its run is active.

The `bootstrap_new_*` directories are the compact reusable increments;
`bootstrap_replay_*` directories are reproducible accumulated snapshots. The
remaining conversion, promotion, demo, and earlier generation directories are
retained experiment inputs, not silently mixed into current training.

Rejected-campaign processed copies were sent to the Windows Recycle Bin on
2026-08-05. Their much smaller raw sources remain under `data/raw/`, and their
processing recipes remain in benchmark/training metadata, so they can be
regenerated if an old experiment must be reproduced. See
`logs/archive/cleanup_20260805_manifest.md` for the exact cleanup set.
