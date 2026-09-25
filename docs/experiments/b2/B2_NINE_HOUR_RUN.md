# B2 nine-hour follow-up — September10

Owner requested roughly9hours of the most valuable follow-up. Started as
b2_nine_hour_20260910, tools/b2_nine_hour.py. One sequential job,8workers;
pinned-input enabled in subprocesses. No model deletion, promotion or runtime
edits. Fixed stages finish even if wall time exceeds9hours (~8–10h estimate).

1. Verify original15/24-channel prepared data hashes. Train fresh control and
   state CNN with seed9053, otherwise identical original commands (EMA.999,
   max30epochs/patience10,256batch). Save all epochs. New separate directories
   models/candidates/b2_seed9053_control and b2_seed9053_state_cnn.
2. Preregister epochs8/15, never substitute an offline best after seeing games.
   Existing seed3173 control/state8/15 plus new seed9053 control/state8/15 and
   gen47epoch17 =>9models,400games each againstv24/v25/v26/v27,3200sims.
   If early stopping prevents a replica epoch, explicitly mark it unavailable
   and continue remaining panels; no ad-hoc replacement or hidden claim of success.
3.200 common self-starts,one game per state,for original control15,state8,state15.
   This removes the duplicate color-swapped self-game and model-specific opening
   distribution confounds.600self-games,not a target of50:50 color balance.
4.6400-sim matched panels,200games each for state8,state15,gen47. This tests
   deeper-search transfer; it is NOT an equal-wall-time architecture comparison.
5. Automated paired-opening bootstrap intervals and Black regression examples.
   Primary comparison:state-CNN minus control at epoch15,each seed separately;
   epoch8/depth diagnostics secondary. Exploratory intervals,not multiplicity-
   adjusted promotion decisions. No release automatically selected/promoted.

Up to4800games plus two trainings. New books:200ordinary,200self,100depth.
Book seeds2040000000/2041000000/2042000000; teachersv24/v25/v26/gen44/gen47;
16plies,temp.5,700sims,8x attempt budget,strict required counts/no --allow-short.
New seeds do not guarantee disjoint chess positions from historical books.
Training data unchanged; matched starts shared by all compared models.

Outputs:benchmarks/b2_validation_20260910/{status,manifest,summary,paired_analysis,
black_regression_examples}.json plus per-game logs and stage receipts. Matches
resume through journals,completed stages hash-checked. Interrupted/unreceipted
training directories fail for inspection (no exact optimizer resume). No automatic
assistant repair or restart after Windows Update is implied; keep machine awake.

Validation before launch:3new planning/paired-statistic tests passed; four actual
native self-games at8sims passed through the new shared-opening driver, with one
row per opening (benchmarks/b2_common_self_smoke_20260910.json).
