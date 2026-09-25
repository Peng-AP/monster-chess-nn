# Search-target corpus diagnostic — September 13, 2026

Read-only analysis while the fixed game campaign runs. No training, weights,
playing search, gates or openings were changed in response to these findings.
Source: `benchmarks/search_targets_20260913/campaign/corpus/groups.jsonl`.
Numbers below count records BEFORE exact-input deduplication / split exclusions.

## What actually went into the new corpus

There are33,652 records:1,280 sampled source roots,2,490 actual CPU NN-leaf roots,
and29,882 complete-turn successor records. Thus88.8% are successors. Only2,875
records (8.54%) have an exact backed value of+1 or−1. This is not predominantly
a collection of already-resolved capture positions. The full teacher trees also
encountered192 repetition terminals and3,294 turn-cap terminals; those are node
counts, not counts of training examples or independent games.

The important structural distinction is **target horizon**: source / CPU-leaf
roots receive two-completed-turn backed values, whereas their stored successors
receive the remaining one-turn backed value. Ranking compares siblings at the
same horizon, but the point-value regression mixes both horizons.

| TRAIN record group | Rows | Raw mean | Backed mean | Mean absolute change |
| --- | ---: | ---: | ---: | ---: |
| Source root, White to move |512|−0.2058|−0.2501|0.1057|
| Source root, Black to move |512|−0.2610|−0.2785|0.0908|
| CPU NN leaf, White to move |1,367|0.0772|0.0775|0.1878|
| CPU NN leaf, Black to move |623|−0.0762|−0.0883|0.1284|
| Successor, White to move |8,882|0.0752|0.2262|0.2029|
| Successor, Black to move |15,012|−0.1253|−0.2819|0.1958|

All values use White's perspective. One-turn successor targets shift strongly
toward the side to move: about+0.151 for White and−0.157 for Black. Validation
successors show nearly the same shifts (+0.151/−0.153). This is consistent with
max/min backup over an incomplete reply cycle, not evidence of a perspective
implementation bug. Two-turn root averages move much less.

## Interpretation and a possible next controlled test

This is a plausible calibration / distribution concern, NOT a demonstrated cause
of the game results. Point-value learning is dominated by shallow successor
targets rather than the actual NN leaves where the evaluator is used. Better
held-out agreement with that mixture need not improve deeper search.

If the full game results do not establish a gain, the next focused test should
use the same complete-turn horizon for point targets on sampled actual NN leaves,
or reserve successor records for sibling ranking rather than point regression.
Increase independent source-family coverage rather than merely repeating the
same22,742 prepared TRAIN states more often. Keep a matched raw-target control,
replay anchor, fixed architecture and fresh both-color confirmation. Do not
launch that alteration mid-campaign or claim this diagnostic proves it will help.
