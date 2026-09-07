# Gen46 controlled data expansion — September 7, 2026

Owner authorized starting the larger generation after expressing concern about
exploiting familiar model openings. No numbered-release promotion is authorized.

## Learning recipe

- Teacher: `models/candidates/bootstrap_main_gen_0045/arena_selected.pt`, epoch 11,
  SHA256 `8ed079aea313ffea79d4f2dd9879d62692d5a49915245dbc5dd86ce8f68a3c86`.
- 4,000 fresh free games and 1,600 fresh book-seeded continuations at 700 sims.
- Existing `data/start_fens/book_v31_seed.jsonl` start pool; no human training rows.
- Reanalyze 80,000 positions at 3,200 sims, retain 40,000 policy-only targets:
  24,000 Black and 16,000 White before mirroring; teacher policy multiplier four.
- Replay: generations 38-42, 44, 45 and new 46; no anchor. Same balancing,
  target shaping, architecture and optimizer as gen45. Fresh scratch training,
  seed 3173, batch 256, maximum 30 epochs, patience 10.
- Eight workers, one game job at a time; expected game-job VRAM about 10 GiB.

Training/gating run: `gen46_expanded_sampled_v3`.
State: `iterations/gen_0046/state.json`; log: `logs/gen46_expanded_sampled_v3.log`.

```powershell
py -3 tools/runs.py start --name gen46_expanded_sampled_v3 -- py -3 src/iterate.py --incumbent models/candidates/bootstrap_main_gen_0045/arena_selected.pt --seed 3173 --data-seed-base 1000000000 --games 4000 --book-seed-games 1600 --reanalysis-sample 80000 --reanalysis-keep 40000 --self-skew-games 200
```

## Seed isolation

The old generation seed stride, 1,009, would overlap gen45's 2,000 free-game
seeds. Explicit `--data-seed-base` separates generation randomness from training
initialization. Omitted, it preserves the historical formula for old resumes.

- Gen46 free base: 1,046,000,000; book base: 1,046,400,000.
- Reanalysis base: 1,046,300,000; composition sampler: 1,046,700,000.
- Training and split seed: 3173 (unchanged).
- Probe/full checkpoint screens: 46,303,173 / 46,403,173.
- Gate first/par/confirmation: 46,503,173 / 46,603,173 / 46,703,173.
- Candidate self-skew: 46,803,173.

## Automatic stages and broader checks

The ordinary iteration chains generation, journaled reanalysis, processing/audit,
replay composition, training, bounded checkpoint screen, advisory model comparison,
and the unchanged sampled gate: 400 fresh gen45 calibration games plus two 400-game
H2H legs at 3,200 sims. PASS adds 200 candidate self-play games and ends without
promotion. FAIL/INCONCLUSIVE stops the ordinary iteration normally; execution
errors record a failed phase and stop safely.

`gen46_transfer_checks` waits for that named run to end. It refuses to proceed
after an execution error or incomplete iteration, but runs after a measured PASS,
FAIL or INCONCLUSIVE if the selected candidate exists. Its output directory is
`benchmarks/generalization/gen46_20260907/`, with `state.json` and `summary.json`.

Declared nonbinding follow-up:

1. Audit actual replay rows/teacher lineage. Save a deterministic game-level
   half-data partition (seed 3173), separately halving free and book-seeded games.
   This does not train a half-data control or discard any data. A future control's
   teacher rows must follow source-game ancestry and existing split membership.
2. Generate 120 distinct randomized eight-ply starts from fixed v24, gen42 and
   gen44 references (40 per source), 700 sims, temperature 1.0, seed 140,000,000.
   The candidate and baseline do not generate this set. It is created after
   training in the benchmark directory and never enters training. This broadens
   the test but is not uniform legal-position sampling or perfect-play truth.
3. Gen45 and gen46 each play the same 120 starts, both colors, against each of
   gen42/gen44: four 240-game matches at 3,200 sims. Same start block and seed
   within each comparison; opponent seed bases 144,000,000 and 145,000,000.
4. Gen46 also plays 200 free games against each reference, seed bases
   147,000,000 / 148,000,000. Do not pool these with selection games.
5. Probe the first ten White turns (rows 0,2,...18) of human games
   `black_2026_07/game_00031.jsonl` and `game_00032.jsonl`, both gen45 and gen46,
   3,200 sims, seeds 101/211/307, five-ply continuations. These are known regression
   cases, not pristine held-out tests; fresh owner play is still required.

The follow-up is launched via `tools/queue_after.py --after gen46_expanded_sampled_v3`
and `tools/generation_followup.py`; its exact command is recorded in
`logs/gen46_transfer_checks.json`. It never changes the binding verdict or promotes.
Completed stage hashes and model/runtime identities protect follow-up resumes;
partial matches use durable journals. A partial/unrecorded book needs inspection
before resuming, not silent regeneration or easier acceptance thresholds.

## Preflight and operating boundary

794 tests and three subtests passed. Expanded iteration dry run validated all
replay inputs, stage commands, separate data seeds and unchanged training seed.
A tiny real follow-up completed start generation, matched baseline/candidate
games, a free match and human diagnostics. The half-data snapshot also ran on
the completed gen45 data without modifying it. Code: commit `e35402f`.

Freeze runtime sources and diagnostic implementations until both jobs finish.
No active assistant polling loop, automatic code repair, prompt, push or promotion.
Allow roughly 13-16 hours including broader diagnostics; contention can extend it.
More fresh data and a stronger teacher change together, so gains cannot be
attributed to quantity alone. Resume original arguments only; never restart an
old generation with changed counts or source files.
