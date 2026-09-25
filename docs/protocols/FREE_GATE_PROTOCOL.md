# Free-play evaluation and bootstrap integration, v2

Implementation started from the untracked September 5 handoff. The numbered
release remains v24 (gen30); gen42 remains the strongest tournament model.
Nothing in this document promotes gen44.

## What changed

`tools/gate_free.py` now emits `free_endpoint_equal_color_v2` reports:

- Captures-only scoring uses `match.game_score`: +/-0.5 cap labels are draws.
- Aggregate scores give White and Black equal weight, even when deduplication
  retains different numbers of games on each side.
- Each leg reports sampled, unique-endpoint, and previously unseen-endpoint
  W/D/L, counts, scores and draw-aware nominal standard errors by color.
- Default coverage is 100 distinct endpoints per color in the first leg and
  100 additional unseen endpoints per color in confirmation. Self-par also
  requires 100 per color. Budget exhaustion is INCONCLUSIVE, not rejection.
- Original score rules are unchanged: each full unique leg must score strictly
  above 0.50; each side must remain within 0.05 of the bar's same-side self-par.
- The novel confirmation subset supplies coverage. Its conditional score is
  shown separately, not substituted for the full confirmation's score.

This is an endpoint-uniform instrument, not a natural-play win-rate estimate.
Raw sampled results remain visible for that reason. Endpoints are keyed by
candidate color, FEN, pending White half, and turn count. Early terminal games
are retained. Repetition-history and prefix-trajectory hashes audit different
histories reaching the same endpoint. Conflicting outcomes or game lengths at
a merged endpoint force INCONCLUSIVE. Identical endpoints are not claimed to
be mathematically independent trials. Reported SEs are descriptive; side-delta
SEs include uncertainty in par but do not establish statistical non-inferiority.

## Evidence and recovery

Each invocation gets an isolated directory under `benchmarks/free_gate/` with
an immutable manifest, per-leg progress files, per-batch journals, and report.
The manifest pins checkpoints by SHA-256, engine/source identity, rule/search
environment, opening sampler, seed schedule, coverage, and budget. Default
batches are 32 games and shrink near the soft deadline. Healthy in-flight games
finish; a budget is not a destructive process-kill deadline.

Each completed match task is fsynced to JSONL with its stable task ID and seed.
Rows also contain the board-state/move trajectory and capture/repetition/cap
ending reason. Resume validates manifests and rows, then schedules missing
tasks only. Torn final bytes are archived beside the journal; malformed complete
records or duplicate task IDs fail loudly. Batch seed blocks are disjoint within
the declared schedule. Cross-leg endpoint overlap is reported separately.

The v2 par cache lives in `benchmarks/free_par_v2/`; it never reads the old
`bar_name@sims` cache. Cache identities include checkpoint content and runtime
configuration. Reuse checks coverage and source-log hashes. Source hashing is
deliberately conservative: editing Python engine/pipeline code invalidates
resume/cache compatibility even if an edit was operational only. Do not edit
those sources while a measurement campaign is running.

An OS-held worker lease serializes gate/match campaigns, generation,
reanalysis, and training command-line runs. It releases on process exit/crash.
Other older scripts that bypass these entry points still require the standing
operating rule: no concurrent worker jobs, at most eight workers.

Example, gen44 against gen42 at playing depth:

```powershell
py -3 tools/runs.py start --name gen44_free_v2_3200 -- py -3 tools/gate_free.py --model models/candidates/bootstrap_main_gen_0044/best_value_net.pt --bar-model models/candidates/bootstrap_main_gen_0042/screen_nominee.pt --sims 3200 --workers 8 --target-per-side 100 --par-per-side 100 --budget-min 180 --seed 7200000 --run-dir benchmarks/free_gate/gen44_gen42_3200_v2 --report-path benchmarks/gen44_gen42_3200_v2.json
```

Resume with the identical arguments, replacing `--run-dir PATH` with
`--resume PATH`. A completed budget-limited report stays completed and
inconclusive; resume does not silently grant another budget. Choose a separately
preregistered follow-up campaign when more measurement is warranted.

## Pipeline policy

`configs/bootstrap_generation_only.json` is the production default recipe read
by `src/iterate.py`; explicit CLI overrides are recorded in generation state.
It uses no anchor, eight accepted increments, 1000 free plus 400 book-seeded
games, 700 generation sims, 20,000 sampled/10,000 retained 3,200-sim teachers,
60% Black teachers, and the existing fresh 30-epoch/10-patience training recipe.

Checkpoint screening still uses the existing bounded shortlist and play-based
nomination. It is not a binding rejection. Offline comparison remains advisory
and now accepts sparse policy storage. The binding stage invokes the free gate
at 3,200 sims. Its two legs replace the old separate high-fidelity stage, which
is explicitly skipped. Self-skew remains diagnostic. Legacy book evaluation
requires `--gate-backend legacy`; it is not silently mixed into free gating.

Promotion requires a complete compatible PASS, matching checkpoint hashes,
coverage and evidence hashes. INCONCLUSIVE leaves the champion untouched.
`--promote-on-pass` affects only the working-generation pointer; numbered
releases still require the owner's decision. Historical state files do not
automatically migrate protocols on resume: changed experiment configuration
is rejected. Do not rewrite old completed phases to make them look like v2.

## Existing gen44 evidence

Original `benchmarks/gate_free_gen44.json` and `gate_free_legs/` are unchanged.
`tools/rescore_free_gate.py` generated the separate
`benchmarks/gate_free_gen44_rescore_v2.json` audit, including source hashes.

| 1,600-sim evidence | White distinct | Black distinct | Equal-color score |
|---|---:|---:|---:|
| First leg | 74 | 121 | 60.57% |
| Full confirmation | 57 | 95 | 61.75% |
| Combined endpoint union | 102 | 177 | 61.59% |

Confirmation overlapped the first leg at 68 endpoints, leaving only 28 unseen
White and 56 unseen Black endpoints. The original scores are promising, but
coverage under the planned v2 policy is INCONCLUSIVE. This is a retrospective
audit, not a new preregistered gate or an assertion that gen44 became weaker.

Gen44 also changed teacher identity, replay volume, generation volume, teacher
volume and seed versus gen42. It is not an isolated anchor-removal experiment.
Do not restore human/old anchor data as an alleged one-variable control.

## Remaining campaign

Verification completed: 754 full-suite tests passed (172 existing PyTorch
deprecation warnings), plus a focused rerun after final pipeline verdict
hardening. The isolated end-to-end rehearsal generated 32 free + 8 book-seeded
games, audited 10 teachers (6 Black/4 White), retained 5,524 rows, trained one
epoch and reached the binding gate. It exposed and fixed the sparse-policy
reader bug in offline comparison. Resume reused the completed data/training
phases. The deliberately undertrained candidate failed; the champion stayed
unchanged. Tests also exercise the passing/no-promotion path and interrupted
confirmation recovery. A real completed two-game journal resumed in 0.2 seconds
with no additional games. Tiny eight-simulation scores are plumbing tests only.

The sequential campaign runner also completed its tiny real rehearsal through
all supporting opponents and self-play. The actual `gen44_depth3200_v2` campaign
is now launched via `tools/runs.py`, with 180 minutes for gen42 par/H2H and
50 minutes each for gen41, v24 and self-play (5.5 hours total soft allocation).
Its directory is `benchmarks/free_gate/gen44_depth3200_v2/`, and its log is
`logs/gen44_depth3200_v2.log`. Eight workers, 32-game batches, 100-per-color
targets; the manifest fixes these settings before games begin. The runner
never promotes or launches training automatically.

After verification: finish the 3,200-sim gen42 comparison, then sequential
gen41/v24 checks and gen44 self-play diagnostics within a declared budget.
Only then decide whether gen44 can generate the next controlled bootstrap
increment. Uncertain coverage must not be called a model failure, and no v25
promotion is automatic. The detailed continuation rule remains in the local,
intentionally untracked handoff.
