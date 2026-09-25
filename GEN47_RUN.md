# Gen47: stateful, mixed-opponent data — September 7, 2026

## Owner authorization and releases

Owner: "Promote 3 releases. Which 3 are up to you. then, queue 47 with the revised data recipe."

Immutable release copies, checked byte-for-byte:

| Release | Source checkpoint | SHA256 |
|---|---|---|
| v25 | gen42 `screen_nominee.pt` | `035324932307273545e2a47545899d09a2dbb5b2965bfbab04aeaf01de86e15a` |
| v26 | gen45 `arena_selected.pt`, epoch 11 | `8ed079aea313ffea79d4f2dd9879d62692d5a49915245dbc5dd86ce8f68a3c86` |
| v27 | gen46 `arena_selected.pt`, epoch 7 | `976294daf7e3d6f0c51c358dd602f11997c7fdf2dc4b255b810b588c253e5459` |

Each lives at `models/bootstrap_vXX/best_value_net.pt`, with a promotion manifest.
`models/bootstrap/champion.json` selects v27. Legacy `tools/gate.py` now names
v27 as its bar without changing any threshold. No old checkpoint, gate verdict,
or generation state is rewritten. Gen42's release rests on tournament evidence;
this is not a claim that its old gate passed. Gen45's binding opponent was gen44,
not gen42. Gen44 remains an intermediate candidate. No transitive Elo arithmetic.

During the ongoing gen46 follow-up, `src/*.py` remains frozen. In particular,
the legacy config fallback and iterate fallback still name v24; the bootstrap
champion pointer and gen47's explicit incumbent select v27. Use the v27 path
explicitly for legacy CLI play until those fallbacks can safely be updated.

## Data recipe

Teacher/parent: v27 (gen46). Total **5,600 new completed games**:

| Component | Games | Search |
|---|---:|---:|
| Ordinary free self-play | 2,080 | 700 simulations |
| Fresh randomized eight-ply prefixes, followed by teacher self-play | 1,600 | 700 |
| Teacher versus older opponents | 1,120 | 700 both sides |
| Completed continuations from distinct current-generation parent games | 800 | 3,200 |

Fresh prefixes come from gen44 epoch9, gen45, and gen46, approximately equally.
These are newly played stochastic prefixes, not the stale v31 FEN pool and not
an evaluation book. Prefix state/history is retained, but prefix moves are not
used as policy training rows.

League opponents: gen42/v25, gen44 epoch9, gen45/v26. Teacher plays 560 as White
and 560 as Black. Each color is divided 187/187/186 across opponents. Wins,
draws and losses all remain. Older-opponent policy rows are masked (value labels
remain); proven exact-finisher moves are usable regardless of which model moved.
The teacher may supply replacement policies through the normal reanalysis sample.
This does NOT claim every masked row gets reanalyzed.

Forks are sampled without tactical rules, result filtering, or human-game inputs:
one settled position at/after eight plies from each of 800 distinct new parent
games, 480 Black / 320 White. They continue to actual termination at 3,200 sims.
Existing threefold and turn-cap semantics remain; cap labels retain the existing
training lean and are not reported as captured-king wins. A timeout or nonterminal
empty action is an error, never an invented draw.

All new positions carry initial FEN plus the entire played action prefix, current
FEN, phase and turn count. Reanalysis replays that state, preserving history and
clocks. Continuations reconstruct repetition counts before play resumes. This
does not add repetition adjudication inside the MCTS tree; search is unchanged.

Source games, forks and teachers share one transitive train/validation/test family.
The root game's result controls split stratification, while each continuation keeps
its own actual outcome. Thus differing outcomes cannot break grouping or leak a
parent into validation through a fork. All parents are new gen47 games, avoiding
cross-generation split reassignment in already accepted replay.

## Training and measurement

Unchanged from gen46: scratch seed3173, architecture, scalar value ramp .5/60,
batch256, LR .002, warmup3, EMA .999, max30 epochs, patience10. No human rows,
no anchor. Replay8: gen39/40/41/42/44/45/46 plus47. Reanalyze80,000 and retain
40,000 policy-only teachers at3,200 sims, 60%Black, policy multiplier4.

Data base1,047,000,000; ordinary tasks occupy the first4,800 seeds, fork selection
+500,000 and fork play +600,000. Canonical reanalysis seed1,047,300,000 and replay
composition1,047,700,000. Training/screen/gate RNG namespaces remain separate.

Same play-based checkpoint selection, advisory offline check, 400-game v27 par,
two400-game binding legs at3,200, and200 self-games on PASS. No automatic v28.
After a measured outcome, nonbinding transfer compares gen47 and v27 on identical
120 newly generated starts against v24 and v26 (240 games per pairing), plus200
free gen47 games against each. These are held-out games, not all held-out opponents:
v26 is in the training league; v24 is outside that league. No evaluation outputs
are imported into training. Follow-up seed170,000,000.

## Implementation and queue

Opt-in adapters under `tools/` leave the ongoing gen46 runtime untouched:

- `stateful_generation.py`: complete states, deterministic tasks, two models maximum
  per worker batch, eight workers maximum, one exclusive game job, per-game durable
  output receipts and resume hashes, bounded stalls, no incomplete outcome labels.
- `reanalyze_stateful.py`: full-state adapter to existing resumable teacher search.
- `process_families.py`: transitive family splitting with unchanged tensor conversion.
- `iterate_stateful.py`: explicit command-plan adapter; the canonical iteration
  still handles replay, training, selection, gating and acceptance.
- `start_gen47.py`: tests, isolated real rehearsal, production, then transfer checks.

```powershell
py -3 tools/runs.py start --name gen47_stateful_mixed_v3 -- py -3 tools/queue_after.py --after gen46_transfer_checks -- py -3 tools/start_gen47.py
```

The queue requires the predecessor follow-up state to be complete, not merely a
dead process. Then it runs the full tests with the worker lease free and a real
28-game/8-sim end-to-end rehearsal (including league, fresh starts, forks, teacher
search, processing/audit, training and play testing). A weak smoke model losing is
expected; any execution error stops before production. Rehearsal artifacts live
under `iterations/rehearsal_stateful_gen47_20260907`, outside the main replay.

Run log `logs/gen47_stateful_mixed_v3.log`; production state
`iterations/gen_0047/state.json`; transfer output
`benchmarks/generalization/gen47_20260907`.

Resume with the same launcher; it uses explicit existing states and validates
recipe/implementation/data hashes. Do not change recipe files, adapters or runtime
after the queue starts. No polling assistant, automatic repair, push or cleanup.

Prequeue CPU verification: twelve new targeted tests passed; release/gate tests also
passed (48 combined). Full suite while gen46 was active hit four existing lease
conflicts (including CLI --help taking the worker lease); those tests are rerun
after the predecessor exits. The old v24 bar assertion was updated for v27.
Actual GPU rehearsal is queued, not claimed completed at queue submission.
