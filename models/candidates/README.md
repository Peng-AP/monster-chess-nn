# Which model to load

## Current, 2026-09-05

| role | path |
|---|---|
| **numbered release (v24)** | `models/bootstrap_v24/best_value_net.pt` — generation 30, promoted 2026-08-22 |
| **strongest measured** | `models/candidates/bootstrap_main_gen_0042/screen_nominee.pt` — leads both Elo ladders |
| **newest** | `models/candidates/bootstrap_main_gen_0044/best_value_net.pt` — trained 2026-09-05 |
| prior release (v23) | `models/bootstrap_v23/best_value_net.pt` — generation 15 |
| prior release (v22) | `models/fresh_start_v22/best_value_net.pt` |
| fixed low anchor (v21) | `models/fresh_start_v21/best_value_net.pt` — **Elo 1000** on both ladders |

**For play or evaluation, load v24** (the release) or **gen42** (the strongest
measured). They are not the same model: the 2026-09-04 round robin puts gen42
at 1781 free Elo against v24's 1564, but promotion is the owner's call and
gen42 has not had a playtest.

`screen_nominee.pt` is the gated artifact for a generation — an epoch snapshot
selected by play, not `best_value_net.pt`, which is the training-loss pick.
Where both exist, **the nominee is the one that was measured**.

Elo ladders and the tier split are in `HANDOFF.md` §2; per-model evidence in
`benchmarks/tournament/`.

---

## Historical ledger (2026-08-16, superseded above)

| role | path |
|---|---|
| **numbered release and gate (V22)** | `models/fresh_start_v22/best_value_net.pt` |
| prior numbered release (V21) | `models/fresh_start_v21/best_value_net.pt` |
| prior stronger gate (V21b) | `models/fresh_start_v21b/best_value_net.pt` |
| prior bootstrap bar | `models/candidates/gen7_scratch/screen_nominee.pt` |
| V22 source checkpoint | `models/candidates/gen9_scratch/screen_nominee.pt` |
| exact Gen9 moves-left experiment | `models/candidates/gen9_mlh_lift/best_value_net.pt` |

Gen9 epoch 6 passed the complete paired gate against Gen7 twice, improving
both colours in the initial leg and clearing the absolute Black floor again on
fresh openings. `screen_nominee.pt` is byte-identical to
`selected_epoch_006.pt` (SHA-256 starts `6a59b1f7`). It still needs the owner's
playtest and explicit release naming; no numbered directory has been created.
It subsequently scored 0.5763 against v21b over 400 paired games
(W 0.6975/B 0.4550). The owner promoted it as V22 on 2026-08-16; the release
copy is byte-identical (SHA-256 starts `6a59b1f7`). Gen10 and its controlled
seed-43 replicate both failed the Gen9 gate, so neither supersedes it.

`gen9_mlh_lift` is not a stronger nominee. It adds only the four moves-left
head tensors; every inherited Gen9 tensor is bit-identical. The head learned a
moderate held-out signal, but bounded search utility scored 0.5062 against its
own off configuration and changed none of 16 capped conversion choices. It is
kept as an exact experimental base, with search consumption off unless
explicitly requested.

**Do not load `best_value_net.pt` from inside a candidate directory.** That is
the offline-selected checkpoint and can differ from the model the arena and
gate actually scored. For the current bootstrap candidates, load
`screen_nominee.pt`.

## Layout

Each directory keeps the models that mean something:

- `screen_nominee.pt` or legacy `arena_selected.pt` — the epoch the arena
  chose. **This is the model the gates scored.** Prefer it whenever it exists.
- `best_value_net.pt` — lowest training loss. Rarely the right one to play.

Intermediate per-epoch checkpoints (`selected_epoch_*.pt`) were moved to
`models/archive/candidate_epochs/<arm>/` on 2026-08-06 — 185 files, 1.43 GB.
Nothing was deleted; every arm kept its arena-selected and training-best
models. Delete that archive directory if you want the space back.
