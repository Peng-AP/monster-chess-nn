# Which model to load

Current playtest ledger (2026-08-15):

| role | path |
|---|---|
| numbered release (V21) | `models/fresh_start_v21/best_value_net.pt` |
| stronger unnumbered gate (V21b) | `models/fresh_start_v21b/best_value_net.pt` |
| V22 candidate / prior bootstrap bar | `models/candidates/gen7_scratch/screen_nominee.pt` |
| **leading V23 candidate** | `models/candidates/gen9_scratch/screen_nominee.pt` |

Gen9 epoch 6 passed the complete paired gate against Gen7 twice, improving
both colours in the initial leg and clearing the absolute Black floor again on
fresh openings. `screen_nominee.pt` is byte-identical to
`selected_epoch_006.pt` (SHA-256 starts `6a59b1f7`). It still needs the owner's
playtest and explicit release naming; no numbered directory has been created.

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
