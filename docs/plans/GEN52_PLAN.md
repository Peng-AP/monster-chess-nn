# Gen52 — self-play plus a model pool (draft, September 26, 2026)

Status: **draft; owner decisions recorded below; not implemented or queued.**
Implementation waits until gen51 production finishes, because gen51 pins
`tools/stateful_generation.py` and the other generation tools by hash.

## Why

Gen51's control nominee (epoch 16) passed gate v4 against v28 decisively
(79.3% over 800 fresh games at 3,200; 95.6% at 12,800; both colours far above
v28's self-par). The older-opponent diagnostics then found a hole the gate
could not see. After e4+d4 …d5, gen51 has converged on **c4 + Ke2** as White:

| Matchup (White after e4+d4 …d5) | Games | Result for White |
|---|---:|---|
| gen51 vs v28 (gate legs), c4 + Ke2 | 317 | 154 W / 150 D / 13 L |
| gen51 vs gen49 (diagnostic), c4 + Ke2 | 61 | 9 W / 2 D / 50 L |
| v28 vs gen49 (Sept 17), mostly e5 + Ke2 | 67 | 24 W / 2 D / 1 L for e5 + Ke2 (27 games) |

gen51 learned from v28's games and beats v28's defence of this line; gen49
defends it differently and wins. Head-to-head against the teacher overstates
general strength (a known non-transitivity; see the round robins in
`CONTEXT.md` §5b). Training against several models, not only the teacher,
targets exactly this.

## Owner decisions (September 26)

- **Model pool, on top of normal self-play, not replacing it.**
- **Pool:** v28 (`models/bootstrap_v28`), gen49 epoch7, gen48 epoch17, v26
  (gen45). gen49 is included because it found the hole.
- **Held out, never trained against:** **B2** (state-CNN epoch 8) and **v27**.
  These stay unseen evaluation opponents for gen52 and later.
- **Amount:** **1,200 pool games**, split evenly across the four pool models
  and both teacher colours (150 per model per colour).
- **Design:** two training runs from one shared generation, like gen51:
  - **Arm A:** shared self-play only.
  - **Arm B:** Arm A's data plus the pool games.
  - A clean yes/no on whether pool games help.

## Proposed recipe (to confirm when implemented)

- Teacher: gen51's winner after the owner's playtest (gen51 control epoch 16
  unless gen51's final results or the playtest say otherwise).
- Shared generation, unchanged from gen51: 2,800 self-play games at 1,600
  simulations with 30-ply exploration, 400 parent-linked forks at 12,800,
  24k→12k reanalysis at 12,800, 8-generation replay, scratch training.
- Pool games: existing `league` task kind in `tools/stateful_generation.py`
  (gen47 precedent: 1,120 games). The teacher plays at 1,600 simulations; the
  pool model plays at the same budget (to confirm). Only the teacher's moves
  carry policy weight; every position keeps its outcome value label. Pool games
  keep the recipe's 30-ply exploration.
- Pool games need their own processed increment so only Arm B sees them. They
  are whole games with their own families, so they need no parent-linked split.
- Deep-value source (gen51's Arm B): keep or drop by gen51's final arm-vs-arm
  result, decided before gen52's plan is frozen.

## Evaluation

- Gate v4 against the teacher, as for gen51.
- **Held-out diagnostics are primary evidence of general strength:** B2 and
  v27 at 3,200 (plus 12,800 if affordable). Pool opponents are reported but
  are not independent.
- A conditional test of the gen51 hole: games from e4+d4 …d5 against gen49's
  defence, reported separately and never mixed into the primary score.
- Arm A vs Arm B head-to-head if both pass.

## Open before implementation

- gen51 final results (Black vs gen49, v27, B2; deep-value arm; arm-vs-arm).
- Owner playtest of the gen51 nominee (the promotion decision is separate).
- Pool-model search budget, and whether pool games also get root noise.
