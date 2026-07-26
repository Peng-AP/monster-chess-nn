# dup1 control arm — report

Ran 2026-07-25 23:33 → 2026-07-26 06:24. Training 245.1 min, gate 94.1 min.

**Question.** Both v18 arms died on the vs-ramp BLACK leg at 0.30, and the
corpus was implicated over the architecture. The most-indicated mechanism was
the human-game duplication multiple: `combined_v17` carries the 96 human games
at **6×**. Does dropping to **1×** rescue Black?

**Design.** `tools/set_human_duplication.py --copies 1` →
`data/raw/combined_v17_dup1` → `data/processed/combined_v17_dup1_r50h60`.
Verified single-variable: only `human_games/` differs, all 1,673 other files
byte-identical, and the processed pair shares the **same 1,175 train games, the
same 80 human games, the same split and seed 42, the same ramp labels**. Only
the multiple differs. Training command is the v18 GAP arm's verbatim
(`git show 2d9bb1d:v18_driver.py`) with only `--data-dir` changed.
Gate protocol also verbatim: 20 games, 400 sims, seed 42, per-side floor 0.40.

## Result — FAIL

| leg | dup1 (1×) | gap (6×) | spatial (6×) |
|---|---|---|---|
| vs ramp (W/B) | **0.70** (1.00 / **0.40**) | 0.60 (0.90 / 0.30) | 0.60 (0.90 / 0.30) |
| vs v17 (W/B) | **0.50** (0.65 / **0.35**) | 0.60 (0.70 / 0.50) | 0.70 (0.80 / 0.60) |
| anchor (Black) | 0.65 (B **0.40**) | 0.80 (B 0.60) | 0.65 (B 0.30) |

Failures: `vs_v17 aggregate 0.50 <= 0.50`, `vs_v17 black leg 0.35 < 0.40`.

## What it establishes

**The duplication hypothesis is not supported.** The deciding leg did move the
predicted way — vs-ramp Black **0.30 → 0.40** — but that is *one game in ten*,
and the other two Black legs moved the opposite way (vs v17 0.50 → 0.35, anchor
0.60 → 0.40). Totalled across all three legs, Black scores **11.5/30 for dup1
against 14/30 for gap**: a 2.5-game difference on n=30, well inside noise
(SE ≈ 2.7 games). De-duplication did not rescue Black, and the evidence mildly
favours duplication having *helped*.

Best epoch 27 of 30 — cap-limited, like five of the six runs before it. Epoch
headroom remains untested (see `HANDOFF_NEXT.md` §9.4).

## Offline metrics disagreed with the arena, again

dup1 beat gap on every test-set metric — policy CE 2.4386 vs 2.6397, value MAE
0.2014 vs 0.2383, winner-sign 0.9250 vs 0.9111 — and then lost the gate. Note
the two test sets are not the same data (dup1's contains the human games once,
gap's six times), so the comparison was never sound. Law 7 holds.

## Owner playtest, same night

He beat it and reported two things, both confirmed by measurement:

1. **As Black it declined to capture a pawn about to promote.** Reproduced at
   `data/raw/human_games/white_2026_07/game_00014.jsonl` ply 9,
   `r1bqkb1r/pPpp1ppp/5n2/8/4PP2/4K3/8/8 b kq - 0 5`. After `Bxb7` White has no
   move from b7; after the played `Rb8` White has `b7xc8=Q`. **dup1 is the only
   model of four that picks Rb8** (400 sims, no root noise) — it rates the
   blunder Q=−0.1478 against the capture's Q=−0.3235. gap, ramp and v17 all
   capture. Not a search bug: `_king_safety_override` does not veto the capture
   (checked). A 400-position static probe found dup1 **mid-pack** on capture
   priors (42.0%, vs gap 46.0% / v17 41.2% / ramp 40.2%), so this is not a
   population-wide prior deficit; the value ranking in search is untested at
   scale.
2. **"Very human feel as White — echoes a structure from my data games."**
   Confirmed and attributed: of all 1,768 corpus games, only **2** share ≥3
   opening positions with the AI's White play, **both are his own human games**
   (0 of 1,086 self-play, 0 of 225 whitefocus, 0 of 201 blackfocus, 0 of 160
   promo_races). Deepest match runs to ply 7 →
   `white_2026_07/game_00007.jsonl`, a game he won as White. This is the echo
   chamber as **style transfer through the policy head**, which
   `policy_weight_for_record`'s outcome mask does not address: these are moves
   from a game he won, so they are legitimate teachers by design.

## Incidental finding: ramp's value head

Across the same 400 promotion-defense positions, ramp evaluates at a mean of
**−0.02** while dup1/gap/v17 sit at −0.33 / −0.35 / −0.47. On the owner's
position it read **+0.58** with a White pawn on b7. The strongest engine on
record is systematically the most optimistic for Black exactly where Black must
defend a promotion — and it is the second gate opponent.

## Next

The corpus remains implicated but the duplication mechanism is now tested and
does not explain it. `data/raw/ps_monster` (829 human-vs-human games, 41.2%
pawn phase against this corpus's 9.4%) is the untried lever, and the opening
finding above sharpens the case: this corpus has essentially one source of
opening variety, the owner. Merge weight is his decision.
