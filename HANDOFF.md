# Monster Chess NN — handoff

**Written 2026-07-26 ~16:00. This file replaces `HANDOFF_NEXT.md` and
`SESSION_2026-07-25.md`, both deleted.** It is the single continuity document.
Local only (`.git/info/exclude`), untracked — rewrite or delete freely.

Durable companions, both tracked and both current: `README.md` (how to run
things) and `CONTEXT.md` (the incumbent/candidate/version vocabulary).

---

## STATE AS OF 2026-08-02 — read this before the rest

**Incumbent: `models/fresh_start_v19`** (was arm K). It beats the previous
strongest engine, `fresh_start_v18_ramp`, at 0.7125 pooled over 40 games/side,
and the owner has playtested it: *"big improvements as black. Very difficult to
crack as white during pawnphase… White is easy to beat, still, but coherent and
sharp."*

`DIRECTIVE.md` (the v19 campaign) is **concluded**. `OVERNIGHT_REPORT.md` holds
the evidence for everything below. Sections 4–9 of this document are the
pre-campaign state and several of their open questions are now answered:

| item | status |
|---|---|
| §4.4 refused capture | **not** a population defect — dup1 mid-pack through search, v17 worst |
| §4.5 ramp optimism | **miscalibration** — same conversion as v17, beliefs 0.41 apart |
| §4.6 epoch headroom | **none** — best epoch 24 under an 80 cap |
| §7.1 ps_monster | merged; **the single biggest gain of the campaign** |
| §7.3 wider gates | done — 20/side/leg plus a confirmation replay |
| §8.1 MCTS probe | done, deck committed |
| §8.2 `--patience` | done |
| §8.4 ramp optimism | done (§4.5 above) |

### The law this campaign established

**Every gain came from the corpus.** Architecture (§4.2), duplication (§4.1),
capacity (2.74× tower), and side-weighting are all measured nulls or worse.
The corpus ladder, by contrast, moved the deciding leg monotonically:
Black-vs-ramp 0.300 (v17 corpus) → 0.450 (+27 owner games) → 0.700 (+ps_monster).

### What changed structurally

- **`clone()` copies 8 plies, not the whole stack** — it was 82% of a late-game
  decision. Full-length games got **4.64× faster**, play verified move-for-move
  identical. §10.4's timings are obsolete.
- **The gate bar tracks the incumbent** (`tools/gate.py`), per the owner's rule
  "every model better than the last, definitively." It was ramp; it is now v19.
  A passing candidate replays the bar leg on a fresh opening seed.
- **Value weights exist** (`value_weights.npy`) — "teach policy, not value" is
  now expressible, which is what made the ps_monster fork an A/B.
- **`plies_to_end` is stamped at generation** so filtering a corpus to a phase
  no longer silently relabels the survivors.
- **`config.DEFAULT_GAME_WORKERS = 8`** — `cpu_count()`-derived defaults crash
  CUDA init on this box and leave orphaned 1.4 GB processes.

### Numbers worth not re-deriving

- **Search is Python-bound, not GPU-bound.** Profiled at 800 sims: move
  generation 35%, board clone/apply 25%, tree/Python 26%, **NN forward 14%**.
  Raising the MCTS leaf batch 16 → 256 buys only 1.53×; there is no more than
  ~15% left in batching. The ceiling is `python-chess` move generation.
- **The cliff is search-limited.** v19-class Black converts 0.36 at 200 sims and
  **0.84 at 1600** against a fixed White. Knowledge is present; 400 sims does
  not extract it.
- **The corpus is 64/36 White by construction** — White moves twice per turn, so
  each turn emits two White records against Black's one.
- **Self-play skew:** v19's own self-play runs ~77.5% White, worse than v17's
  56%, in a variant Black is supposed to win. Improving both sides improved
  White faster.
- **The post-promotion class is the live problem.** With Black holding 3+
  heavies and White still holding pawns, the corpus records Black converting
  **36%** — and v19 self-play converts **30%**. The pessimism is accurate for
  the model, so **self-play regeneration cannot fix it**; corrected labels need
  a Black stronger than the White it faces (asymmetric search, owner games, or
  an oracle that handles more than a bare king).

### Hazards now pinned by tests

Ramp labels are positional (filtering records relabels survivors); match seeds
closer than the game count replay the same games and look like confirmation;
`config`'s `VALUE_TARGET_FLOOR/HORIZON` are 0.97/10 and **not** the ramp's
0.5/60 that every `r50h60` dataset was built with.

### Still open, still needing the owner

- **§7.4** the hand-corrected label precedent (`white_2026_07/game_00013`).
- **`data/raw/combined_v16` is on neither this box nor the transfer drive.**
  Ramp trained on it; every v18 arm that lost to ramp trained on v17; every
  current corpus inherits v17. One directory copy settles the confound.
- **The per-side floor has a shelf life.** As Black approaches correct play,
  White's achievable score against it falls below 0.40 by the nature of the
  game, and every candidate would fail the White leg on the variant rather than
  on quality. White's criterion will need to become resistance (survival
  length, which the ramp already encodes) rather than win rate.

---

## 1. The rules, exactly

White: king + 4 pawns on c2–f2, **two moves per turn**. Black: full army, one
move. No castling for White (king starts on e1, no rook).

`monster_chess.is_terminal` fires **only on king absence** — there is no
checkmate in this implementation, a king must actually be captured (owner rules
correction 2026-07-04). Consequences, all pinned by
`tests/test_king_capture_rules.py`:

- King capture wins **unconditionally**, even if the capturer's own king is left
  attacked — the game ends before any reply.
- White in check may still spend both moves capturing the Black king.
- A "defended" Black king inside White's double-move range is **not** safe.
- White's **first** half-move may pass through check; only the completed turn
  must leave White's king safe.

Move limit `MAX_GAME_TURNS = 150`, with a symmetric ±0.5/0 relabel by heuristic
sign at the cap (a position-dependent proxy, applied equally both directions).

### 1.1 Two rules where we differ from playstrategy.org — both settled

Surfaced replaying their corpus, both put to the owner 2026-07-25, both decided
**in favour of the behaviour already implemented**. Pinned by
`tests/test_ruleset_divergences.py` so they stop being re-litigated.

1. **En passant is conferred only by the LAST move of White's turn.**
   python-chess holds one ep square and each push recomputes it, so `e4` then
   `f4` leaves only `f3` capturable. playstrategy allows either. Consequence,
   asserted in the test: the *same* final position offers Black `d4e3` when the
   double push is White's second move and not when it is the first. A faithful
   copy of their rule needs up to two ep squares at once, which no FEN can
   carry, so it would mean hand-rolled Black move generation. Frequency: the
   opportunity never arises in the other 2,966 games.
2. **White may not END its turn with its own king attacked** (the
   forced-blunder exception, where every option hangs, is unchanged and lives in
   `_get_white_actions`). playstrategy accepted `12. Ke6,f5#` in game
   `gaBbbLhv` with White's king on the `Bc8–d7–e6` diagonal and scored it 1-0;
   under our rules Black replies `Bxe6`, captures the king, and **wins** — the
   recorded result inverts. Their engine appears to use ordinary check/mate
   semantics in which the White king can never be captured, which is a deeper
   difference than "their checkmate is our king capture one ply later."

---

## 2. The owner's binding rules

Learned the hard way; violating any of these has cost real time.

| Rule | Detail |
|---|---|
| **Commits** | Owner identity only (`Peng-AP`). **No Claude co-author trailer.** One-line messages. |
| **Push** | Never unless asked. `main` is far ahead of `origin` by his choice. |
| **Long / multi-worker jobs** | Explicit go only. **Never while he is playing** — matches eat 6 workers. |
| **Gates** | **Never weaken a threshold to let a recipe through.** Per-side floor **0.40** on every leg of every opponent, aggregate must beat 0.50. |
| **Versions** | A number needs automated evidence **plus** his playtest. Rejected → `models/rejected/`, the number stays free. |
| **Metrics** | He rejected proxy scorecards outright: *"my eval is not replaceable."* Do not build automated substitutes for his judgement. |
| **His observations** | Confirmed by measurement **every single time** checked. Debug the code first; never hedge — measure and report the number. |

---

## 3. Established laws

Findings that should not be re-derived.

1. **Pawn-phase cliff.** Black converts 91–100% with White's pawns gone, 7–14%
   from 3–4-pawn positions the owner wins 100% of. This is *the* problem.
2. **Echo chamber (outcomes).** His Black wins double as demonstrations of
   AI-White losing. Fix in place: `policy_weight_for_record` masks human-game AI
   moves so only the winning human's moves teach policy.
3. **Value saturation was the kneecap.** Fix is the end-anchored ramp:
   `result * γ^min(plies_to_end, horizon)`, floor 0.5 / horizon 60, scalar head.
   Measured: fraction of labels at |v| > 0.9 — v17 **75.5%**, ramp **10.3%**.
4. **Label shaping is invisible under the WDL head.** The ramp requires
   `--value-head scalar`.
5. **Near-mate labels poison Black when combined with the ramp** — any order,
   any dose. Blending and fine-tuning are dead ends.
6. **Ruled out as the cliff's cause:** search depth (800 sims → 0.70),
   ramp-further, encoding width, the value-head architecture (§4.2), and the
   human duplication multiple (§4.1). **Capacity is not ruled out** — the
   `v18_cap` run widened the tower 45% but total params only 8% and never
   touched the value head.
7. **Offline metrics and play strength are decoupled.** Demonstrated four times,
   most recently §4.1.
8. **Aggregates mask per-side collapse.** Burned the project four times. Always
   report W/B separately.
9. **The heuristic anchor is saturated for White**; only its Black leg informs.

### 3.1 A tenth, new on 2026-07-26: the echo chamber has a second form

Law 2 is about *outcomes*. There is also **style transfer through the policy
head**, which the outcome mask does not touch.

The owner played the dup1 arm and said its White play "echoes a structure I had a
lot in my data games." Measured: of all **1,768 games** in `combined_v17`, only
**2** share ≥3 opening positions with the AI's White play, and **both are his
own human games** — 0 of 1,086 self-play, 0 of 225 whitefocus, 0 of 201
blackfocus, 0 of 160 promo_races. Deepest match runs to ply 7 →
`data/raw/human_games/white_2026_07/game_00007.jsonl`, a game **he won as
White**.

That is why the mask cannot help: those are moves from a game he won, so they
are legitimate policy teachers by design. **His repertoire is effectively the
corpus's only source of opening variety at this depth**, and the model plays it
back at him. His own gloss on the position it reached: *"technically good for
black (just black model screws it up)."*

This is the strongest argument for `ps_monster` (§7.1) — 829 games of opening
structures the project has none of.

---

## 4. What the last two days established

### 4.1 The duplication hypothesis is dead (2026-07-26)

`combined_v17` carries the 96 human games at **6× in-file duplication**. Both
v18 arms died on the vs-ramp **Black** leg at 0.30, the corpus was implicated
over the architecture, and duplication was the most-indicated mechanism. Tested
properly:

`tools/set_human_duplication.py --copies 1` → `data/raw/combined_v17_dup1` →
`data/processed/combined_v17_dup1_r50h60`. Verified single-variable: only
`human_games/` differs, all **1,673** other files byte-identical, and the
processed pair shares the **same 1,175 train games, same 80 human games, same
split, same seed 42, same ramp labels**. Training command was the v18 GAP arm's
verbatim with only `--data-dir` changed. Gate protocol verbatim.

| leg | **dup1 (1×)** | gap (6×) | spatial (6×) |
|---|---|---|---|
| vs ramp (W/B) | **0.70** (1.00 / **0.40**) | 0.60 (0.90 / 0.30) | 0.60 (0.90 / 0.30) |
| vs v17 (W/B) | **0.50** (0.65 / **0.35**) | 0.60 (0.70 / 0.50) | 0.70 (0.80 / 0.60) |
| anchor | 0.65 (B **0.40**) | 0.80 (B 0.60) | 0.65 (B 0.30) |

**Verdict FAIL** — `vs_v17 aggregate 0.50 <= 0.50`, `vs_v17 black leg 0.35 <
0.40`. Evidence: `benchmarks/gate_dup1_result_20260726.json`, two
`match_v18_dup1_*` JSONs, `benchmark_best_value_net_20260726_043056.json`.
Report and driver in history at `75cbc74`.

**Read it carefully.** The deciding leg moved *exactly as predicted* — vs-ramp
Black 0.30 → 0.40 — and it means nothing: that is one game in ten, and the other
two Black legs moved the opposite way. Totalled across all three legs, Black
scores **11.5/30 for dup1 against 14/30 for gap** — a 2.5-game difference on
n=30 against SE ≈ 2.7. De-duplication did not rescue Black, and the evidence
mildly favours the duplication having *helped*. Looking only at the leg the
experiment was designed around would have produced a false win.

Note also dup1's **White leg vs ramp was 1.00** — ten from ten against the
strongest engine on record — while its Black leg in the same match was 0.40. The
aggregate reads 0.70 and hides that. Law 8 again.

**Offline metrics disagreed, again.** dup1 beat gap on every test metric (policy
CE 2.4386 vs 2.6397, value MAE 0.2014 vs 0.2383, winner-sign 0.9250 vs 0.9111)
and lost the gate. The two test sets are not the same data — dup1's contains the
human games once, gap's six times — so that comparison was never sound.

### 4.2 The architecture A/B was a null result (2026-07-25)

One corpus, two value heads, everything else identical.

- **GAP** (default): `AdaptiveAvgPool2d(1) → 128 → 64 → 1 → tanh`.
- **Spatial** (`--spatial-value-head`): `Conv 1×1 → BN → ReLU → Flatten(Cv*64)
  → 256 → 1 → tanh`, keeping the 8×8 layout. +504,128 params (10,285,697 →
  10,789,825, ~5%).

Motivation was measured, not aesthetic: the GAP feature vector has an
**effective rank of ~6** (v17 6.0 / ramp 7.1) against **~481/494** for the
policy head's spatial input, and in this variant *where* a pawn sits decides
everything.

Both arms read **Black 0.30 vs ramp** — identical, not merely similar. A change
that rebuilds the value signal's entire geometry moved the deciding leg by
**zero**. Spatial did win offline (value MAE 0.2277 vs 0.2383, sign 0.9198 vs
0.9111) and paid in policy CE (2.7572 vs 2.6397).

**Retraction to be aware of:** an earlier claim that "spatial beats GAP on every
leg" was false. On the anchor's Black leg — the only part law 9 says informs —
spatial is **0.30 against GAP's 0.60**. Different trade, not an improvement.

Other measurements from that audit, all reproducible: `policy_fc` alone is
**8.40M of 10.3M params (81.5%)**, the residual tower 18%, the value head 0.2%.

### 4.3 A live bug found in the owner's play path (2026-07-25)

`parse_move`'s manual-SAN fallback rebuilt each legal move's SAN **without
file/rank disambiguators** and returned the first match from an *unordered set*.
With two queens bearing on one square, "Qa5" matched both and it picked one
arbitrarily. Not import-only — this is the function the notebook and `play.py`
use for the owner's typed moves, and two queens on one square is a promotion
endgame.

Fixed at `b408e66` by splitting out `parse_move_candidates` (0 = illegal, 2+ =
ambiguous; `parse_move` returns None rather than guessing) plus an "Ambiguous —
did you mean Qaa5 or Qca5?" prompt. Regression test built on the real position
from playstrategy game `D3p2sAyu`.

Blast radius **measured**: across all **139,416 plies in 2,966 games**, exactly
**1 ambiguous ply in 1 game**, and that game is not one of the 829 imported.
`data/raw/ps_monster` re-imported byte-identically after the fix, proving the
corpus was never corrupted.

### 4.4 The owner's blunder report, diagnosed — still open (2026-07-26)

He reported dup1-as-Black "inexplicably refusing to capture a pawn about to
promote." Confirmed, at
`data/raw/human_games/white_2026_07/game_00014.jsonl` ply 9:

```
r1bqkb1r/pPpp1ppp/5n2/8/4PP2/4K3/8/8 b kq - 0 5
```

Verified by the engine's own generator: after `Bxb7` White has **no** move from
b7; after the played `Rb8`, White has `b7xc8=Q`. A blunder, not a judgement call.

**dup1 is the only model of four that picks Rb8** (400 sims, no root noise). It
rates the blunder **Q=−0.1478** against the capture's **Q=−0.3235**. gap, ramp
and v17 all capture. **This is the value head, not the search** — MCTS
faithfully maximised a wrong value function. Root priors for the capture:
dup1 0.0907, gap 0.1147, ramp 0.1172, **v17 0.5834** (v17 spent 93% of visits
there).

Two hypotheses were tested and **both failed**, which is why this is still open:

- **`_king_safety_override` does not veto the capture.** Both moves pass
  `_hangs_king`; only `d8e7`/`e8e7` are vetoed. The override is innocent.
- **No population-wide prior deficit.** A 400-position static probe (Black to
  move, White pawn on the 7th, capture available) has dup1 **mid-pack**:
  capture-is-top-1 42.0%, vs gap 46.0% / v17 41.2% / ramp 40.2%.

So an n=1 value failure is confirmed and its cause is unknown. The untested
measurement is an **MCTS-based** probe at population scale — the static one only
measures priors and the failure lives in Q. Needs workers (§8.1).

### 4.5 Incidental: ramp's value head is systematically optimistic

Across those same 400 promotion-defense positions, mean value (Black POV):
**ramp −0.02**, dup1 −0.33, gap −0.35, v17 −0.47. On the owner's position ramp
read **+0.58** with a White pawn on b7 about to queen.

The strongest engine on record is by far the most optimistic for Black exactly
where Black must defend a promotion — and it is the second gate opponent. Not
explained. Could be insight, could be miscalibration; nobody has checked.

### 4.6 Epoch headroom was never tested, and the record said otherwise

Raised by the owner mid-run. `patience = 10` is a **bare local at
`src/train.py:887`** — no CLI flag, no config constant, not in the run manifest.
Early stopping therefore needs ten consecutive non-improving epochs, so under a
30-epoch cap **any best epoch above 20 cannot have early-stopped** — it ran out
of cap.

| run | best epoch | ended by |
|---|---|---|
| `v18_ramp` | 20 | patience, exactly at 30 |
| `v18_ramp2` | 24 | **cap** |
| `v18_detox` (wdl) | 27 | **cap** |
| `v18_hybrid` | 26 | **cap** |
| `v18_ramp17_gap` | 25 | **cap** |
| `v18_ramp17_spatial` | 21 | **cap** |
| `v18_dup1` | 27 | **cap** |

Six of seven ended at the cap. The memory `training-run-audit-facts` asserted
the opposite ("no epoch headroom exists for the ramp recipe... both ramp runs
early-stopped on their own") and has been retracted in place.

**Cap-limited does not prove under-training** — the best epoch might still be 25
with a cap of 60. It proves the question was never asked while being recorded as
answered.

Cheap to make comparable: `StepLR(step_size=1, gamma=0.95)` is epoch-indexed and
**independent of `--epochs`**, and the shuffle generator is seeded
`args.seed + epoch`, so **the first 30 epochs of an 80-epoch run are the same
trajectory as a 30-epoch run**. Raising the cap extends rather than
re-specifies. (Under a cosine schedule with `T_max=epochs` this would not hold.)
The run ends at `min(cap, best + 10)`; at 8.25 min/epoch that is 5.1h if
improvements stop at 27, 11h if they run to 80. LR decay for reference:
epoch 30 → 5.3e-4, 45 → 2.4e-4, 60 → 1.1e-4, 80 → 4.0e-5, i.e. the schedule
self-extinguishes around 70–90 and implies its own budget.

**Owner decision 2026-07-26: parked for the GPU box ("in a week").** This box is
`torch 2.12.1+cpu`. Do not spend a CPU night on it. Add `--patience` first.

---

## 5. The PlayStrategy corpus

### 5.1 What it is and why it is usable

`playstrategy.org` (a Lichess fork) has supported Monster since Dec 2023 and
exposes a **public API with no authentication**. Census: **8,566 distinct
Monster games, 2,971 human-vs-human**, 309 crawled users, 2023–2026 — about 31×
the owner's 96 human games.

Compatibility was **verified, not assumed**: their starting FEN is byte-identical
to `config.STARTING_FEN`; White's double move is a comma pair in their PGN
(`1. f4,Kf2 e6`) matching our atomic `(m1,m2)` action; **2,966 of 2,971 games
replay with zero errors** through `MonsterChessGame` + `parse_move`, including
king captures, moves through check, and promotions. Their checkmate is usually
our forced king capture one turn later (enumerated on a real game: all 12 Black
replies lost the king), so `Result` maps to `game_result` at a cost of ~2 plies,
worth γ² ≈ 0.977 on the ramp label — with the `gaBbbLhv` exception in §1.1.

**All five replay failures are diagnosed** (2026-07-26), and each self-detects as
a failed replay, so nothing entered the corpus silently:

| game | cause | ruleset divergence? |
|---|---|---|
| `eoJPDAsf`, `5z1jpaeR` | 159 and 171 turns, past `MAX_GAME_TURNS = 150` | no |
| `D3p2sAyu` | our own SAN ambiguity bug (§4.3) | no — ours |
| `BDarxSwF` | en passant across White's double move | **yes** (§1.1) |
| `gaBbbLhv` | White ended its turn with its king attacked | **yes** (§1.1) |

### 5.2 The bots are worthless and are excluded

Three bots (`PST-Greedy-Tom`, `PS-Greedy-Two-Move`, `PS-Greedy-One-Move`)
account for 5,595 of the games. Humans score **0.998 as White** (736W-1D-1L) and
**0.986 as Black** over 1,460 of them; median bot rating **979**; on 94 bot turns
the bot matched our heuristic's static-eval-best move **34%** of the time
against ~3% chance — greedy family, no search.

**Do not salvage the human side of bot games either.** Winning 99% against an
opponent that punishes nothing means those moves were never tested — the echo
chamber with a weaker partner.

### 5.3 The shipped dataset

```
py -3 tools/import_playstrategy.py --bundle data/playstrategy/ps_games.json \
    --min-elo 1600 --manifest data/playstrategy/ps_dataset_manifest.json \
    --out-dir data/raw/ps_monster
```

| | |
|---|---|
| Games | **829** of 2,966 |
| Records | **43,939** (→ 87,878 positions after mirroring) |
| Filter | **both** players Elo ≥ 1600 |
| Balance | **none** — owner's decision |
| Draws | excluded (40 games) |
| Policy weighting | winner-only by default (`policy_weight` 1.0/0.0); `--all-moves` overrides |
| Value | `mcts_value` is **0.0** — PGN carries no search output. These are policy teachers with outcome labels. Do not read them as if they carried search information. |

**Owner decisions embedded here.** (1) Filter on **Elo, not win rate** — score in
a closed pool averages 0.5 by construction, so a high win rate can just mean weak
opposition; **both** players must clear the bar, because with winner-only
teaching a strong player beating a 900 teaches untested moves, the same defect as
the bots. (2) **No side balancing**: *"If ELO is high, don't worry about
balancing the sides. Black should be better, anyways."* A balanced variant (783
games, exactly 13,492 positions per side) was built first and discarded on his
instruction.

**Record granularity changed 2026-07-26** (`b6bfd1a`). A White turn now emits
**two half-move records** (`half` 0 and 1) like the self-play sources. It
previously emitted one record per turn with a `"m1,m2"` policy key, which
`policy_dict_to_target` marginalizes to m1 — so White's second move was imported
and then discarded, in a variant defined by the second move. Record count went
29,140 → 43,939; the 57-record gap between `half=0` (14,856) and `half=1`
(14,799) is turns that ended on the first move via king capture. Pinned by
`tests/test_import_playstrategy_halves.py`.

**Note the asymmetry this exposed.** The owner's own notebook-written human games
are still one-record-per-turn and **cannot** be fixed retroactively — the
notebook only ever saved a visit distribution for m1. So *no* human-sourced data
teaches White's second move except ps_monster. Verified by survey of White
records: self-play sources are exactly balanced `half=0`/`half=1` ((root)
12,438/12,437, blackfocus 3,404/3,404, promo_races 2,055/2,055, whitefocus
1,914/1,914), while `human_games` has 1,302 pair records and 5,262 single records
and **no `half` field at all**.

### 5.4 What he was told and has not acted on

Black's share of *wins* does **not** rise with Elo in this data — 37.9% / 38.1% /
38.6% / 40.5% / 35.9% / 38.6% at 0/1400/1500/1600/1700/1800, with White winning
~60% of decisive games at every band. So this corpus will not *demonstrate*
Black being better; it is 1600–1800 blitz, not correct play. Black is nonetheless
the **position** majority (53.7%) because Black's wins run longer.

Concentration: at the 1600 cut the top two players are 27% of player-slots (down
from ~50% unfiltered). `oruro` (429) and `woll` (185) dominate the unfiltered
pool.

### 5.5 Tooling and browsable viewer

| tool | purpose |
|---|---|
| `tools/census_playstrategy.py` | snowball crawl of the Monster population; skips BOT accounts; resumable |
| `tools/build_ps_bundle.py` | fetch each PGN, replay-validate, emit `ps_games.json` |
| `tools/import_playstrategy.py` | convert to our JSONL schema; `--bundle` (no network), `--min-elo`, `--balance`, `--manifest`, `--all-moves`, `--include-bots`, `--include-draws` |

Viewer, all 2,966 games with per-player head-to-head records and a replayable
board:
**https://claude.ai/code/artifact/427b80d2-9da6-4721-832a-d121b9af3057**
(current as of 2026-07-26 — shows the shipped 829-game selection). The build
script lives in a session scratchpad, not the repo.

**API notes.** No auth. Rate-limit ~1 req/s, set a real User-Agent. Use
`?perfType=monster` (not `variant=`). `/game/export/{id}` embeds
`{ [%clk 0:03:00] }` which contains `]` — split movetext on the header blank
line, **never `rsplit("]")`**.

---

## 6. Current state, verified 2026-07-26 16:00

**Tests:** 166, all pass, ~11 s — `py -3 -m unittest discover -s tests`

**Git:** clean but for `src/play.ipynb` (modified by the owner's play session,
+66/−6 — deliberately untouched). Recent commits:

```
b6bfd1a  PlayStrategy import: emit White half-move records
5902d40  Keep the dup1 gate verdict in benchmarks/
ac02e91  Retire the concluded dup1 driver and report
75cbc74  dup1 control arm rejected
c97c753  Pin the two playstrategy rules divergences as settled decisions
1c7d297  Tool to rebuild a corpus at a chosen human duplication multiple
b408e66  parse_move: refuse ambiguous SAN
e5e3f30  Retire the concluded v18 driver and report
78f97f8  PlayStrategy import toolchain
11beff9  v18 run evidence: both arms rejected
```

**Models.** `models/candidates/` is **empty**. `models/rejected/` has 15,
including `v18_dup1`, `v18_ramp17_gap`, `v18_ramp17_spatial`. Nothing under
`models/` is tracked.

**Corpora.**

| path | what |
|---|---|
| `data/raw/combined_v17` | the incumbent recipe: v16 + 13 owner games, human at symmetric 6× |
| `data/processed/combined_v17_r50h60` | what both v18 arms trained on (ramp labels, 15ch, seed 42) |
| `data/raw/combined_v17_dup1` + `data/processed/combined_v17_dup1_r50h60` | the concluded 1× control (§4.1) |
| `data/raw/ps_monster` | **829 PlayStrategy games, rebuilt with half-move records** |
| `data/playstrategy/` | census + bundle + manifest cache, gitignored, regenerable |

**Human games:** `white_2026_07` 15, `black_2026_07` 24, `curriculum_2026_07`
52, plus 4+3 from 2026-03. The two newest are the 2026-07-26 dup1 playtest:
`black_2026_07/game_00023.jsonl` (AI as White — the echo game) and
`white_2026_07/game_00014.jsonl` (AI as Black — the refused capture). Neither is
in any corpus yet. `white_2026_07/game_00013.jsonl` carries a **hand-corrected
outcome label** (−1 → +1 after a mistyped final move; all 26 Black replies
enumerated as losing) documented in `game_00013.correction.md` — the precedent is
still unsanctioned (§7.4).

**Corpus composition** (`combined_v17`, raw records, measured):

| | 6× human | 1× human |
|---|---|---|
| total records | 72,629 | 61,959 |
| human_games share | 18.1% | 4.0% |
| Black to move | 36.4% | 34.0% |
| Black-win-labeled | 54.8% | 50.0% |
| **pawn phase (wP≥3)** | **9.4%** | **7.3%** |

Human games are **22–25% pawn phase internally** against ~9% corpus-wide, so at
6× they supply **42.8% of every wP≥3 position in the corpus** while being 18% of
it; at 1× that falls to 13.7%. Also: all 9,675 move-limit endings carry −0.5 and
none carry +0.5 or 0 — no game reached the cap with the heuristic favouring
White.

**Disk.** `data/processed/` holds ~13.4 GB across seven datasets, of which
`combined_v13` (2.6G), `combined_v16` (2.7G), `combined_v16_r50h60` (2.7G) and
`combined_v17_dup1_r50h60` (2.4G) belong to concluded work. All are regenerable
from `data/raw/` in ~20 min each. **Not deleted — data removal is the owner's
call.**

---

## 7. Open decisions — all need the owner

### 7.1 Whether to merge `ps_monster`, and at what weight

**The big one, and now the only untried lever.** The case has strengthened twice:

- It is **41.2% pawn phase** against this corpus's 9.4%. Merging at 1× takes the
  corpus to ~18–19% and raises raw wP≥3 records from 6,814 to ~19,000. No source
  the project has ever had is remotely this dense in the phase that defines the
  open problem.
- Law 3.1: his repertoire is the corpus's **only** source of opening variety, and
  ps_monster brings 829 games of structures the project has none of.
- Its side balance is 51/49 White/Black to move against the corpus's 63.6/36.4.
- Position-level label mix barely moves: merging shifts Black-win-labeled from
  54.8% to ~54.5%. The "White wins 60% of their games" worry is a game-level
  artifact that does not survive to the position level, because Black's wins run
  longer.

**The honest counterweight.** In the pawn phase specifically, ps_monster's
outcome labels say Black wins **39.7%** of the time (against `combined_v17`'s
36.3% — marginally *better*, not worse). But the owner wins those positions
**100%** of the time. So merging roughly triples the evidence for a conversion
rate he demonstrably exceeds. That splits the cliff into a testable fork:

- **If the cliff is a knowledge problem** (Black doesn't know the technique),
  this is the best data the project has ever had and the gain shows in policy.
- **If the cliff is a value-head belief problem** (Black evaluates the phase as
  lost and plays accordingly), merging deepens it.

At 43,939 records ps would be **37.7%** of a 1× merge with `combined_v17` — the
largest single-source composition change in the project's history, and the last
data addition (13 games at 6×) measurably moved Black −0.063 top-1.

**Splitting the fork directly is not currently expressible.** The pipeline has
`policy_weights.npy` but no value-weight equivalent, so "teach policy from
ps_monster, don't teach value" would need a new `value_weights.npy` path
mirroring the policy one. Bounded change; not yet scoped.

### 7.2 Any further training run

Long-job rule. Nothing is queued and the box is idle.

### 7.3 Whether to widen gate matches to 40 games

At n=10 per leg a truly 0.50 model trips the 0.40 floor **17.2%** of the time per
leg; across four legs that compounds to ~53% (upper bound — legs correlate). At
n=20 per side the per-leg rate at true 0.6 falls 5.5% → 2.1%. **The floor itself
stays at 0.40 — never relax it.** §4.1 is a concrete case where n=10 could not
separate signal from noise.

### 7.4 A one-time blessing of the hand-corrected-label precedent

`white_2026_07/game_00013`. Done correctly and documented, but it should be
explicitly sanctioned rather than becoming ambient practice.

### 7.5 The 5060 Ti box

Owner 2026-07-26: **"in a week."** Queued for it: the epoch-headroom run (§4.6)
and the cliff-vs-sims experiment.

---

## 8. Ready to run, needing no owner time

1. **MCTS-based promotion-defense probe** across dup1/gap/ramp/v17 — the static
   probe (§4.4) only measures priors and the failure lives in Q. Uses workers, so
   not while he is playing. ~1h.
2. **Add `--patience` to `train.py`** — prerequisite for §4.6, ~4 lines.
3. **`make_promo_deck.py` re-aim** — never decided, still wrong: 153 of 163
   positions are the owner promoting *as Black*, the opposite of the side that
   needs work.
4. **Explain ramp's value optimism** (§4.5) — nobody has looked.

## 9. Do not do

- Further blending, fine-tuning, or label-shape variation on the existing corpus
  (law 5).
- The policy-upweighting fix. **Falsified:** on the owner's White games ramp's
  `policy_top1_white` is **0.786** against v17's **0.614** — the predicted
  deficit does not exist.
- The side-specialized-heads run. Rationale withdrawn: it rested on a retracted
  seesaw claim, and §4.2 measured the architecture lever as null. Restoring it
  as-deleted builds three policy heads, **+16,843,394 params → 27,129,091 total
  (2.64×)**, of which **8,396,864 receive no gradient** (confirmed:
  `policy_fc.weight.grad` all-zero after backward).
- Any automated proxy for the owner's judgement.
- Relaxing any gate threshold.

---

## 10. Operational notes

### 10.1 Detached execution (session-death-proof)

Agent-session background jobs die with the session. Working pattern:

```powershell
$su = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine='cmd /c py -3 -u driver.py >> driver_stdout.log 2>&1'
  CurrentDirectory='C:\Users\AaronPeng\Desktop\monster-chess-nn'
  ProcessStartupInformation=$su }
```

Inside the driver: `signal.signal(signal.SIGINT, signal.SIG_IGN)`,
`SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)`, and
`except BaseException` → log `CHAIN ABORTED`. `ShowWindow=0` matters.

Three traps, each of which bit in practice:

- **Never `sys.exit()` inside that `try`.** `SystemExit` is a `BaseException`, so
  a clean run logs the failure marker. Call `main()`, then exit outside it.
- **Do not give the child `stdout=PIPE`** if you want to watch it —
  `subprocess.run` holds every line until exit, leaving a 4-hour run
  unobservable. Inherit stdout.
- **Rehearse the whole chain at tiny scale first.** Doing so caught the v18
  driver reading the wrong arm's anchor (both wrote
  `benchmark_best_value_net_*.json`; fixed with an mtime floor per stage) and
  caught the dup1 gate driver reading `benchmark.py`'s schema for a `match.py`
  file. **The two schemas differ:** `match.py` emits
  `a_score`/`a_as_white`/`a_as_black`; `benchmark.py` emits
  `candidate_score`/`white_strength`/`black_strength`. Confusing them yields
  `None` on every leg and fails the gate on a parsing bug.

### 10.2 Watching a detached job from an agent session

**`kill -0 <windows_pid>` in Git Bash does not work** — it operates on MSYS PIDs
and reports dead for a living process. An `until ... || ! kill -0 PID` loop exits
on its first pass and then happily prints the newest *stale* result file, which
looks exactly like a real result. Watch for the concrete artifact (a result JSON
newer than launch) and bound the wait; check liveness through PowerShell/CIM.

Also: `--epochs 30` on the command line contains the substring "epoch", so a
watcher grepping for `epoch` fires the instant the job starts. Match on `^Epoch`
or on the artifact.

### 10.3 Repo conventions

Root holds **only** the ACTIVE overnight driver + report. On conclusion: commit
the evidence, `git rm` driver and report so history is the archive, delete logs.
Durable evidence lives in `benchmarks/`. Rejected candidates go to
`models/rejected/` and the version number stays free.

### 10.4 Timings on this box (`torch 2.12.1+cpu`, `cuda False`)

30-epoch training **~245 min** (8.25 min/epoch); 20-game match **34–60 min**;
heuristic anchor **25–46 min** — the variance is real and comes from how long
Black's games run (long Black conversions are the slow case);
`sec_per_decision` ~1.3 at 400 sims. Check `CPU`/`StartTime` on the python
process before declaring a stall.

### 10.5 PowerShell / Bash quirks

- `2>$null` on native commands fakes exit 255.
- Piping file content into `py` mangles encoding — read the file inside Python.
- **Bash heredocs mangle `\n` inside Python string literals.** Write the script
  to a file with the Write tool instead. This has bitten three times.
- Notebook round-trip: `json.dumps(nb, indent=1, ensure_ascii=False) + "\n"`
  reproduces `src/play.ipynb` exactly; use targeted string replaces.

### 10.6 Worktree

`phase5-deletions` branch at `C:/Users/AaronPeng/Desktop/monster-chess-phase5`.

---

## 11. Memory files

`black-winning-with-correct-play`, `pawn-phase-cliff` (the running campaign
log), `commit-style-preference`, `measurement-and-heuristic-findings`,
`policy-prior-hypothesis-falsified`, `architecture-audit-2026-07-25`,
`training-run-audit-facts` (**contains a retraction — read it**),
`owner-observations-are-ground-truth`, `rework-plan-execution-state`,
`desktop-compute-plan`, `playstrategy-corpus`,
`v18-architecture-ab-null-result`.

---

## 12. Process notes for whoever picks this up

Two failure modes have cost this project more than any bad recipe.

**Reading the leg the experiment was designed around.** §4.1 would have been a
clean win by that standard. Always total the per-side legs and compare the
difference against the noise floor before believing a direction.

**Believing a summary over the artifact.** Every load-bearing claim in previous
handoffs that turned out wrong — the White/Black "seesaw", "spatial beats GAP on
every leg", "no epoch headroom exists", "ps records match the notebook's human
format" — was written confidently and then contradicted by ten minutes of
measurement. Three separate times a clean number came from testing the easy
path: two parser bugs that passed validation against an endpoint that did not
exercise them, and two blunder metrics reading 0.0% because the failure class was
*illegal* in the source ruleset. **Distrust any success rate that has not been
tested against the hard case**, including the ones in this document.
