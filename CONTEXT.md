# Monster Chess NN — context

**The single durable reference: rules, vocabulary, laws, state, data, and
operational knowledge.** The live campaign and next steps are `DIRECTIVE.md`;
how to run things is `README.md`. Retired documents (`HANDOFF.md`,
`OVERNIGHT_REPORT.md`, `PHASE0_REPORT.md`, concluded directives and drivers)
live in git history, and every measured claim below is backed by a JSON
artifact in `benchmarks/`.

---

## 1. The game, exactly

White: king + 4 pawns on c2–f2, **two moves per turn**. Black: full army, one
move. No castling for White.

`monster_chess.is_terminal` fires **only on king absence** — no checkmate; a
king must be captured (owner rules correction 2026-07-04). Pinned by
`tests/test_king_capture_rules.py`:

- King capture wins **unconditionally**, even if the capturer's own king is
  left attacked — the game ends before any reply.
- White in check may still spend both moves capturing the Black king.
- A "defended" Black king inside White's double-move range is **not** safe.
- White's **first** half-move may pass through check; only the completed turn
  must leave White's king safe.

Move limit `MAX_GAME_TURNS = 150`, with a symmetric ±0.5/0 relabel by
heuristic sign at the cap.

Two settled divergences from playstrategy.org, pinned by
`tests/test_ruleset_divergences.py` (both decided in favour of our behaviour):
en passant is conferred only by the **last** move of White's turn; White may
not **end** its turn with its own king attacked (forced-blunder exception
unchanged, in `_get_white_actions`).

## 2. Where the project stands (2026-08-15)

**The bootstrap loop works, and its defect was never the game.** Two apparatus
faults cost more than every recipe idea combined: fine-tuning from the champion
instead of training fresh (~31 Elo, 800 games, CI [0.5104, 0.5796]) and offline
checkpoint selection instead of play-based (~52 Elo, z=+4.17). A single 35-minute
from-scratch run beat the model five generations of fine-tuning had produced.

**The current bottleneck is CONVERSION, not strength and not the colour gap.**
Measured 2026-08-15 on full games from the true opening at 1600 sims: **every
draw is a cap draw** — all exactly 225 plies, while decisive games top out at
147, with nothing in between. In all 24 sampled capped games Black finished
ahead on material (mean **+26.4**, White on a **bare king in 21**) while both
sides shuffled: a mean of 12.6 distinct positions in the last 100 plies. The
exact solver proved **6 of those 24 games contained a forced king capture within
four Black moves** that search walked past — with **zero budget exhaustion**, so
the negatives are proven too. These are unconverted wins, not fortresses. See
`REPORT.md` §30.

**The colour gap is largely a search artifact.** On identical positions with
sims as the only variable, Gen9's self-play gap closes **0.250 -> 0.067** from
400 to 1600 sims and v21b's closes **0.133 -> 0.033** — both by about 74%. The
two models get there by opposite means: Gen9 converts White's lost wins into
Black **wins**, v21b into **draws** (§29.2). Deeper search benefits whichever
side plays Black; that is a fact about the game, not about a model.

| model | role |
|---|---|
| `models/fresh_start_v21b` | **the bar.** Owner: *"it'll be the gate but I'm not impressed enough for it to be 22."* Playtest still outstanding. |
| `models/fresh_start_v21` | holds the version number. The bar and the number are separate again, exactly as when v17 held the number and v18_ramp was the bar. |
| `models/candidates/gen7_scratch/screen_nominee.pt` | passed the gate against v21b (pooled 0.5600 over 400 games, z=+2.40); the working bar inside the bootstrap loop. |
| `models/candidates/gen8_scratch/screen_nominee.pt` | paired re-screen selected epoch 7; it passed the first Gen7 leg but failed fresh confirmation at Black 0.3500. No promotion. |
| `models/candidates/gen9_scratch/screen_nominee.pt` | **passed the complete paired gate against Gen7.** It scored 0.5950 (W 0.7200/B 0.4700), then confirmed at 0.5575 (W 0.7050/B 0.4100). Leading v23 candidate; owner playtest and numbering remain outstanding. |
| `models/candidates/gen10_scratch/screen_nominee.pt` | rejected. It scored 0.5225 against Gen9 but Black 0.3850 failed the unchanged 0.40 floor; seed-43 also failed at 0.4925/W 0.6300/B 0.3550. No confirmation was earned. |

**The loop has now produced a clean successor.** Gen8 remained a useful data
increment but did not survive a fresh paired confirmation. Gen9 generated from
the unbeaten Gen7 bar, accumulated Gen7–Gen9 replay, trained fresh, selected
epoch 6 by worst-colour calibrated play, and cleared every binding leg twice.
The absolute 0.40 floor was not changed. See `REPORT.md` §§26–27.

**Opening diversity was measured directly, not inferred.** The historical
800-game sampled v21b–v21 re-anchor contained 650 unique states and 684 unique
state+model-colour games (85.5% effective), so its standard error needs only a
1.08× correction, not the feared ~2×. A paired-book re-anchor reproduced the
same result (0.5238 versus 0.5225 sampled). Current screens and gates use pinned,
mixed-provenance books with disjoint selection and test blocks (`REPORT.md`
§24.3). Gen10 then demonstrated that replay accumulation is not monotonically
improving: its best balanced checkpoint traded White strength for a small Black
gain and failed the Gen9 gate (`REPORT.md` §28).

**Gen9 post-gate diagnostics.** Its paired self-skew is W 0.6675/B 0.3325 over
400 games. Directly against v21b it scores 0.5763 overall, W 0.6975/B 0.4550
(400 games, paired SE 0.0157). Gen9 remains the clear successor candidate after
both Gen10 seed paths failed.

**Opening books start games already decided.** The p16 book used by screens
and gates has White down **1.3 of its 4 pawns** on average, with both armies
intact in only **3%** of its 800 entries. An 8-ply book has 78% intact. Pairing
controls for it, but the measurement begins from lopsided middlegames rather
than the opening. A shallow book needs **temperature 1.0**, not 0.5 — the
earlier 8-ply build failure (31 of 100 unique) was sampling temperature, not a
reachability ceiling; at 1.0 it produced 60 unique entries in 34 seconds with
zero duplicates (`REPORT.md` §31).

**Next strength experiment:** use the currentized multi-fidelity tuner on the
Gen10 corpus against Gen9, with paired book entries 680–747 reserved for trial
ranking. Treat training seed as a nuisance variable and send only a tuner
winner to a fresh binding book; do not mine more seeds against the same gate.

### The v20-era ledger (2026-08-05, retained)

- **Incumbent and formal gate bar: `models/fresh_start_v20`.** The owner
  promoted the preserved `lc0b_attention_ema_wide64` checkpoint on 2026-08-05.
  Future candidates must clear v20 twice without moving the established
  0.40 per-color floor or 0.50 aggregate threshold.
- **v20 lineage.** The 32-channel `models/candidates/lc0b_attention_ema` keeps
  v19_B's exact data and training recipe, changing only to the compact
  attention policy head plus EMA (`0.999`) validation/checkpoint weights. It
  passed the unchanged binding gate: against v19_B it scored **0.7125 overall,
  0.925 as White, 0.500 as Black**, then **0.7375 / 0.925 / 0.550** on the
  automatic fresh confirmation. It also scored 0.825 against ramp and 1.000
  against the fixed heuristic anchor. This is the strongest automated result
  on record at that stage and passed the owner's playtest.
- **Owner gate passed 2026-08-05.** The owner found `lc0b_attention_ema` very
  strong on both sides, able to handle his White attack and vicious as White;
  only occasional spotty Black conversion moves remained. It is now the
  approved playing-strength bar for successor work (checkpoint preserved in
  place; no historical artifact was overwritten).
- **Promoted v20 source: `lc0b_attention_ema_wide64`.** The only change is widening
  the attention query/key channels from 32 to 64. Against the approved model
  it improved both colors twice: calibrated **+0.050 Black / +0.025 White**
  at 40 games, 400 sims, then **+0.1625 Black / +0.0625 White** at 80 games,
  800 sims. The campaign copy and all prior bars remain preserved.
- **v20 color-skew baseline.** An 80-game native self-match at 400 sims with
  identical weights/search on both sides produced **47 White wins, 16 Black
  wins, 17 draws**: White score **0.6938**, Black score **0.3063**. This is a
  color baseline, not a head-to-head model-strength result.
- **V21 screen closed without promotion (2026-08-05).** Attention widths 96
  and 128, a side-specific attention adapter, capture-only WDL, and scalar +
  capture-WDL auxiliary heads all failed repeatable positive deltas in both
  colors. The best Black-leaning arm (`v21_mixed_capture_wdl_w003`) produced
  +0.225 Black / +0.000 White once but repeated at +0.075 / **-0.300**.
  V20 remains incumbent. Capture terminal labels are useful pressure toward
  Black conversion but currently trade away White/general strength; see
  `REPORT.md` §12 and the `v21_*_20260805.json` artifacts.
- **Promotion-aware policy is implemented; first isolated candidate rejected
  (2026-08-05).** The legacy 4096 source/destination policy merged q/r/b/n
  promotions. The opt-in 4288 ABI adds 192 distinct promotion logits in both
  Python and native search while old checkpoints remain compatible. A frozen
  v20 lift trained only the 1,548-parameter promotion delta on 3,790 augmented
  rows. Held-out full-policy top-1 on promotion rows rose 7.8% -> 57.8% (val)
  and 4.1% -> 64.5% (test), but the binding gate was 0.5125 then 0.5000 versus
  v20; confirmation Black scored 0.325 below the 0.40 floor. Keep the
  representation fix and reject `v21_promotion_policy_exact` as v21. Do not
  turn obvious tactical givens into phase-specific gates or adapters; they
  must emerge from general training and search.
- **The iterative bootstrap pipeline is implemented (2026-08-05).**
  `src/iterate.py` is now a resumable manifest-driven state machine covering
  self-play, Black-weighted general deep-search reanalysis, isolated processing,
  replay composition on the exact v19_B/V20 anchor, conservative end-to-end
  fine-tuning, offline regression checks, the binding gate, self-color skew,
  and explicit promotion. It never overwrites numbered V20. A live smoke run
  completed generate -> reanalyze -> process with 2 games, 200 positions, one
  deep-search teacher, 402 augmented rows, and no illegal targets. The first
  real generation must hold architecture fixed; moves-left remains opt-in so a
  pipeline result is not confounded with a model change.
- **Bootstrap triage corrected (2026-08-06).** Held-out policy/value comparison
  is advisory by default; imperfect self-play targets are not a strength oracle,
  and only the unchanged binding game gate rejects a successfully trained
  candidate. The first production arm had Black sign accuracy +3.14 points but
  was denied games by a White policy top-1 drop of 1.52 points under the old
  rule. Its recovered game gate scored 0.650 overall / 0.725 White / 0.575
  Black in the first V20 leg, then failed the independent confirmation at
  0.3875 / 0.575 / 0.200. It was therefore rejected for unstable play rather
  than proxy metrics. The loop also honors the measured eight-worker GPU default
  (7.11 decisions/s versus 5.39 at four) instead of the obsolete hard
  four-worker NN cap.
- **Bootstrap production contracts added (2026-08-06).** Processed
  champion-generated data is accepted into a run-local immutable registry
  immediately after processing, independent of whether its candidate later
  passes. Recent accepted generations therefore accumulate across explicit
  `--continue-after-reject` runs. Replay keeps validation/test membership
  unchanged and deterministically smooths only the training indices across
  side, true capture outcome, and material-count phase quantiles. Checkpoint
  selection now compares every epoch against V20 on identical validation rows,
  optimizing worst-color policy/sign gains with fixed incumbent regression
  guards. If every epoch violates those guards, the generation ends as
  `rejected_training`; it cannot fall back to an unsafe or duplicate model.
- **Bootstrap checkpoint play-testing hardened (2026-08-06).** Four production
  generations all received real binding games. Generation four showed that the
  validation proxy chose the wrong stopping point: preserved epoch two passed
  a fresh full gate (V20 0.525 overall / 0.600 White / 0.450 Black, confirmation
  0.6875 / 0.800 / 0.575), while selected epoch four failed. The 80-game,
  800-simulation calibrated read still rejected epoch two: +0.275 White and
  +0.119 aggregate, but -0.0375 Black. Its pooled self-match White score was
  0.781 versus V20's 0.694, confirming increased White skew. The pipeline now
  play-tests every unique preserved epoch, nominates by worst calibrated color,
  and requires an independent calibrated 80x800 improvement in both colors
  after the binding gate. V20 remains champion.
- **First bootstrap successor cleared every automated gate (2026-08-06).**
  `models/candidates/bootstrap_gen5_teacher3200_full/selected_epoch_002.pt`
  keeps the V20 architecture and fine-tunes the full replay with 4x-weighted,
  policy-only teachers searched at 3200 simulations. Its calibrated 80-game,
  800-simulation read improved **+0.0125 Black / +0.1375 White / +0.075
  overall**. The independent full binding protocol then scored 0.650 / 0.825 /
  0.475 against V20 and 0.550 / 0.675 / 0.425 on the fresh V20 confirmation;
  it also retained 0.900 against ramp and 0.950 against the heuristic. Its
  equal-settings self-match reduced pooled White score from V20's 0.6938 to
  **0.5875**. This is the strongest bootstrap successor candidate, but it is
  not V21 until the owner playtests and promotes it.
- **Moves-left follow-up did not clear the Black stability bar (2026-08-06).**
  An otherwise identical auxiliary-head arm learned the remaining-length
  target and produced two positive checkpoints in the full A/B screen. Epoch
  six beat the fixed successor at 80x800 (+0.025 Black / +0.100 White), but
  failed the fresh V20 binding Black leg at 0.375 < 0.400. The earlier,
  Black-leaning epoch three independently read -0.050 Black / +0.1375 White
  against the fixed successor. Keep the head implemented and opt-in; do not
  promote this arm.
- **Bootstrap production audit completed 2026-08-06.** Deep-search teachers
  now inherit the source game's train/validation/test split (the pre-fix demo
  leak was 10/40 and 61/180; both are now zero). Worker phases fail after a
  bounded no-progress interval, generated batches have an explicit completion
  floor, reanalysis/replay publish atomically, replay artifacts are fully
  hashed, phase seeds do not overlap, concurrent loops are locked out, and
  pipeline training memory-maps the large replay arrays. Full discovery now
  passes 522 tests.
- Owner's read after playing the v19-era models: *"White isn't doing
  terribly — the play is coherent and attacking chances are taken; definitely
  improved. Black's defending play is a big improvement as well. **Black
  conversion is a big problem.**"*
- **The live problem is Black conversion**, but the next intervention is
  general iterative learning rather than a post-promotion rule patch. The
  pipeline reserves 60% of deep-search teachers for Black and retains explicit
  per-side promotion floors.
- **The earlier, unpromoted v20-named experiment closed 2026-08-03: both arms
  rejected** (`v20` Black 0.10
  vs v19; `v20w` PASS-then-FAIL — its gate pass was move-limit relabels, Black
  0.50 → **0.30** under captures-only scoring; both in `models/rejected/`).
  Asymmetric generation at this scale did not produce a Black that converts
  against v19; the turn cap tested dead (2.7× moves, identical captures). See
  `REPORT.md`. The engine rewrite and re-baseline are now complete; the active
  campaign is the post-E5 data ladder and separately measured E6 search work.
- **Post-E5 data ladder closed 2026-08-04: all three arms rejected.** Against
  `v19_B`, frozen-corpus base / +e1500 policy-only / +41 owner games scored
  0.250 / 0.100 / 0.075 as Black (overall 0.4750 / 0.3875 / 0.2125). The bar
  remains `v19_B`; the active work is now Black-first search and opt-in
  LC0-derived architecture screens.
- **The rewrite's E0-E5 phases are complete.** Rules, heuristic and encoding
  reached exact parity; native MCTS passed 400/400 move agreement and its
  200-game self-match; the full gate runs in 6.5 minutes instead of ~34. E5
  established the new captures-only/native zero point and moved the gate bar
  to v19_B. The first E6 solver measurement was a null (+0.02 true
  captures at both 1600 and 3200 sims), so it stays off by default.

## 3. Model lifecycle vocabulary

- **Incumbent** — the strongest approved numbered model; the baseline every
  candidate must improve upon. *Avoid:* current model, latest run.
- **Candidate** — a trained model being evaluated for the next unclaimed
  version number. Not yet a version. *Avoid:* version, release.
- **Version** — a numbered model that passed automated evidence **and** the
  owner gate. *Avoid:* run, experiment.
- **Promotion** — the decision that turns a candidate into the next version
  and the new incumbent. *Avoid:* rename, automatic acceptance.
- **Owner gate** — the final playing-strength assessment by the owner.
  Promotion requires it. *Avoid:* optional playtest.
- **Rejected candidate** — failed automated evidence or the owner gate; goes
  to `models/rejected/` and the version number stays free.

## 4. The owner's binding rules

| Rule | Detail |
|---|---|
| **Commits** | Owner identity only (`Peng-AP`). No co-author trailer. One-line messages. |
| **Push** | Never unless asked. `main` is ahead of `origin` by his choice. |
| **Long / multi-worker jobs** | Standing go carried in the active directive. **Never while he is playing.** |
| **Gates** | **Never weaken a threshold to let a recipe through.** Per-side floor 0.40 on every leg, aggregate must beat 0.50 on model legs. Thresholds are constants with no CLI flag, asserted by test. |
| **What counts as a win** | *"A win by time shouldn't be counted the same as win by capturing the king"* (2026-08-03). Only a king capture scores a win; a move-limit ending scores a **draw**, symmetrically. The ±0.5 *training label* is unchanged. **Every gate/match result before 2026-08-03 was computed under the old rule and is not comparable to results after it** — including v19's promotion and the whole v19 ladder. |
| **The bar** | *"Every model should be better than the last, definitively."* The bar is the **strongest engine on record**, not whatever holds the version number, and must be cleared **twice** on independent opening seeds. Now `v21b`, with `v21` holding the number. "Definitively" has a measured meaning: two 40-game reads of one matchup came out 0.575 and 0.725, and on 2026-08-07 three candidates passed a 40-game bar leg and then scored 0.4800 / 0.4825 / 0.5019 over 800 games each. The bar leg is 200 games for that reason. **Evidence changed the sample size; no threshold moved.** |
| **Versions** | A number needs automated evidence **plus** his playtest. |
| **Metrics** | No proxy scorecards: *"my eval is not replaceable."* |
| **His observations** | Confirmed by measurement **every single time** checked. Debug the code first; measure and report the number. |

**Playtest operating point: 3200 sims** (owner, 2026-08-07: *"Let play.ipynb
play on 3200 sims from now on"*; `src/play.ipynb` sets `SIMULATIONS = 3200`).
Supersedes the 1500-sim point, which itself superseded the v17-era "800 max".

**Monitors.** *"A monitor on everything, please"* — and a log tailer is not
enough on its own. If a run is killed or hangs, the log simply stops and
silence is indistinguishable from working. Every long run gets a log monitor
**plus** a liveness watchdog polling `runs.py` state, which emits on any
transition out of `RUNNING`.

## 5. Established laws

1. **Pawn-phase cliff.** Black converts 91–100% with White's pawns gone,
   7–14% from 3–4-pawn positions the owner wins 100% of.
   **1a (2026-08-02): the live form is the post-promotion class** — Black
   holding 3+ heavies while White still has pawns. Corpus ground truth by
   White's remaining force (records / games Black-won-vs-White-won):
   bare king 48,788 / **941–83 (88% Black)**; king+1p 13,184 / 449–336;
   king+2p 11,206 / 478–617; king+3p 25,534 / **446–782 (36% Black)**. The
   model plays won positions as lost because the corpus says they are.
2. **Echo chamber (outcomes).** His Black wins double as demonstrations of
   AI-White losing. `policy_weight_for_record` masks human-game AI moves.
3. **Value saturation was the kneecap.** Fix: end-anchored ramp
   `result * γ^min(plies_to_end, horizon)`, floor 0.5 / horizon 60, scalar
   head. (Recorded saturation 10.3% vs the rebuilt corpus's 24.5% is an
   unresolved denominator discrepancy; epoch-1 reproduction says the corpus
   is right. Non-blocking.)
4. **Label shaping is invisible under the WDL head** — the ramp requires
   `--value-head scalar`.
5. **Near-mate labels poison Black when combined with the ramp.** Blending
   and fine-tuning on the existing corpus are dead ends.
6. **Measured nulls, all closed as levers:** value-head architecture
   (GAP vs spatial), human duplication (6× vs 1×), **capacity** (2.74× tower,
   arm C), **side-weighting** (`--black-weight 1.75`, arms W/CW — including
   the 2×2 with capacity: all cells within 0.8 SE), **checkpoint soup**
   (v19_KB, K/B cosine 0.932: 0.80 vs ramp but 0.35 vs K and 0.25 vs B —
   both parents beat it).
7. **Offline metrics and play strength are decoupled.** Demonstrated five
   times.
8. **Aggregates mask per-side collapse.** Always report W/B separately.
9. **The heuristic anchor is saturated for White**; only its Black leg
   informs.
10. **Style transfer through the policy head.** Pre-ps_monster, the owner's
    repertoire was the corpus's only source of opening variety and the model
    played it back at him.
11. **Data beats architecture.** Same recipe, seed, and architecture with
    only the corpus varied produced the campaign's only real gains:
    Black-vs-ramp 0.300 (v17 corpus) → 0.450 (+27 owner games) → 0.700
    (+ps_monster). The control rung reproduced the historical 0.300 exactly.
12. **Conversion is search-limited, in both classes, and it survives a strong
    opponent.** B as Black on the cliff deck converts 0.36 → 0.52 → 0.67 →
    **0.84** at 200/400/800/1600 sims (vs heuristic White@400). v19 on the
    **post-promotion** deck, measured 2026-08-03 against **v19's own White@400**
    — a strong opponent, so it passes the transfer gate — reads
    0.21 → 0.28 → 0.48 → 0.70 on `black_win_rate`.

    **Decompose that, because `black_win_rate` counts `result < 0` and so
    includes the −0.5 move-limit relabel.** True king captures are
    **0.09 → 0.15 → 0.20 → 0.30**; −0.5 dominant-but-unfinished games are
    12 / 13 / 28 / 40 of 100. Search genuinely helps — true conversions triple
    (~3.5 SE) — but **at 1600 sims 40% of games still reach the turn cap with
    Black dominant and unable to finish.** Mean length rises 55 → 112 plies:
    longer, not more decisive. The owner's shuffling report survives a 4×
    increase in search. Knowledge is present in both classes; 400 sims does not
    extract it.

    **Corollary: the corpus's 36% is an accurate record of 400-sim play, not a
    poisoned label** — which is why masking the class (L2) attacks the wrong
    thing and generating at depth (L3) attacks the right one.

    **Second corollary — finishing is a separate failure from converting, and
    neither search nor time fixes it.** Deeper search reaches the won position
    and then cannot end the game. Raising `MAX_GAME_TURNS` 150 → 400 (v19 both
    sides @800 sims, same 100 starts, 2026-08-03) left true captures at
    **0.20 → 0.20** and dominant-unfinished at 28 → 27, while mean length went
    88 → 197 plies: given 2.7× the moves, Black converts identically. The
    unfinished games are **not** conversions awaiting more time.

    So Black's residual failure is a *technique* gap. More search finds the won
    position; more moves do not cash it. The scripted oracle is the only thing
    that ever solved finishing, for the bare-king class, and it is struck for
    the rest (§5 do-not-do). The remaining sources of finishing technique are
    the owner's own games and nothing else currently identified.
13. **Self-play cannot bootstrap symmetrically.** v19 self-play converts the
    post-promotion class at 0.300 against the corpus's 0.36 — no better than
    the data it came from. Corrected labels need a Black stronger than the
    White it faces.
14. **Weak-opponent gains may not transfer.** CW posted the best cliff
    conversion of any arm (0.520, vs heuristic White) and the worst gate
    Black leg ever recorded (0.25, vs v19). Conversion measured against weak
    opposition proves nothing until it survives a strong White.
15. **Optimism is miscalibration, not insight** (bias vs own realized play:
    ramp +0.518, B +0.283, v17 +0.130, K +0.119 — K and B convert
    identically at ~0.40 while believing very differently). But the recipe
    trains value on `game_result` ramp labels, so **a generator's calibration
    never touches the labels — choose generators by play strength.**
16. **The corpus is 64/36 White by construction** — White's two moves per
    turn emit two records against Black's one. Self-play from the new arms is
    *more* White-skewed than from v17 (White overall in self-play: v17 0.562,
    ramp 0.637, B 0.700, K 0.775): both sides improved, White improved more.
17. **New data must be weighted to survive dilution.** 240 asymmetric-search
    games merged into `combined_v19_K` moved the post-promotion class's
    Black-win rate only 38.6% → 41.6% — they are 13% of the games in a class
    that already holds 1,613. Trained as-is (`v20`) the arm **failed** with a
    Black leg of 0.10 vs v19; the identical games at `value_weight` /
    `policy_weight` **4.0** (`v20w`, 28.6% of total policy weight against 10.1%
    of records) **passed**, Black 0.50. Same corpus, same seed, one variable.
    ~2.5 SE at n=20/leg, so treat the magnitude as provisional — but the
    direction says a small high-quality source is invisible at 1× against an
    established corpus. Prefer weights to duplication (D1 machinery, no split
    leakage).
18. **Search is Python-bound, not GPU-bound.** At 800 sims post clone-fix:
    move generation ~35%, board clone/apply 25–32%, tree/Python ~25%, NN
    forward **14%** (7.7% at batch 256). Leaf batch 16→256 buys 1.53×; ≤~15%
    remains in batching. The ceiling is `python-chess` move generation.

### The scripted oracle's 9/12 (diagnosed 2026-08-04)

`verify_scripted_mate --games 12` reports **9/12**, deterministically — it seeds
`random` per game, so the three failures reproduce exactly. All three fail the
same way: **Black loses its heavy pieces**, starting from K+Q+R+R vs a bare
king.

| failing start | heavies lost, at turn |
|---|---|
| `1r3qk1/1r6/8/8/8/8/3K4/8` | 5, 20, 21 |
| `q2k2r1/5r2/8/8/8/8/8/6K1` | 9, 51 |
| `1kr5/4r2q/8/8/8/8/8/K7` | 23, 80, 81 |

Not an endgame subtlety: the first loss lands at turn 5-23, during the approach,
and losses then arrive in consecutive turns as the fence collapses. The cause is
that `ScriptedMate._white_reply_min` searches depth 1 with a single danger-gated
extension — the comment records that depth 2 "exploded combinatorially" — which
is not enough to see a **double-moving king capture two squares away**, or two
heavies in one turn.

**Why this matters beyond the oracle:** `data_generation` hands Black to this
algorithm unconditionally once a position qualifies, and its moves are recorded
as training data at `policy 1.0`. A quarter of the canonical class is being
labelled with games Black loses.

**FIXED 2026-08-04** on the owner's instruction ("fix the mate bot"). Two
guards, both now on by default:

* **Material guard** — filter out any move after which White can win a heavy or
  the king, when a move exists that does not. It asks the engine for White's
  real pair list rather than reasoning about geometry, so a capture that would
  leave White's own king en prise is correctly not counted as a threat. Cheap
  in exactly this class: `mate_algo_applicable` requires a *bare* White king,
  so the pair list is ~8x8 king moves. Stands down when everything hangs —
  forced is forced, the same discipline as the king-safety override.
* **Forced-capture preflight** (depth 3), previously opt-in.

Result: **9/12 -> 11/12**, and the one remaining failure changed character
entirely. It no longer loses anything: it reaches the 150-turn cap at -0.5 with
queen and both rooks intact. That is the known *finishing* problem, not a
blunder, and no forced capture exists within depth 3 there. Separately the
oracle still converts 8/8 of the E0(b) walked-past positions.

### Do not do

- Blending, fine-tuning, or label-shape variation on the existing corpus
  (law 5).
- The policy-upweighting fix (falsified: ramp's `policy_top1_white` 0.786 vs
  v17's 0.614 — the predicted deficit does not exist).
- Side-specialized heads (rationale withdrawn; 2.64× params, 8.4M of them
  gradient-dead).
- Any automated proxy for the owner's judgement.
- Relaxing any gate threshold.
- **Extending the scripted oracle beyond the bare-king class** (owner,
  2026-08-03: *"the complexity will skyrocket"*). With White pawns on the board
  the pawns are simultaneously targets and promotion threats, so the fence /
  confinement geometry the algorithm is built on stops being fixed, and both it
  and its verifier grow without bound. `scripted_mate.py` stays as-is for the
  class it already solves.
- Quoting deck conversion rates as population rates — decks are built from
  Black-won games and played vs weak White; they compare models fairly and
  estimate nothing else.
- Pooling matches whose seeds are closer than the game count (see §8).

## 6. The v19 campaign ledger (2026-08-01/02, all evidence in `benchmarks/`)

**Phase 1 measurements:** M1 — no epoch headroom (best epoch 24 under an 80
cap). M2 — dup1's refused capture was n=1, not a population defect; the
incumbent v17 was worst through search. M3 — ramp's Black-optimism is
miscalibration (law 15). M4 — the cliff is search-limited (law 12).

**The corpus ladder** (frozen recipe: ramp r50h60, scalar, 15ch, seed 42,
30/10; gate bar was ramp, 20/side/leg + confirmation replay):

| arm | corpus adds | verdict | vs ramp pooled W / B |
|---|---|---|---|
| control | — (`combined_v17`) | FAIL | 0.550 / **0.300** |
| O | + 27 owner games | PASS | 0.775 / 0.450 |
| **K → v19** | + ps_monster, value-masked | PASS | 0.725 / **0.700** |
| B | + ps_monster, full value | PASS | **0.875** / 0.650 |

Cliff conversion (150 identical pawn-phase starts, heuristic White): v17
0.280, control 0.327, ramp 0.347, K 0.473, **B 0.527**. The K/B fork: value
labels bought game strength (B beats K 0.625, n=80) and cost calibration
(+0.283 vs +0.119) while conversion stayed identical — the owner promoted K
and could not distinguish them at the board.

**Set aside this campaign** (`models/rejected/`): `v19_C` (2.74× tower; gated,
Black 0.35 vs v19 behind a 0.55 aggregate), `v19_CW` (capacity + side-weight;
gated, Black **0.25** — law 14's proof), `v19_KB` (soup; gated by three
20-game matches, both parents beat it). **Never gated:** `v19_W`
(side-weight alone) — set aside on a conversion null (0.480 vs v19's 0.473),
which is weaker evidence than the others here; `v19_KS` was a preliminary on a
superseded data batch, not a fair test of self-play (law 13 / §7).

**D3's premise failed:** whole games from 412 cliff starts came out **9.5%**
pawn-phase by record (games leave the phase; the tail dominates) and 63/89
White wins. That batch (`data/raw/nn_v19_cliff_selfplay`) is superseded — do
not merge it.

## 7. Data

**Corpora** (`data/raw/` → `data/processed/*_r50h60`):

| corpus | what |
|---|---|
| `combined_v17` | frozen — v17/ramp/v18 arms trained on it; never mutate |
| `combined_v19_base` | v17 + the 27 owner games at 6× |
| `combined_v19_K` / `_B` | + ps_monster (243,542 positions, 195,292 train); K masks ps value (87,862 records), B does not |

**Labels:** ramp targets derive from `game_result` via `_discounted_results`
(floor 0.5 / horizon 60). `config.VALUE_TARGET_FLOOR/HORIZON` are **0.97/10 —
not** the r50h60 every current dataset uses. `mcts_values.npy` is written but
is not the training target. `plies_to_end` is stamped at generation and
preferred by the processor, so phase-filtering no longer relabels survivors;
corpora lacking the field train exactly as before. `policy_weights.npy` and
`value_weights.npy` (D1) make "teach policy, not value" expressible per
record; weight 0 is zero gradient, default all-ones is bit-identical.

**ps_monster** — 829 playstrategy.org games / 43,939 records, both players
Elo ≥ 1600, bots excluded (humans score 0.998 against them), winner-only
policy weights, `mcts_value` 0.0, White turns as two half-move records,
41.2% pawn phase. The single biggest gain of the v19 campaign. Owner
decisions embedded: filter on Elo not win rate; no side balancing. API: no
auth, ~1 req/s, `?perfType=monster`, split movetext on the header blank line.

**Owner games** — the highest per-record value of any source: 27 uncorpused
games moved the deciding leg +0.150 on their own. Harvest via
`tools/add_owner_games.py`. His notebook games are one-record-per-turn, so
**no human source but ps_monster teaches White's second move**.
`white_2026_07/game_00013` carries the hand-corrected label (§7.4, still
unsanctioned).

**The scripted oracle** — `src/scripted_mate.py` (verified by
`src/verify_scripted_mate.py`) plays the conversion once White is a **bare
king**; `data_generation` hands those endings to it. That is why the
bare-king class is 88% solved in the corpus. It abstains from king+pawns —
exactly the broken class (law 1a).

**Decks** (`data/start_fens/`): `promotion_defense_deck_v1.jsonl` (400),
`cliff_starts_v2.jsonl` (412 wP≥3 starts, 200 ps / 212 owner),
`postpromo_starts_v1.jsonl` (400 post-promotion starts: Black 3+ heavies vs
White king+pawns).

## 8. Hazards pinned by tests

- **Ramp labels are positional** — filtering records relabels survivors
  unless `plies_to_end` is present (`tests/test_ramp_label_positional.py`).
- **Match seeds closer than the game count replay the same games** and
  masquerade as confirmation (`tests/test_match_seed_separation.py`; gate leg
  pairs proven disjoint).
- **Worker defaults**: `config.DEFAULT_GAME_WORKERS = 8`. `cpu_count()`-based
  defaults (14–16) crash CUDA init and orphan 1.4 GB processes
  (`tests/test_worker_defaults.py`). Throughput plateaus at 8 anyway.
- **`clone()` copies `CLONE_HISTORY_PLIES = 8` plies** — full-stack copying
  was 82% of a late-game decision; fix verified move-for-move identical,
  4.64× on long games (`tests/test_clone_history_depth.py`).
- **Gate thresholds are constants with no CLI flag**, asserted by test;
  `tools/gate.py` cannot report PASS from a rehearsal.
- **A book position is FEN + `white_half_pending` + `turn_count`**, and a
  duplicate-game key must be the sequence of SETTLED positions, not the move
  list: White moves twice, so its two half-moves in either order transpose to
  the same position (`tests/test_opening_book.py`,
  `tests/test_finisher_engine.py`).
- **The finisher is opt-in** (`MONSTER_FINISHER`) and budget exhaustion falls
  through to the network — "no answer" is never "no win"
  (`tests/test_finisher_engine.py`). `MONSTER_SOLVER` (in-tree certainty
  propagation) was measured a **null** on the same games.
- **`match.py` and `benchmark.py` emit different result schemas**
  (`a_score/a_as_white` vs `candidate_score/white_strength`) — confusing them
  fails gates on a parsing bug.
- **A book position is FEN + `white_half_pending` + `turn_count`**, never FEN
  alone. `board.turn` stays WHITE across White's pending half, so the FEN
  cannot say which half is next; and rebuilding from a FEN restarts
  `turn_count` at 0, which would hand a deep position the full 150 turns again
  and lower its draw rate (`tests/test_opening_book.py`).
- **Under a book, independence lives in the entry index, not the seed.** Two
  legs at different seeds but the same offset replay identical openings and
  agree by construction — which would have made the gate's confirmation leg a
  tautology reporting `confirmed: true`. `gate.book_leg_offsets()` allocates
  disjoint blocks, confirmation last; the full gate needs 220 entries.
- **Policy targets are stored sparse** (`src/sparse_policy.py`, CSR in
  `policies_sparse.npz`); `open_policies` reads either format and prefers
  sparse, so a stale `policies.npy` cannot silently win. Parity with the dense
  form is asserted at **zero tolerance** — a divergence here would not crash,
  it would train on different targets (`tests/test_sparse_policy.py`).

## 9. Operational notes

**Timings (GPU box, RTX 5060 Ti):** training epoch 10–16 s; 30-epoch run
6–8 min; matches ~179 games/hour at 400 sims → a full gate ≈ 34 min;
full-length game @400 sims ≈ 121 s.

**Detached execution** (session-death-proof) — CIM `Win32_Process.Create`
with `cmd /c py -3 -u driver.py >> log 2>&1`, `ShowWindow=0`, working dir
`C:\Users\perfp\Desktop\monster-chess-nn`. Inside the driver:
ignore SIGINT, `SetThreadExecutionState`, `except BaseException` → log
`CHAIN ABORTED`. Traps that have each bitten: never `sys.exit()` inside that
`try` (SystemExit is a BaseException); never give the child `stdout=PIPE`;
rehearse the whole chain at tiny scale first.

**Watching a detached job:** `kill -0 <pid>` in Git Bash lies (MSYS PIDs) —
check liveness via PowerShell/CIM and watch for the concrete artifact (a
result JSON newer than launch), never a log grep (`--epochs 30` contains
"epoch").

**Shell quirks:** `2>$null` on native commands fakes exit 255; don't pipe
file content into `py` (encoding); **never build Python scripts in Bash
heredocs** (mangles `\n` — use the Write tool); notebook round-trip is
`json.dumps(nb, indent=1, ensure_ascii=False) + "\n"`.

**Repo conventions:** root carries `CONTEXT.md` + `DIRECTIVE.md` (+
`README.md`) only. Evidence goes to `benchmarks/` before the next run starts;
concluded work retires to git history; rejected candidates to
`models/rejected/` and the number stays free. The owner authorized the
2026-08-05 cleanup: rejected derived corpora were removed, recoverable from the
Windows Recycle Bin until it is emptied and regenerable from raw sources plus
recorded recipes. `data/processed/` since then also carries the bootstrap
generations (`bootstrap_new_main_gen_*`, `bootstrap_replay_main_gen_*`), which
**accumulate** by the owner's choice. A composed corpus was 17.3 GB until
policy targets went sparse; it is now ~3.7 GB for the same rows.
