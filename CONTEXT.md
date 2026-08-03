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

## 2. Where the project stands (2026-08-02)

- **Incumbent: `models/fresh_start_v19`** (arm K), promoted 2026-08-02 after
  automated evidence plus the owner's playtest.
- **Strongest engine on record: `models/candidates/v19_B`** — beats ramp
  0.7625 pooled to v19's 0.7125, and beats v19 head-to-head 0.625 over 80
  independent games (2.2 SE). Unrejected. Whether the gate bar should be B is
  an open owner decision (DIRECTIVE §4).
- Owner's read after playing the v19-era models: *"White isn't doing
  terribly — the play is coherent and attacking chances are taken; definitely
  improved. Black's defending play is a big improvement as well. **Black
  conversion is a big problem.**"*
- **The live problem is Black conversion** — above all the post-promotion
  class (law 1a). The campaign targeting it is `DIRECTIVE.md`.

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
| **The bar** | *"Every model should be better than the last, definitively."* The bar is the **strongest engine on record**, not whatever holds the version number, and must be cleared **twice** on independent opening seeds (two 40-game reads of one fixed matchup once came out 0.575 and 0.725 — one leg over 0.50 confirms nothing). **Unsettled right now:** `tools/gate.py` has `BAR = "vs_v19"` (the incumbent), while the strongest engine on record is `v19_B`. Owner decision open (DIRECTIVE §4.1); until then report both legs. |
| **Versions** | A number needs automated evidence **plus** his playtest. |
| **Metrics** | No proxy scorecards: *"my eval is not replaceable."* |
| **His observations** | Confirmed by measurement **every single time** checked. Debug the code first; measure and report the number. |

**Playtest operating point: 1500 sims.** He measured 2000 buys nothing over
1200 (2026-08-02); the earlier "800 max" rule was v17-era and is superseded by
that measurement.

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
12. **The cliff is search-limited.** B as Black converts 0.36 → 0.52 → 0.67 →
    **0.84** at 200/400/800/1600 sims against a fixed heuristic White@400.
    Knowledge is present; 400 sims does not extract it.
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
17. **Search is Python-bound, not GPU-bound.** At 800 sims post clone-fix:
    move generation ~35%, board clone/apply 25–32%, tree/Python ~25%, NN
    forward **14%** (7.7% at batch 256). Leaf batch 16→256 buys 1.53×; ≤~15%
    remains in batching. The ceiling is `python-chess` move generation.

### Do not do

- Blending, fine-tuning, or label-shape variation on the existing corpus
  (law 5).
- The policy-upweighting fix (falsified: ramp's `policy_top1_white` 0.786 vs
  v17's 0.614 — the predicted deficit does not exist).
- Side-specialized heads (rationale withdrawn; 2.64× params, 8.4M of them
  gradient-dead).
- Any automated proxy for the owner's judgement.
- Relaxing any gate threshold.
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
- **`match.py` and `benchmark.py` emit different result schemas**
  (`a_score/a_as_white` vs `candidate_score/white_strength`) — confusing them
  fails gates on a parsing bug.

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
`models/rejected/` and the number stays free. Data deletion is the owner's
call (`data/processed/` holds ~33 GB of regenerable datasets, most of them concluded).
