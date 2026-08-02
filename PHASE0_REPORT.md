# Phase 0 close-out — match calibration and the gate driver (2026-08-01)

Closes the last open item in DIRECTIVE §1. Two things were asked for: derive
games/hour, and encode the gate protocol in one committed driver. Both are
done. Getting there turned up a performance bug worth more than the
calibration, and three worker defaults that would each have crashed an
unattended run.

**Owner decision, taken and encoded (§6.3): the bar is `fresh_start_v18_ramp`,
and a candidate must clear it twice on different openings.**

Evidence: `benchmarks/gate_fresh_start_v18_ramp_20260801_183353.json`,
`benchmarks/gate_fresh_start_v17_20260801_190524.json`,
`benchmarks/gate_rehearsal_fresh_start_v18_ramp_20260801_180002.json`.
Commits `d6a9f65`, `def4b10`, `478350f`.

---

## 1. The answer: 179 games/hour, and a full gate costs 34 minutes

A complete 100-game gate (40 + 40 + 20) ran in **2,010 s**.

| leg | games | wall | s/game |
|---|---|---|---|
| vs `fresh_start_v17` | 40 | 922 s | 23.1 |
| vs `fresh_start_v18_ramp` | 40 | 912 s | 22.8 |
| heuristic anchor | 20 | 176 s | 8.8 |

Anchor games are ~2.6× cheaper — the heuristic engine is cheap per decision and
those games end sooner.

**Five Phase 3 arms therefore cost ~2.8 h of gating, not the ~40 h they would
have cost this morning.** The Phase 3 schedule in DIRECTIVE §8 stands.

## 2. Why it was 40 h this morning: `board.copy()`

Profiling one late-game decision at 400 sims:

```
18.6 s of 22.6 s   chess.Board.copy()   (3,030,787 copy.copy calls)
 0.63 s            NN forward           (2.8%)
```

python-chess copies the **entire move stack** by default, one `copy.copy` per
ply, and MCTS clones once per node expansion — so clone cost grew with game
length. That is why a late-game decision cost 2.80 s against an opening
decision's 0.55 s, and why the games that reach the 150-turn cap (the common
NN-vs-NN case: 3 of the first 3 games observed) were the expensive ones.

The GPU was never the lever here. NN inference is 2.8% of a late-game decision.

**Fix:** `MonsterChessGame.clone` copies `CLONE_HISTORY_PLIES = 8` plies.
Exactly one consumer reads history — `mcts._own_previous_moves`, for
oscillation detection, at offsets −1/−3/−4. Eight is that with a doubled
margin. Below 4 the oscillation override would silently stop firing, including
in the owner's own play path (`src/play.py:128` searches from a clone), so
`tests/test_clone_history_depth.py` pins the constant against those offsets by
parsing them out of `mcts.py`.

**Verified move-for-move, not result-only.** Same seeds, whole move list
compared, full history vs bounded:

| seed | plies | result | full | bounded | speedup | |
|---|---|---|---|---|---|---|
| 7 | 225 | −0.5 | 563.2 s | 121.4 s | **4.64×** | identical |
| 11 | 35 | +1 | 15.9 s | 12.1 s | 1.31× | identical |
| 23 | 50 | +1 | 27.4 s | 18.2 s | 1.50× | identical |

Short games gain least — less history to copy — which is the predicted shape.
Aggregate match throughput at 8 workers went **3.73 → 7.11 decisions/s**.

Self-play generation uses the same code path, so Phase 2's cliff decks at 1,600
sims inherit this.

## 3. Three worker defaults that were set to crash this box

14 workers dies during CUDA init with `fatal : Memory allocation failure` and
leaves 14 orphaned ~1.4 GB processes — the zombie pattern that has cost this
project runs before. The defaults in the tree were:

| file | default | on this box |
|---|---|---|
| `src/data_generation.py` | `os.cpu_count()` | **16** |
| `tools/promotion_probe.py` | `cpu_count() - 2` | **14** |
| `tools/match.py` | `cpu_count() - 2` | **14** |

M2 tonight and Phase 2's self-play generation would both have hit this
unattended. All three now share `config.DEFAULT_GAME_WORKERS = 8`, pinned by
`tests/test_worker_defaults.py`, which fails on any `cpu_count()` reaching a
default. Nothing loses its `--workers` flag.

8 is not a compromise — throughput plateaus there: 4 workers 5.39 decisions/s,
8 → 7.11, 12 → 7.38, 14 → crash.

## 4. The gate driver

`tools/gate.py` runs all three legs and emits one verdict JSON.

- **One schema.** Every leg goes through `match.run_match`, now the single
  producer of `a_score` / `a_as_white` / `a_as_black`. The §10.1 trap —
  reading `benchmark.py`'s `candidate_score` / `white_strength` shape for a
  `match.py` file and scoring `None` on every leg — is now unreachable.
- **Thresholds are constants, not flags.** `tests/test_gate_protocol.py`
  asserts no `--floor` / `--threshold` / `--min-side` / `--aggregate` string
  exists in the file. A threshold reachable from argv is a threshold that gets
  tuned when a run disappoints.
- **The dup1 case is a regression test**: 0.50 vs v17 with a 0.35 Black leg
  must fail on both counts; 0.40 exactly must pass the floor; a 0.95/0.30 leg
  must fail behind its healthy 0.625 aggregate (law 8).
- **Rehearsals cannot pass.** `--protocol quick` shrinks every leg and stamps
  `binding: false`, verdict `REHEARSAL`. A smoke test must not be able to look
  like a gate result.
- Per-side totals across all legs are reported with `se_points = √(n·0.25)`, so
  §12's "total the legs against the noise floor" is in the artifact rather than
  left to whoever reads it.

Exit status is 1 on any non-PASS verdict, so a driver chaining gates stops on
its own.

## 5. What the calibration run actually said — read this part

The calibration was a real gate on `fresh_start_v18_ramp`, the sparring partner
and second gate opponent.

| leg | aggregate | White | Black |
|---|---|---|---|
| vs `fresh_start_v17` | 0.575 | 0.60 | 0.55 |
| vs itself (degenerate) | 0.4625 | 0.60 | **0.325** |
| heuristic anchor | 0.55 | 0.90 | **0.20** |

**5.1 A 40-game leg is noisier than 40 games sounds.** Ramp's gate measured
ramp-vs-v17 at **0.575**. The incumbent's gate (§6) measured the *same
matchup* from the other chair at **0.725**. Two independent 40-game samples,
0.15 apart, nothing changed in between. The old n=20 record reads 0.70.

Pooled over all 100 games ramp beats v17 at about **0.65**, and that is the
number to carry. What the spread is really saying is that per-leg variance is
dominated by the **sampled opening set** (16 plies at temp 0.5), which both
sides of a leg share — so the two sides within one leg are correlated and the
effective sample is smaller than n=40 implies.

`tools/gate.py` already handles the part that matters: the seed is derived per
leg index, so **every candidate meets identical openings on the corresponding
leg**. Arm-vs-arm comparisons are paired and this variance largely cancels.
Absolute matchup estimates do not get that protection — do not quote a single
leg as *the* strength of a matchup, including the ones in this report.

Note what this would have done under the old habit: had ramp's gate been the
only run, "the 0.70 does not replicate, it is 0.575" was right there and
wrong. Process note §12, on the first day of using the new protocol.

**5.2 The self-match is the interesting leg.** Ramp against itself scores 0.60
as White and 0.325 as Black. A model cannot be stronger than itself, so this is
not about ramp — it is a measurement of the *game as this engine plays it*: at
400 sims, **White scores ~0.6 against Black in self-play**, in a variant whose
established law is that Black wins with correct play. That is the pawn-phase
cliff expressed as a single number, and it is the cleanest baseline the
campaign has for whether an arm has moved anything.

**5.3 Ramp as Black against the heuristic anchor is 0.20.** Law 9 says only the
anchor's Black leg informs. The sparring partner scores 0.20 there. Taken with
5.2 and with the owner's rejection of ramp for gifting pawns, the engine the
project treats as its strongest is specifically weak on the side that has to
convert.

**5.4 Ramp fails its own gate**, on `vs_ramp` Black 0.325 and anchor Black
0.20. The `vs_ramp` leg is degenerate for this candidate and should be read as
a null calibration, not a verdict. The anchor Black failure is real.

## 6. Incumbent gate — `fresh_start_v17`, and one decision for the owner

Same protocol, same driver, 1,854 s (194 games/hour — consistent with §1).

| leg | aggregate | White | Black |
|---|---|---|---|
| vs itself (degenerate) | 0.5125 | 0.575 | 0.45 |
| vs `fresh_start_v18_ramp` | **0.275** | **0.30** | **0.25** |
| heuristic anchor | 0.55 | 0.70 | **0.40** |

Verdict **FAIL**, on `vs_ramp` White 0.30 and Black 0.25.

**6.1 The self-match null checks out.** v17 against itself scores 0.5125 — a
model cannot beat itself, and the driver says so. That validates the plumbing
end to end on a case with a known answer.

Its side split is 0.575 White / 0.45 Black against ramp's 0.60 / 0.325. Both
engines favour White in their own self-play, in a variant where **Black wins
with correct play**. Ramp's split is the wider one, which is worth remembering
about the model the project calls its strongest.

**6.2 The anchor floor is satisfiable — barely.** v17's anchor Black leg is
**0.40 exactly**, clearing the inclusive floor by nothing at all. (dup1 also
read exactly 0.40 there; `tests/test_gate_protocol.py` pins that 0.40 passes.)
So the answer to §5.4's question is yes: the protocol is satisfiable as
written, and ramp's 0.20 is a genuine deficiency rather than an impossible bar.

**6.3 DECIDED (owner, 2026-08-01): the bar is ramp.** *"Every model should be
better than the last, definitively. Last should be ramp."*

Encoded in `tools/gate.py`, which now:

- makes **`vs_ramp` the decisive leg** — its aggregate must beat 0.50, and it
  is played first so a run that dies partway still has the leg that matters;
- keeps the aggregate requirement on `vs_v17` too (free: anything beating ramp
  beats v17) and the 0.40 per-side floor on every leg, unchanged;
- adds a **confirmation leg**: a candidate that clears everything replays the
  bar leg on a fresh opening seed and must clear it again. This is what
  "definitively" costs, and §5.1 is why it exists — one 40-game leg of a fixed
  matchup swung 0.575 to 0.725, so a single reading above 0.50 confirms
  nothing. Only passing candidates pay the extra 23 minutes.

`fresh_start_v18_ramp` stays in `models/rejected/` and stays unpromoted — it
holds no version number, it is only the strength bar. The original discussion
that this replaces is kept below because it records what the incumbent's
numbers were when the decision was taken.

**The old open question.** The incumbent scores
**0.275** against the sparring partner. The protocol requires every v19
candidate to reach **0.40 on each side** of that leg. That is not a small
step up from the incumbent — it is a demand that a candidate be far stronger
against ramp than v17 has ever been, and dup1 was already partly rejected on
these legs.

The owner's answer was the strict one, and stronger: ramp is not just
floor-bearing, it is *the* bar. Expect arms to die on `vs_ramp` while beating
the incumbent — that is now the intended behaviour, not a symptom.

**6.4 Both engines are weaker as Black.** Per-side totals across all legs:
v17 White 0.49 / Black 0.36; ramp White 0.66 / Black 0.39. Law 1, unmoved, now
with a per-leg baseline to measure v19 arms against.

---

## 7. State

Tests **188/188** (`py -3 -m unittest discover -s tests`, ~9 s), up 22:
`test_clone_history_depth` (7), `test_gate_protocol` (12),
`test_worker_defaults` (3).

Unchanged and deliberately untouched: `src/play.ipynb` (owner's, kernel
metadata only).

Two full gates are on record as baselines every v19 arm is measured against:
`fresh_start_v17` and `fresh_start_v18_ramp`, same protocol, same seeds, same
legs.

Phase 0 is closed. Nothing in Phase 1 is blocked: M1's epoch-headroom run needs
only training, and M2/M3/M4 all use the worker default that no longer crashes.
§6.3 does not block Phase 1 either — it only changes how Phase 3's verdicts are
read, and it can be answered any time before the arms are gated.
