# DIRECTIVE — the v20 campaign: Black conversion (2026-08-02)

**Objective, in the owner's words: make the models good enough to beat him.**
His read after playing the v19-era models: *"White isn't doing terribly — the
play is coherent and attacking chances are taken; definitely improved. Black's
defending play is a big improvement as well. **Black conversion is a big
problem.**"*

So this campaign has one target: **Black converting won positions** — the
post-promotion class first (law 1a: corpus 36% Black in a class the owner wins
100% of), the wP≥3 cliff behind it.

**Why this class and not another.** It is not a rare ending: **1,613 of 2,624
games in `combined_v19_K` (61.5%) pass through it**, and Black wins only 38.6%
of those. Nearly two-thirds of all games are decided in a class the owner
converts every time. Caveat kept honest: *reaching* the class is not the same
as holding a won position in it, so the ceiling is not 61.5% of games flipping. White improvement is welcome fallout, not
a goal; its criterion redesign is a queued decision (§4.4), not work.

**Authorization.** The owner's standing go of 2026-08-01 carries forward:
overnight and multi-worker runs launch unattended, with the absolute
carve-outs unchanged — no workers while he is playing, promotion requires his
playtest, and no gate threshold moves. Ever.

---

## 1. What changed conceptually since the v19 plan

The v19 directive's next lever was "arm S: regenerate cliff self-play at 1600
sims and train on it." Four findings retired that plan as written:

1. **Symmetric self-play manufactures the disease** (laws 13, 1a). Games in
   which Black cannot convert record "lost" as ground truth; regenerating at
   higher sims sharpens both sides and reproduces the corpus's pessimism.
2. **Weak-opponent gains may not transfer** (law 14, the CW proof: best cliff
   conversion, worst gate Black leg). Any lever whose only evidence is
   conversion against heuristic White is unproven.
3. **Generator calibration was a red herring** (law 15). Labels derive from
   `game_result`, so the generator's value head never touches them.
   "v17 must generate because it is calibrated" is retired — generators are
   chosen for **play strength**.
4. **Ground truth outranks search.** The bare-king class is 88% solved
   because the scripted oracle finishes those games; the broken class is
   exactly where the oracle abstains. The mechanism that solved one class is
   the template for the next.

## 2. The levers, in priority order

**L1 — ~~extend the scripted oracle to the post-promotion class~~. STRUCK by
the owner, 2026-08-03: *"I don't think messing with the oracle is smart, the
complexity will skyrocket."*** The bare-king conversion is a hand-built 2-ply
minimax over a fence/confinement evaluation; generalising it to White king +
pawns means the pawns become both targets and promotion threats, so the
confinement plan stops being a fixed geometry and the algorithm — plus its
verifier — grows without bound. Kept below only as the record of what was
considered.

_Struck rationale, retained:_

**L1 (struck) —**
`src/scripted_mate.py` plays the verified conversion once White is a bare
king, and `data_generation` already hands endings to it. Extend it to Black
3+ heavies vs White king **+ pawns**, incrementally — king+1 pawn, then +2,
then +3/4 — with `verify_scripted_mate.py`-style verification at every step.
The owner converts this class 100%, so the technique exists and is
algorithmic. Payoff is triple: perfect labels at scale; self-play games that
end in conversions instead of recorded failures; a finishing module the play
path itself can use. Pure Python, no GPU, fully testable — the
highest-value engineering item in the project.

**L2 — RETIRED 2026-08-03 by the PPC curve.** Its premise was that the class's
labels are poisoned pessimism. They are not: they are an accurate record of
400-sim play, and the same model at 1600 sims converts the same positions 0.70.
There is nothing wrong with the labels to mask — the model was simply not
searching deep enough when they were made. Kept below as the reasoning that was
superseded.

_Retired rationale:_ **L2 — stop teaching the poison (cheap, and the weakest of the three).** The
king+3-pawns class holds 25,534 *records*; of the 1,228 *games* reaching it,
White won 782 (64%) in a class the owner wins 100% of. D1's `value_weights.npy` exists precisely for
"do not teach value from this." Build a corpus variant masking value on the
class (policy untouched), retrain the frozen recipe, gate. One evening, and
it isolates how much of the conversion failure is *taught* pessimism.

**Why this is ranked below L1 and L3 despite being cheapest:** masking can only
*remove* wrong teaching, never *supply* right teaching, and it leaves the value
head untrained on the class rather than correctly trained. It is also not
obvious the labels are wrong *for this model* — v19 self-play converts the
class at 0.300 against the corpus's 0.36 (law 13), so the labels describe what
it actually does. They are wrong relative to what is *achievable*, which is a
different claim and the one L1/L3 attack directly. Run L2 as a control, not as
the fix.

**L3 — asymmetric generation. The machinery already exists** (verified
2026-08-03): `--train-side black --simulations N --opponent-sims M
--use-model X --opponent-model X` builds Black at N sims and White at M, each
with its own evaluator. Nothing needed writing. Smoke: 2 games from
`postpromo_starts_v1` at Black@800 / White@200 → **Black won both**, 153
records, which is the shape L3 wants (games that end in conversions rather
than recorded failures). Generate from `postpromo_starts_v1.jsonl` + `cliff_starts_v2.jsonl`
with Black ~1600 vs White 400 — law 12 says Black converts 0.84 there — using
the **strongest** players (B/v19, not v17), truncation now label-safe via
`plies_to_end`. Secondary payoff: these are the first games in which White
faces an opponent that punishes it, i.e. White resistance data — do not mask
White's side without measuring first.

**L4 — the owner flywheel (standing process, not an event).** His 27
uncorpused games moved the deciding leg +0.150 — the largest per-record
effect on record (law 11). Harvest every play session via
`tools/add_owner_games.py`. When he wants a menu: him as Black from
`postpromo_starts_v1.jsonl` is a live demonstration of exactly the technique
L1 scripts and L3 approximates.

## 3. Measurement discipline (new this campaign, born from CW)

- **The transfer gate.** No Black-conversion gain counts until confirmed
  against a **strong White** — same positions, v19's or B's White at equal
  sims — not only heuristic White. CW would have looked like the campaign's
  best model otherwise.
- **The operating point.** Gates run at 400 sims; the owner plays at 1500.
  Before any candidate reaches him, replay its decisive comparison at 1500
  (one leg, cheap) to catch ordering flips.
- **Loop health.** In a variant where Black wins with correct play, healthy
  progress shows the **self-play White rate falling toward 0.5** — it is
  0.775 for v19, worse than v17's 0.562 (law 16). The gate cannot see this
  (it compares against a fixed opponent), so report it per candidate or the
  loop amplifies its own bias invisibly.
- Gate protocol otherwise unchanged: per-side floor 0.40 on every leg,
  aggregate over 0.50 on model legs, confirmation replay on a fresh (and
  disjoint — see hazards) opening seed, every leg reported per-side.

## 4. Open decisions — all the owner's

1. **The bar.** `tools/gate.py` currently tracks the incumbent (v19). The
   strongest engine on record is `v19_B` (0.7625 vs ramp; 0.625 head-to-head
   over v19, n=80), unrejected in `candidates/`. The owner's own rule made
   rejected-ramp the bar over v17 — "the strongest thing that exists, not
   whatever holds the number." Options: bar = B, or bar = both. Until
   decided, run reports show legs against **both** v19 and B.
2. **§7.4** — the hand-corrected label (`white_2026_07/game_00013`) is inside
   every current corpus; bless the precedent or evict the game.
3. **`data/raw/combined_v16`** — on neither this box nor the transfer drive.
   One directory copy from the CPU box reproduces ramp from source and
   settles the v16→v17 confound behind every v18-era loss.
4. **White's criterion.** Not urgent — his read is that White is coherent and
   improved — but as Black approaches correct play, White's achievable score
   falls below the 0.40 floor by the variant's nature (Black wins with
   correct play). Redefine White's leg as *resistance* (survival length; the
   ramp label already encodes it) before it starts failing candidates on the
   variant instead of on quality.

## 5. Sequence

| order | item | cost |
|---|---|---|
| ~~first~~ | **PPC sims curve — DONE 2026-08-03: 0.21 → 0.28 → 0.48 → 0.70, +0.49 at 6.9 SE against v19's White.** Search-limited, transfer-gate clean. | done |
| **now** | **L3: generate from both decks at Black@1600 / White@400, train, gate** | overnight |
| ~~now, parallel~~ | ~~per-side sims in `data_generation`~~ — already exists via `--train-side` | done |
| after L3 machinery | asymmetric generation from both decks | overnight |
| then | train the arm, gate with the transfer gate | ~1 day |
| control, any time | L2: mask the class, retrain, gate | ~1 evening |
| continuous | L4: harvest every owner session | minutes |

**Why the curve comes first.** L1/L2 and L3 rest on opposite diagnoses of the
same failure, and the experiment separating them had not been run. Law 12's
0.36 → 0.84 curve was measured on **`cliff_starts_v2`**, not on the
post-promotion deck — all four artifacts confirm it — so L3's justifying number
is imported from a different class. Worse, it was measured against *heuristic*
White, which law 14 and §3's own transfer gate say does not transfer. The curve
below is therefore run against **v19's White**, so its answer counts:

- **rises steeply** → the technique is reachable by search → L3, and L2 is
  masking labels that are not the binding problem;
- **flat** → the technique is not in the model at any depth. With L1 struck,
  the remaining routes are L4 (owner games in this class — real demonstrations,
  does not scale) and L2 as a control; and the honest conclusion may be that
  this class is not fixable by data alone at the current search depth, which is
  itself a result worth having before more compute is spent on it.

## 6. Standing discipline (unchanged, non-negotiable)

- The laws and do-not-do list in `CONTEXT.md` §5 stand. No side-specialized
  heads, no proxy scorecards, no threshold changes, no blending or
  fine-tuning on the existing corpus.
- Commits: owner identity, one-line, no co-author trailer. No push unless
  asked.
- Root carries `CONTEXT.md` + this directive (+ `README.md`) only. Every
  claim in run reports cites its JSON in `benchmarks/`; concluded work
  retires to git history.
- Detached runs per `CONTEXT.md` §9; watch the artifact, not the log grep.
- Distrust any success rate not tested against the hard case.

*Written 2026-08-02, superseding the concluded v19 directive (git history).
This is the active campaign document; it retires when v20 is decided.*
