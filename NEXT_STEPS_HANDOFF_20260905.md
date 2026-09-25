# Handoff: validate gen44 and complete the bootstrap pipeline

Written 2026-09-05 following a repository review and a second evaluation of the proposed next steps.

**This is an intentionally untracked local handoff. Do not stage or commit this file unless the owner subsequently requests it.** The current request is to write the handoff; the implementation and experiments below are proposed work, not work completed during this handoff turn.

## 1. Objective and intended outcome

The owner's long-term objective is a model family that improves through repeated self-play and training, with reliable automated decisions about whether each candidate improves on its predecessor. Black strength and conversion remain particularly important, while White must remain strong.

The immediate objective is to establish whether gen44 is a reliable successor to gen42 at the owner's playing depth, then make the bootstrap pipeline capable of repeating that improvement without manual orchestration or misleading gate results.

The desired deliverables are:

1. A tested, resumable free-play gate with correct scoring, equal color weighting, explicit sample coverage, and trustworthy provenance.
2. A corrected interpretation of the existing gen44 evidence, preserving the original reports.
3. Targeted 3,200-simulation evidence for gen44 against gen42, gen41, and v24, plus gen44 self-play diagnostics.
4. A single explicit production recipe and an end-to-end iteration path that actually uses the new gate and generation-only corpus policy.
5. A controlled next bootstrap generation, followed by a clearly defined decision about continuation.
6. Updated tracked project documentation and logically separated implementation/evidence commits when implementation is authorized.

Do not spend the next campaign primarily on another full round robin, architecture experiments, tactical special cases, or repeatedly extending a marginal match until it passes.

## 2. Current repository and model state

At this review:

- Workspace: `C:\Users\perfp\Desktop\monster-chess-nn`.
- Branch: `main`, 18 commits ahead of the locally recorded `origin/main`.
- HEAD: `450002f` — `Bring the docs up to date for a fresh reader`.
- Last verified full suite: `741 passed, 172 warnings, 3 subtests passed` using `py -3 -m pytest -q`. The warnings were PyTorch pin-memory deprecations. This was run during the preceding review, not rerun for this document.
- `src/play.ipynb` is modified. The current handoff identifies its outputs as owner/session material and says never to stage it.
- The gen44 gate finished. Its report and supporting logs/cache are presently untracked.
- No model was promoted, no training was started, and no implementation was changed during the reevaluation.

| Role | Checkpoint |
|---|---|
| Numbered release, v24 = gen30 | `models/bootstrap_v24/best_value_net.pt` |
| Gen30 source, byte-identical to v24 | `models/candidates/bootstrap_main_gen_0030/screen_nominee.pt` |
| Strongest model in the completed tournament | `models/candidates/bootstrap_main_gen_0042/screen_nominee.pt` |
| New successor candidate | `models/candidates/bootstrap_main_gen_0044/best_value_net.pt` |

Checkpoint hashes verified in the preceding review:

```text
v24 / gen30:
9c0128c43670e46d0d4fffbd5fe065b930361f5b484400423e0127b2f6ecbc24

gen42 screen_nominee:
035324932307273545e2a47545899d09a2dbb5b2965bfbab04aeaf01de86e15a

gen44 best_value_net:
96b9b57f74467e5825c9ef8aa1ea3bfc89f584f21773d49d89751aca8ab753cb
```

For gen42, use the measured `screen_nominee.pt`. For gen44, the checkpoint actually tested was `best_value_net.pt`; no screen nominee exists from this run. Its training used `--select-metric decisive`, so calling this file a "lowest training loss" pick is incorrect. Its best decisive selection metric occurred at epoch 9; training stopped at epoch 19.

The numbered release, working generation champion, and strongest measured candidate are separate concepts. Verify the actual opponent path and checkpoint hash in each report. Historical legs named `vs_v24` sometimes actually played gen33 through a `--bar-model` override.

## 3. Standing owner preferences

These are recorded in the current documents and conversation:

- Free play decides strength. Books remain useful for controlled diagnostics and continuity.
- Free games must be deduplicated, with runs bounded by useful unique coverage or elapsed time.
- New training uses generation self-play data only: no human corpus, no v19 anchor, no outside game data.
- Black improvement matters greatly; aggregate improvement must not conceal White regression.
- Prefer general learning/search improvements over handcrafted responses to obvious tactical situations.
- Numbered promotion remains the owner's decision and includes a playtest.
- Long compute jobs use `tools/runs.py`, explicit worker counts, and readable logs.
- Do not run concurrent game-worker jobs on this machine. Eight workers is the established setting; larger/concurrent workloads have frozen or crashed it.
- Preserve owner changes and existing evidence. Do not destructively clean data or models as part of this plan.
- When commits are authorized, use the owner's existing identity, no `Co-Authored-By` trailer. Do not push without an applicable explicit request.
- No subagent work was requested for this task.

## 4. Evidence that motivates the plan

### 4.1 The tournament changed the interpretation of recent progress

The September 4 tournament played ten models in 45 pairings, with book and free legs for each pairing: 90 legs and 36,000 games at 3,200 simulations. The book matches used the same opening block across pairings. Free results were deduplicated on the logged opening key.

With v21 anchored at 1000, the documented fitted ladders are:

| Model | Book Elo | Free Elo |
|---|---:|---:|
| gen42 | 1353 | 1781 |
| gen41 | 1342 | 1750 |
| gen40 | 1325 | 1741 |
| gen36 | 1333 | 1713 |
| gen38 | 1343 | 1699 |
| v24 | 1325 | 1564 |
| gen33 | 1335 | 1558 |
| gen26 | 1314 | 1535 |
| v23 | 1141 | 1254 |
| v22 | 1106 | 1088 |

The useful result is the broad separation between the later candidates and v24/gen33/gen26. Book play compressed much of the improvement that free play detected. Gen36 is a concrete historical false rejection: it failed a shallow book gate but subsequently beat gen33 strongly in free play.

Do not turn the fitted ladder into a precise ordering of close candidates. Free play depends on the pairing's opening choices and has substantial non-transitivity. The top cohort's exact ordering is unresolved. Also, the phrase "top five are tied" in older prose means insufficient evidence to distinguish all of them, not proof of equivalence.

Evidence: `benchmarks/tournament/tournament.json`, its individual match reports and `.lines.jsonl` files, and commits `303f367` and `ff1b7c3`.

### 4.2 Gen44 passed the current gate, but its confirmation has limitations

Original report: `benchmarks/gate_free_gen44.json`.

| Quantity | First leg | Confirmation |
|---|---:|---:|
| Simulations per side | 1600 | 1600 |
| Games played | 600 | 400 |
| Within-leg unique games | 195 | 152 |
| Unique White games | 74 | 57 |
| Unique Black games | 121 | 95 |
| White score | 0.608108 | 0.535088 |
| Black score | 0.603306 | 0.700000 |
| Reported aggregate | 0.6051 | 0.6382 |
| Equal-color aggregate | 0.605707 | 0.617544 |

Gen42's cached self-par was measured at 1600 simulations on 173 unique games: White 0.517857 (84 games), Black 0.483146 (89 games).

The original program reported `PASS`, `confirmed: true`, no failures, and 61.8 minutes elapsed. Its requested total budget was 55 minutes, but it checks the clock between whole 200-game batches and therefore overshoots.

The reevaluation recomputed the following from the saved game logs:

- **68 of the 152 confirmation keys also occurred in the first leg.** Different seeds did not yield an entirely fresh set of scenarios. The RNG samples are separate, but the legs reuse substantial deterministic scenario evidence.
- The union contains **279 distinct keys**, not 347: 102 White and 177 Black.
- Across that union, White scored **0.573529**, Black **0.658192**, and their equally weighted mean was **0.615861**.
- On confirmation keys unseen in the first leg, White scored 0.482143 over 28 games and Black 0.776786 over 56 games. This is a small, conditionally selected subset, not a replacement population estimate. It highlights how little new White evidence the second leg added.
- No duplicate key in these saved legs had conflicting outcomes or game lengths.
- No logged game had a +/-0.5 cap label, and all opening records were marked complete.
- Ordinary scores across all 1,000 sampled games were White 0.676, Black 0.496, aggregate 0.586. These retain opening-frequency weighting; they describe a different estimand from equally weighting distinct openings. Do not apply naive independent-game confidence intervals to this pooled diagnostic without accounting for the sampling/stopping procedure.

Interpretation: gen44 remains a strong candidate. The existing result should not be described as two wholly independent, equally color-weighted confirmations, or as conclusive evidence of large improvement on both colors. Preserve the historical PASS and supplement it with corrected analysis.

The logs are under `benchmarks/gate_free_legs/`:

```text
par_gen42_b0.lines.jsonl
par_gen42_b200.lines.jsonl
par_gen42_b400.lines.jsonl
vs_bar_b0.lines.jsonl
vs_bar_b200.lines.jsonl
vs_bar_b400.lines.jsonl
vs_bar_confirm_b0.lines.jsonl
vs_bar_confirm_b200.lines.jsonl
```

The first launch, `gatefree`, ended incomplete during par measurement. `gatefree2` restarted the gate and produced the final report. Earlier commentary incorrectly called the first stopped process a completed first gate and the second an independent gate; the logs show a restart. The current shared filenames overwrote the earlier attempt's same-named batches. There is no basis to assert that two full gates completed.

### 4.3 Gen44 is not a clean anchor-removal experiment

Compare the recorded configurations:

| Setting | Gen42 | Gen44 |
|---|---|---|
| Generating/teacher model | gen33 | gen42 |
| Anchor | v19-era anchor included | none |
| Replay window | 12 | 8 |
| Free self-play games | 700 | 1000 |
| Book-seeded games | 300 | 400 |
| Reanalysis sample / retained | 16000 / 8000 | 20000 / 10000 |
| Black teacher fraction | 0.60 | 0.60 |
| Training seed | 3162 | 3173 |
| Train rows | 1,008,330 | 687,614 |

The architecture and core optimizer recipe remained the same. Several data-related variables changed together. A successful gen44 establishes that the combined recipe can work; it does not isolate which change caused the gain.

Crucial provenance correction: the replay increments called gen36 through gen42 were generated and reanalyzed using **gen33**, even though the models trained in those generations belong to the later strength tier. The name of a generation's data directory identifies a run, not necessarily the strength or identity of its generating model. Use the `incumbent` and hash in `state.json`/`accepted_data.json`.

Gen44's current replay sources are gen36, 37, 38, 39, 40, 41, 42, and 44. Gen43 is a killed run with no accepted increment. The gen44 replay has 864,822 rows: 687,614 train, 86,148 validation, 91,060 test.

Gen44 generated 1,400 games successfully, with no failed or timed-out games. Its free generation had 380 White wins, 470 Black wins, 150 draws; its book-seeded generation had 94 White wins, 272 Black wins, 34 draws. These are generation results with training exploration, not a substitute for evaluation self-par.

The new increment contains 210,340 rows after mirroring and teacher inclusion. The audit verified 10,000 one-row teachers, 20,000 mirrored teacher rows, 6,000 Black/4,000 White teachers, zero teacher value weight, and source-linked split membership. Teacher policies receive 4x weight in composition.

Gen44 trained a 1,909,699-parameter, 15-plane ResNet with attention policy, scalar value, AdamW, EMA 0.999, LR 0.002, batch 256, 30-epoch ceiling and patience 10. Moves-left, SE blocks, and side adapters were disabled. Validation loss bottoms around epoch 9, then rises modestly while training loss falls. Gen42 also showed this pattern with the anchor included. The claim that removing the anchor newly caused overfitting is unproven.

Relevant files:

- `iterations/gen_0042/state.json`
- `iterations/gen_0044/state.json`
- `iterations/gen_0044/logs/train.log`
- `iterations/gen_0044/logs/reanalyze.log`
- `iterations/accepted_data.json`
- `data/processed/bootstrap_replay_main_gen_0044/replay_manifest.json`
- `data/processed/bootstrap_new_main_gen_0044/generation_audit.json`

## 5. Step 1: repair the free-play gate and preserve evidence

Expected initial effort: approximately 1-2 hours for the focused gate work; integration adds time in step 3. Re-estimate after reading the actual call sites.

### 5.1 Scoring and coverage

1. Use `tools.match.game_score` as the canonical captures-only scorer. `gate_free.py` currently uses `(result_for_a + 1) / 2`, which wrongly gives +/-0.5 cap labels scores of 0.75/0.25. Their correct match score is 0.5. This did not affect gen44's saved games, but must be fixed before reuse.
2. Compute aggregate as `(white_mean + black_mean) / 2`. Deduplication creates unequal color counts; pooling all retained rows otherwise weights the more diverse color more heavily.
3. Record raw counts, unique counts, W/D/L, score, and uncertainty separately per color. Record raw sampled scores as a secondary diagnostic, clearly distinguished from the unique-opening metric.
4. Set per-color coverage targets. A total of 200 unique games could contain too few White games to answer the owner's question.
5. Measure duplicate overlap both within and across legs. Preserve the full independently sampled confirmation result, and explicitly identify the additional previously unseen evidence. For the planned novel confirmation target, continue collecting until the specified number of unseen keys per color is reached or the budget expires.
6. Explain that rejecting previously seen keys changes the distribution of the novelty subset. Do not silently present its score as the natural sampled win rate. The gate's unique-opening coverage metric must be named and versioned.
7. Validate the identity used for deduplication. The current key is color + FEN + pending-half flag + turn count. The game also carries repetition history and search can reuse trees. Identical logged endpoint keys produced identical results/lengths in gen44's logs, but that observation is not a general proof of equivalence. Inspect these dependencies; record enough provenance/history to detect false collisions without changing gameplay merely to simplify logging.
8. Define behavior for games terminating before the sampled prefix completes. They are valid game evidence and must not disappear or all collapse into a missing-opening key.

### 5.2 Verdict semantics

Retain the current policy values: aggregate strictly above 0.50 on the required legs, each side no worse than the bar's measured same-side self-par minus 0.05. Do not tune those thresholds based on gen44's results.

Add a distinction between:

- **PASS:** sufficient required coverage and all fixed checks passed.
- **FAIL:** the completed, adequately covered evaluation fails the fixed checks.
- **INCONCLUSIVE:** missing color coverage, unfinished confirmation, unusable provenance, or insufficient evidence under the agreed precision requirements. This must not advance the champion or be reported as proof the candidate is weaker.

Use uncertainty appropriate to draws, unequal per-color counts, and scenario overlap. The existing Bernoulli expression is only an approximation. Include uncertainty in the measured par when discussing calibrated side changes. Do not treat "not significantly worse" as proof of non-inferiority or convert a broad interval into a definitive strength claim.

Before new matches, write the evaluation version, coverage/budget limits, and decision rules into a manifest. Any extra statistical criterion should be fixed before outcomes are inspected and should not relax the existing floors. Fixed-budget sampling with an honest inconclusive state is sufficient; no elaborate sequential-testing framework is required for this first repair.

### 5.3 Recovery and provenance

- Give each invocation a unique run directory; filenames must not collide across candidates, retries, or concurrent launches.
- Write per-game records incrementally as games finish. The current `match.py --game-log` writes only after the entire batch finishes; aggregate checkpointing alone cannot recover exact deduplication evidence.
- Resume only after validating checkpoint hashes, engine/rules/search configuration, seed schedule, scoring version, and existing records.
- Key the par cache by checkpoint content hash and relevant configuration, not just `bar_name@sims`. Store per-color counts and enough statistics to assess precision and extend coverage.
- Store explicit checkpoint hashes, actual paths, seed ranges, opening-temperature settings, engine/rule flags, batch counts, output paths, and completion status in reports.
- Prevent overlapping worker campaigns. The existing iteration lock and run-launch conventions may provide reusable pieces.
- Reduce or adapt batch size near the deadline. Preserve finished games on interruption; do not kill a healthy in-flight game simply to hit a soft minute estimate.
- Allow positive argument validation and fail clearly on malformed records or missing colors.

### 5.4 Verification and evidence preservation

Focused tests should exercise failure-prone contracts rather than mirror source text: captures-only cap scoring, equal-color means under uneven deduplication, overlapping confirmation sets, incomplete openings, invalid cache reuse, partial-write/resume, and underpowered verdicts.

Use CPU fixtures/mocked game results for most tests. Follow with one tiny real end-to-end rehearsal. The last 741-test pass does not establish correctness of `gate_free.py`; the reevaluation found no dedicated tests for it.

Preserve `gate_free_gen44.json` and the original logs. Write a separate versioned rescoring report with the recalculated counts and scores. Do not silently replace the old PASS with revised numbers under the same protocol label.

Done when a stopped/restarted rehearsal retains finished evidence, cannot double-count it as fresh confirmation, cannot pass with insufficient color coverage, and reproduces the gen44 recomputation above.

## 6. Step 2: validate gen44 at 3,200 simulations

Run worker workloads sequentially with eight workers. Pin the actual gen44 checkpoint tested at epoch 9 before collecting new evidence.

| Comparison | Initial target | Purpose |
|---|---|---|
| gen44 vs gen42, first leg | 100 unique games per color | Direct improvement at playing depth |
| gen44 vs gen42, confirmation | 100 previously unseen keys per color, retain full sampled record too | Additional evidence beyond repeated opening scenarios |
| gen44 vs v24 | 100 unique games per color | Direct comparison with the numbered release |
| gen44 vs gen41 | 100 unique games per color | Check another strong opponent for matchup-specific weakness |
| gen44 vs itself | Target comparable per-color coverage; fix budget before launch | Self-play color skew and ending diagnostics |

Measure or reuse a valid gen42 self-par at **3,200** simulations. The existing 1,600-simulation cache cannot calibrate this test. Comparisons to other opponents need an appropriate baseline before interpreting absolute per-color scores as gains; they can still report raw per-color scores without that interpretation.

Suggested measurement allocation is 3-6 hours initially, refined from the first real batches. The unique target is aspirational within the fixed budget, not a promise. Novel opening yield will fall as seen sets grow. Stop with an honest incomplete/inconclusive result if coverage cannot be reached; do not repeatedly extend based on whether the score is favorable.

For self-play report actual White wins, Black wins, draws, repetition endings, cap endings, game lengths, and opening diversity. Add a terminal-reason field if the current logs do not distinguish repetition from cap draws. Self-play balance is diagnostic: do not optimize toward a 50/50 color split as a proxy for strength.

Save enough trajectory information to show representative Black wins, draws, and losses on a board. Choose examples by documented sampling/selection rules; illustrative games are not population estimates. Avoid launching a separate large showcase campaign merely to obtain attractive games.

### Decision after testing

- If the deeper gen42 comparison supports an advantage, the fixed color safeguards pass, and the v24/gen41 checks reveal no material regression, gen44 may become the working generation champion.
- Its numbered release promotion remains the owner's call after playtesting. Do not automatically create v25.
- If gen44's advantage disappears at 3,200, hold gen42 as champion and study the depth/side pattern before generating a long chain from gen44.
- If the result is uncertain, retain the candidate and specify the unresolved comparison. Do not call a budget-limited result a model failure.
- Do not derive a gen44 tournament rating by adding its head-to-head Elo to gen42's ladder rating. Direct matchup evidence is the relevant claim.

## 7. Step 3: integrate the policy into the actual iteration pipeline

This is necessary before another unattended multi-generation campaign. Current code contradicts the documentation:

- `src/iterate.py` constructs its binding command using **`tools/gate.py`**, not `tools/gate_free.py`.
- `--anchor-data` still defaults to `combined_v19_B_r50h60_capture`.
- `--replay-generations` still defaults to **4**.
- Recent generations were run `--through-phase train`, with later screening/gating orchestrated separately. Their `state.json` status is commonly `partial` even when separate benchmark reports exist.
- Gen44's external gate report is not automatically reflected as a completed pipeline gate or champion advancement.

Implementation work:

1. Create one explicit production recipe/configuration for the generation-only policy and the workload demonstrated by gen44. Avoid a collection of undocumented command-line overrides.
2. Make anchor exclusion and the intended eight-generation replay policy the normal production behavior. Historical replay reproduction can retain an explicit opt-in anchor path; the default must not silently violate current policy.
3. Wire the repaired free gate into `binding_gate`, report parsing, recovery, and champion eligibility. Support historical book evaluation explicitly for diagnostics/compatibility.
4. Audit checkpoint screening, the separate high-fidelity stage, and self-skew. Do not leave a shallow legacy book stage capable of rejecting the winner before the new free gate, or force redundant confirmation stages without a distinct purpose.
5. Keep offline validation advisory for playing strength. It can nominate/check checkpoint health, but a small loss regression must not silently skip all play testing.
6. Advance the champion pointer only after complete, compatible gate artifacts. Keep numbered release promotion separate.
7. Continue retaining audited accepted data if the trained candidate fails. Validate dataset hashes and source-linked split membership on resume.
8. Track the generating/teacher checkpoint explicitly for each increment. Do not infer quality from generation number alone.
9. Record abandoned gen43 as abandoned in operational summaries, preserving its files. It contains state/lock metadata and should not be described literally as an empty directory. There is no need to recycle the generation number or delete it.

Verify with a tiny end-to-end run that exercises generation, processing, replay, training, gate, resume, and pointer eligibility using temporary output locations. Do not advance the real champion during a rehearsal. Add contract coverage for the transitions most likely to lose an overnight run.

Step 1 plus this integration is estimated at roughly 2-4 hours initially. Unexpected state-machine/schema issues may extend it; report those rather than cutting recovery checks.

## 8. Step 4: test whether the gain compounds

If gen44 passes step 2, use it to generate **gen45** with the same established settings as gen44:

```text
generator / reanalysis teacher: validated gen44 checkpoint
anchor: none
replay window: 8 accepted generation increments including current
free self-play games: 1000
book-seeded games: 400
generation simulations: 700
reanalysis sample: 20000
reanalysis retain: 10000
reanalysis simulations: 3200
Black teacher fraction: 0.60
teacher policy multiplier: 4.0
replay balancing alpha: 0.5
value floor / horizon: 0.5 / 60
architecture: same 15-plane attention-policy ResNet, scalar value
optimizer: same AdamW / LR 0.002 / EMA 0.999 / batch 256 recipe
epoch ceiling / patience: 30 / 10
workers: 8
training: fresh initialization with a recorded new seed
```

Keep the book-seeded generation input explicit and preserve its provenance. Free-play evaluation does not imply that all training games must begin at the initial position. Do not remove the 400 book-seeded games as an additional unrecorded experiment.

Keep 30 epochs / 10 patience for this first compounding test. Gen44 already early-stopped at 19; there is no current causal evidence that altering this is the next strength improvement. Do not reintroduce moves-left, SE, side adapters, or other architectural changes during the compounding test.

Checkpoint selection should use a small, predetermined shortlist of plausible saved checkpoints, followed by actual play. Fix the shortlist rule before viewing arena results, for example a bounded set of distinct checkpoints representing the offline selection optimum and nearby training stages. Verify the rule against the existing screen implementation before choosing exact epochs. Do not expend a full high-power gate on every epoch, and do not select an epoch using the confirmation games that will later certify it.

### Continuation rule

- If gen45 improves under the repaired protocol, advance the working champion and continue the same recipe.
- If it fails, keep gen44 as champion and retain gen45's accepted data. Run one further generation from the unchanged champion with that added replay data.
- If that also fails, pause recipe repetition and move to the diagnostic branch below. This prevents an endless loop of identical attempts while allowing one data-accumulation step.
- A failed candidate's data is not automatically poor data: it may have been generated by the unchanged strong champion and passed all ingestion checks.
- Report generation cost, teacher yield, training duration, shortlist cost, gate cost, unique games per color, and the actual generating model. These determine whether the loop is sustainable.

Allow roughly 3-5 hours for the first complete successor cycle as an initial estimate; refine it from the repaired evaluation throughput. Earlier gen44 generation through training took about 2.2 hours, before the new screening/confirmation requirements.

## 9. Step 5: experiments only if the fixed recipe stalls

First retrain on the **same frozen corpus** with another training seed. Freeze the architecture, optimizer, split membership, shortlist rule, and evaluation schedule. This distinguishes training variability from insufficient data improvement. A seed repeat must not become an unlimited search for a lucky pass.

If seed variation does not explain the stall, choose one variable based on the evidence:

- **Generation search depth:** compare the existing 700-simulation generation with a higher-depth arm, using a declared compute/game budget and the same evaluation conditions. Distinguish better labels from merely fewer examples due to extra compute.
- **Replay volume or freshness:** compare windows or sample volume while preserving the generation-only policy and tracking the actual generating teachers. Do not equate an old run number with old-model data without checking provenance.

Fix the primary comparison, budget, and acceptance criterion before running the arm. Do not switch to a flattering secondary statistic afterward.

Do not restore the v19/human anchor as the default overfitting response. The owner explicitly removed it, gen42 also overfit with the anchor, and gen44 changed several other factors. A later anchor experiment would require the owner to revise that data policy.

Do not infer that teacher split, corpus doubling, or an architecture addition caused the later tier jump merely because one successful model used it. Existing teacher-split and doubling studies had mixed or negative preregistered results. Preserve that distinction between a useful resulting checkpoint and a proven recipe intervention.

## 10. Documentation and commit deliverables

When the implementation campaign proceeds, update these tracked documents in concert with evidence:

- `HANDOFF.md`: gen44 gate completed, evidence caveats, actual next action, and current champion/release distinction.
- `CONTEXT.md`: reconcile the entry-point description, remove stale active instructions, add the scoring/overlap and data-provenance lessons.
- `README.md`: correct v23/gen23 current-state passages, old anchor/default commands, test count, and the real implemented pipeline.
- `REPORT.md`: either add the new evidence with explicit dates or clearly direct readers to the later campaign reports; it currently stops at August 18.
- `benchmarks/INDEX.md`: index the corrected gate/evaluation artifacts and retain the distinction between flat generated indexing and campaign directories.
- `data/processed/README.md`: correct the assumption that gen36+ data was generated by the stronger tier and the causal overfitting claim.
- `models/candidates/README.md`: distinguish the actual gen44 tested checkpoint from an arena nominee, and correct the generic "lowest training loss" description.

Suggested commit boundaries, subject to the implementation actually produced:

1. Free gate correctness and focused tests.
2. Match logging/recovery and provenance support, if substantial enough to stand alone.
3. Iteration integration and production recipe, with state-machine tests.
4. Preserved gen44 evidence, corrected analysis, and targeted evaluation results.
5. Current documentation and subsequent generation evidence.

Commit only complete, relevant artifacts. Preserve original benchmark reports and identify rescoring versions. Never include the owner's notebook changes by accident. **Leave this handoff file untracked.**

## 11. First actions for the next session

1. Read this handoff, then check `git status`, latest commits, active runs, and checkpoint hashes. External work may have advanced since this snapshot.
2. Read `tools/gate_free.py`, relevant parts of `tools/match.py` and `src/benchmark.py`, and the gate/state transitions in `src/iterate.py`.
3. Reproduce the inexpensive gen44 log census: 195/152 within-leg unique keys, 68 overlap, 279 union keys, and equal-color score 0.615861 for that union.
4. Preserve current untracked evidence before another run can overwrite shared batch names.
5. Implement step 1, verify it, then begin the targeted validation. Integrate step 3 before launching the next complete bootstrap generation.
6. Maintain a short status record containing the current phase, exact commands, output paths, last completed evidence, and the decision that follows. A watcher or detached script is not proof an assistant is actively reviewing its output.

The stopping condition for this plan is a defensible gen44 promotion/hold decision plus a working iteration path that can train, evaluate, preserve data, and advance a champion under that same measurement standard. Producing a new numbered model is conditional on the evidence and the owner's playtest, not guaranteed by spending a fixed number of hours.
