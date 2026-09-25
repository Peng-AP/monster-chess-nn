# Search-backed CPU evaluator targets — September 13 overnight

**Complete06:17 Eastern.** Final report: `SEARCH_TARGETS_RESULTS.md`.208 final
real games audited; no established both-color improvement and no promotion.
All queued work has finished. The stages and recovery history below are retained.

Owner authorized plan → implement → wait, using long waits / completion milestones.
Start approximately 00:56 Eastern. Preserve all existing work and evidence. No
promotion, deletion or push. Keep the absolute840→512→32→1 architecture and the
baseline CPU playing search fixed. The official release remains v27/gen46 and
GPU gen47 remains the playing reference.

## Hypothesis and important constraint

The previous speed changes tied baseline in games. CPU8s versus CPU2s against
fixed GPU2s gained an inconclusive4.69pp, all from Black. This experiment tests
whether teaching better state values and relative move preferences helps more.

The existing GPU PUCT search does NOT track the CPU's full threefold history
inside its tree, and uses shaped turn-cap values. Reusing its Q values as exact
CPU-rule targets would be misleading. Instead, add a **label-generation-only,
complete-turn minimax tree**, with batched gen47 GPU evaluation at its frontier.
It uses all legal half-moves, max/max/min semantics, a two-completed-turn horizon,
exact capture-before-cap precedence, draw-valued caps, full settled repetition
history and the same exact frontier capture check as CPU search. No beam or
pawn-specific rules. It is a bounded search-backed teacher, NOT a claim of
perfect play or a proven stronger engine than existing gen47 PUCT.

This is tooling for training targets, not a replacement playing search. Optional
CPU leaf-lineage instrumentation must leave default decisions/nodes unchanged.

## Stages

1. Preserve the installed runtime and fixed-node snapshot. Implement the bounded
   label tree and optional lineage for actual sampled CPU NN leaves. Validate
   perspective, consecutive White moves, repetition, raw EP, cap/capture ordering,
   complete-turn boundaries, node-limit failure and default-runtime parity.
2. Run a small TRAIN/VAL pilot. Measure generation throughput and ensure useful
   nontrivial target / sibling-ranking differences before the full launch.
   Nominal corpus:1024 TRAIN and256 VAL roots from the existing family-isolated
   B2 corpus, balanced between settled White and Black, deterministically sampled.
   At each root, baseline leafepoch3 CPU searches50k nodes and samples up to2
   actual uncached, non-proven NN leaves, including full move lineage. Label the
   runtime root plus those leaves. No TEST, book, human or match roots enter data.
   Pilot may reduce to512/128 roots if the projected label job exceeds2hours;
   declare the final count before full generation. Never reduce search depth or
   accept partial trees silently. A tree exceeding250k nodes is logged/skipped.
3. For each two-turn teacher tree, retain its root and up to8 settled complete-
   turn successors. Include teacher-preferred, cheap-evaluator-preferred and
   seeded diverse alternatives. Record both gen47 raw and backed-up values on
   EXACTLY the same inputs. Derive sibling ranking targets only between states
   reached after the same completed root turn; use White-view values and explicit
   mover sign. Terminal states are exact constants, not network training rows.
   Report phase/side coverage, target disagreement and rejection counts.
4. Deduplicate by exact evaluator input (phase/rights/rawEP/budget included),
   exclude TRAIN overlaps with originalVAL/TEST and newVAL inputs, and VAL overlaps
   with originalTRAIN. Keep provenance and root-family grouping. Report conflicting
   repeated-input targets; do not pretend the840 features encode repetition history.
5. Three matched fine-tuning arms, all initialized from CPU leafepoch3:
   - raw: gen47 raw values on the new states (matched data-distribution control);
   - backed: two-turn-root / one-turn-successor search-backed value regression;
   - ranked: backed regression plus a modest sibling-ranking loss.
   All receive the same original raw-value replay anchor, same new inputs, seed,
   optimizer, batch counts and12epoch maximum. Initial schedule: AdamWlr0.0001,
   wd0.0001; half replay and half new-state regression. Ranking coefficient0.10,
   teacher margin capped0.25, ignore gaps<0.05. Batch2048 replay/2048 new rows.
   Save each epoch; choose one per arm by the same held-out composite of replay
   MSE and backed-target MSE. No TEST-based choice and no per-epoch game tests.
   Every nominated arm must pass native export parity and receive actual games.
6. Guarded chain rehearsal, then unchanged CPU plus all three trained nominees
   each receive24 games against GPUgen47 at300ms on a newly generated frozen
   random opening book. Nominate by overall score, with ties favoring unchanged.
   Then unchanged and best TRAINED nominee each receive32 fresh common games
   againstgen47 at2seconds, regardless of development outcome. This prevents
   dismissing a candidate solely on fast-clock behavior or validation MSE.
   Follow best trained with24 games against B2 at2seconds and12 common-start
   selfgames each for it andgen47 at1second. Report both colors independently.
   Full replay audit and result/continuation report; no automatic promotion.

## Operation

One heavy job at a time, <=12GiB allocated VRAM, one resident timed-match worker.
Use a reproducible new opening seed, shared starts across comparison arms, and
fresh starts for confirmation. Do not train on those books. Full source/model
hashes, successful stage receipts, progress journals and bounded stage deadlines.
Failed generation/training/parity must stop dependent matches. Existing completed
data/models are never overwritten. Inspect completion milestones, not individual
games. Expect roughly one hour of implementation/validation, up to two hours of
generation, short fine-tuning, and several hours of games; finish coherent stages
rather than changing gates to meet an arbitrary wall-clock deadline.

The teacher is imperfect; better agreement is not strength. If all arms fail,
preserve the diagnosis and do not spawn another architecture or silently retune
the gate overnight. Record concrete next steps from the completed evidence.

## Implementation / pilot checkpoint

Implemented training-only `native.LabelTree`, plus optional `collect_leaf_paths`
on CPU leaf sampling. Default fixed-node actions, values, nodes and cutoffs match
the saved prior runtime.26Rust tests and12 focused Python tests passed, including
label-tree versus fixed-turn CPU minimax, max/max/min, terminal precedence,
history draws, rawEP encoding, capped-tree rejection and exact sampled lineage.

The first pilot stopped safely on an existing Windows-path-key mismatch in raw
source provenance lookup; that was fixed without deleting its output. The second
pilot (`pilot_v2`) completed24 source roots,66 teacher trees,586 records and94,731
GPU-evaluated frontier states in12.33seconds including final hash checks. No tree
caps. Mean absolute raw-versus-backed target difference0.214;319/586records differ
by more than0.1. These are disagreements, not verified corrections.

The predeclared pilot rule therefore keeps the full1024TRAIN/256VAL source-root
size. `search_targets_rehearsal` completed: full suite → prepare pilot data →
three tiny matched training arms → fresh mini-book → all match branches → audit.
All16 rehearsal games replayed successfully. The source-identical real managed
run `search_targets_campaign` launched at01:19 Eastern, PID62476. Its outputs are
under `benchmarks/search_targets_20260913/campaign`; log is
`logs/search_targets_campaign.log`. Runtime/model/training/match dependencies
remain frozen. Await stage completions; do not use the tiny rehearsal scores as
strength evidence.

## Recovery amendment, approximately01:49 Eastern

The first real chain stopped during development_raw when Windows denied an
atomic replacement of status.json. Completed data/weights/book and all original
game files are preserved. The status-file observer may have exposed the short
reader-lock race. No playing code, targets, weights, gate or openings changed.

Added bounded PermissionError retries to atomic_json (one second maximum), with
simulated transient/permanent failures and a real Windows reader-lock test.
Added explicit --reuse-from recovery: require matching source sets; permit only
receipt-I/O/driver changes; hash-validate prepared data, training recipes,
selected native weights and original frozen book. Restart ALL game stages in a
new output folder; exclude the interrupted campaign's games from final totals.
A full recovery rehearsal is required before the real restart. Nothing deleted.
Use process-completion waits after restart, not a reader of the live status file.

`search_targets_recovered` PID60588 is queued behind the recovery rehearsal.
It requires the source-identical successful `recovery_rehearsal/summary.json`;
new real game outputs go under `campaign_recovered`. The original campaign's
corpus, prepared data, model folders and opening book are reused unchanged.

Recovery-final01:50: first recovery rehearsal exposed relative/absolute spelling
of the saved mini-book path; normalize evidence paths before lookup. No input
changed. `recovery_rehearsal_v2` passed909tests +3subtests and16replay-audited games.
The real restart is now `search_targets_recovered_v2`, PID24548, output
`campaign_recovered_v2`, using that successful source-identical receipt. Previous
recovery names are historical; all saved data/models/book are still unchanged.
