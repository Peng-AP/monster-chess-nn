# Gen50 checkpoint recovery — final September17

Completed2026-09-17 01:16:31Eastern, launchedSeptember16 19:07:34 (~6h9m).
All3,568games +72root probes finished; no score-dependent truncation, code
change, restart, training, model overwrite or promotion. Epoch14 remains the
existing arena selection. Nothing further queued by this bounded campaign.

Evidence: `benchmarks/gen50_recovery_20260916/production/summary.json`.
All25production receipts, frozen runtime/code/model identity, exact nominee
hash, all288conditional replays and72probe task-file hashes verified after
completion. Normal-start journal audits ran before stage publication.
Rehearsal:983tests+3subtests,172games+24probes, native bridge parity,
all26receipts and clean resume verified.

## Screening (not independent confirmation)

80normal games per model/opponent,40percolor,3,200simulations.
Same opponent seed blocks across candidates, not guaranteed identical games.

| Candidate | Overall vsgen49 | Black vsgen49 | White vsv27 | White vsB2 |
|---|---:|---:|---:|---:|
| gen49 control |52.5%|85%|97.5%|97.5%|
| epoch10 |48.125%|67.5%|78.75%|87.5%|
| epoch13 |68.75%|66.25%|80%|86.25%|
| epoch14 control |79.375%|98.75%|76.25%|90%|
| epoch15 |76.875%|100%|83.75%|92.5%|
| epoch23 |76.875%|100%|91.25%|87.5%|

Fixed eligibility selectedepoch15. Epoch23 missed the requirement not to
reduce either older-opponent White score relative toepoch14. This screen
cannot establish superiority, and thresholds were not relaxed after results.

Nominee path `models/candidates/bootstrap_main_gen_0050/selected_epoch_015.pt`;
SHA256 `85e5d01132f3c69ccd10b381b293476ef7b9fad56034fdedc57312c849f1803c`.
No copy to arena_selected.pt; this is a separate research nomination.

## Independent confirmation

| Match | Games | Overall | White | Black |
|---|---:|---:|---:|---:|
| gen49 @3200 leg1 |400|74.875%|51.75%|98%|
| gen49 @3200 leg2 |400|73%|48.25%|97.75%|
| combinedgen49 @3200 |800|73.9375%|50%|97.875%|
| v27 @3200 |160|94.375%|88.75%|100%|
| B2 @3200 |160|78.4375%|90%|66.875%|
| gen49 @12800 |160|62.8125%|26.25%|99.375%|

Standard gatePASS. Combinedgen49 WDL:White62W276D62L;Black384W15D1L.
Deepgen49 WDL:White6W30D44L;Black79W1D0L.
Selfplay@3200 actual colors:7Whitewins68Blackwins85draw, White30.9375%.
ModelA self score47.1875% is NOT the actual White score.

Compare toepoch14's earlier independent tests, with different seeds/sample
sizes:epoch14 vsgen49@3200 W56.75/B93.25;@12800 W46.875/B62.5;
Whitevsv27 84.5%, WhitevsB2 91.5%. Epoch15's v27 White recovery is modest,
B2 did not improve, and deep White play is substantially weaker in the samples.
Not a clean balanced replacement. Do not turn nominal errors from repeated
opening trajectories into claims about perfect play or independent structures.

## Policy/value crossover findings

Root:full history e4,d4,...d5,c4; White secondhalf. Pure native MCTS, no
earlystop or finisher, deterministic matched seed/root; not ordinary game mode.
All paired evaluator outputs retain side-to-move perspective. Same-model
buffer crossover exactly matched ordinary native bridge outputs in rehearsal.

Epoch14 raw NN legal prior forc5 is2.25%; gen49's4.89%. Yet at3,200:

| Policy source | Value source | Search c5 share |
|---|---|---:|
| gen49 | gen49 |2.13%|
| epoch14 | epoch14 |35.70%|
| gen49 | epoch14 |54.52%|
| epoch14 | gen49 |4.81%|

The ordinary-budget c5 problem follows the value/search interaction much more
than the root prior. Swapping policies changes which leaves are explored, so
this is not an additive causal decomposition or a validated mixed engine.

At51,200, gen49/gen49 itself favorsc5 (67.54%), andepoch14policy/gen49value
favorsc5 (87.96%); full epoch14 instead favorsKe2 (89.56%, c5 3.55%).
Thus gen49 value is not globally correct, gen50 value is not globally wrong,
and simply increasing search or transplanting one value head is not a proven
repair. Search can encounter a different evaluation error as its horizon grows.

Matched continuations support localized failures, not a solved label:
afterc4 at3,200 vsv27, gen49won2/2;epoch14 and15lost2/2;epoch23won2/2.
At12,800 epoch14/15 eachwon1,drew1. Aftere5+c4 ...f6, epoch15won2/2vsv27
atbothbudgets, whileepoch14split1W1L at3,200. Only2games/cell, correlated
diagnostic roots; don't rank broad strength from those counts.

## Conclusion and next useful experiment (not queued)

Checkpoint substitution did not deliver an unqualified White/Black recovery.
Keep current selection and preserveepoch15 as a strongly Black-oriented option.
No evidence supports a hard-coded move ban or another architecture rewrite.

Next investigate generic VALUE calibration against completed search-backed
continuations: select positions by model/search disagreement, cover both White
halves andBlack, label with actual completed outcomes, and compare frozen-policy
value-only training to an unchanged control. Use independent positions/families
and normal-start multi-opponent tests atbothbudgets. Do not label draws as
fortresses or treat teacher root values as ground truth. A concrete training
recipe and controls need a separate authorization/plan; no hidden retraining
was started after this campaign.
