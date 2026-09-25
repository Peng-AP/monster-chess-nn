# Frozen-policy value calibration — September17

Owner authorized tonight: plan, implement, wait. No release, deletion, overwrite
of existing models/data, commit/push, architecture change, or move-specific rule.

## Hypothesis and controls

Gen50epoch14's c5 rawprior2.25% becomes35.7% at3,200search. Substitutinggen49
value reduces it to4.81%, but deeper crossover is nonmonotonic. Checkpoint
recovery found no clean balanced replacement. Test generic value calibration,
not a transplant or a hard-coded opening fix. Three arms:
1. unchangedgen50epoch14;
2. replay: same checkpoint, only existing scalar value-head parameters trained;
3. continuation: identical fit, half replay/half new completed-game outcomes.

Freeze backbone, policy and ALL non-value buffers bit-for-bit. Backbone remains
in evaluation mode. Existing GAP value-head features may be cached; assert
cached/full forward parity. Export ordinary compatible checkpoints. No changes
to search, finisher, repetition rules, policy head or architecture. Raw policy
logits must be exactly equal before/after fitting on verification inputs.

## New labels, not recycled evaluation games

192fresh normal-start epoch14selfplay games at3,200, separate per-color trees.
Existing match opening sampler:temp.5 through primitive ply16, then0; no noise.
From each family select one nonterminal root between plies4 and120. Assign
phases deterministically:96Black,48White-first,48White-second. Rank eligible
positions by |gen49 raw value - epoch14 raw value| plus |epoch14 raw value -
epoch14 recorded search value|, all in side-to-move perspective. Exclude
finisher-intercepted decisions. Do not select by specific moves or observed
win/loss. Store all scores and exact source position/provenance.

Each root receives two completed continuations at6,400:
epoch14White/gen49Black andgen49White/epoch14Black.384continuations,576new
games total. Restore full driver history; fresh trees, usual native history
limitation unchanged. Outcomes describe these players, not perfect play.
Repetition/caps are0, actual king captures +/-1. Never label a draw a fortress.

Family split:144train,24validation,24test (rehearsal8/2/2 of12parents).
Parent and both continuations stay together. Extract at most8 evenly spaced
positions per phase per game, including continuation root. Record family IDs,
half-turn, result and ending. Training labels only from completed audited games.
No old benchmark games or hand-selected regression roots used for fitting.

Existing replay:32,768train and4,096validation positions, sampled from the
originalgen50 replay split with positive value weight. Use capture_results,
not heuristic/teacher point values. Both trained arms use strict outcome labels,
so any change from the old distance-tempered target is shared by both controls.
New held-out inputs have precedence: remove duplicate encoded inputs from
training and from lower-priority validation sources; test > newval > oldval >
train. Report exclusions and conflicting outcomes; duplicate inputs within
a split receive their averaged target. This cannot make these inputs unseen
by the original pretrained checkpoint; it isolates the new fitting experiment.

## Fit and selection

Both fits initialize identically, AdamW LR1e-4,weight_decay1e-4,seed26017,
12epochs x128updates,batch512. Balanced phase sampling50%Black,25%eachWhite
half. Continuation arm mixes256replay+256new; replay arm512replay. Add0.1MSE
anchor to the original value prediction. Same update count, no policy loss.
Choose lowest mean of replay/new phase-balanced validation outcome MSE;
every arm gets games even if validation worsens. Test split evaluated only
after checkpoint choice. No per-epoch games and no offline-only rejection.

## Independent normal-start tests

Screen all3arms vsgen49,epoch14,v27,B2:80games each at3,200,960total.
Common opponent seeds across arms (not guaranteed identical trajectories).
Nominate a trained arm only if aggregate vsepoch14 >=50%, Blackvsgen49 >=
baselineBlack minus5pp, and Whitevsv27 ANDWhitevsB2 >=baseline respectively.
Rank eligible arms by mean older-opponent White score, then vsepoch14 overall,
then Blackvsgen49, then armname. Otherwise retain baseline. Selection is not
independent confirmation. No threshold changes after seeing results.

Always confirm the nominee with standard3,200epoch14gate (400calibration,
400H2H,400confirmation),160eachvsgen49/v27/B2/self at3,200,160eachvsgen49
andepoch14 at12,800. The binding comparison is the unchanged initialization,
not just an older opponent.2,160confirmation games. Total3,696full games,
excluding rehearsal and fit.
No automatic promotion, no overwriting arena_selected. Publish failure as well
as success and report every arm/budget/color, selfplay by actual color.

## Operational safeguards

Root-level opt-in scripts, no modification to previous frozen tools/tests.
Pin model, runtime, code, original replay file hashes, configs and artifacts.
Atomic per-game records and stage receipts; exact resume for completed work.
Training output directory created exclusively; interrupted fits are retained
and refused, not silently overwritten. Campaign lock and GPU worker lease.
Max8gameworkers, sequentialheavyjobs, <=12GiBVRAM target. Rehearse every stage,
freeze/perspective/leakage tests, native model loading and all test branches.
Expected8–12hours depending on game lengths and contention; no hard cutoff.
Wait on process completion with long interruptible sleeps and milestone checks.
