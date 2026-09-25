# Gen50 — September 16 overnight

Authorization: owner requested plan, implement, wait for tonight. No promotion,
cleanup, destructive restart, commit or push. Preserve all earlier evidence.

## Evidence and hypothesis

The completed September15 studies comprise 1,999 games and108 root probes.
At equal12,800 search gen49 scores78.4375% vsgen48 (White98.75%, Black58.125%)
and99.375% vsB2. Gen49@12,800 scores65.625% vsitself@3,200.
The deeper gen49 selfplay is14White/36Black/110draw, White43.125%.
B2 White loses64/64 additional ...d5 continuations at12,800 after drawing
16/16 at3,200. This is not proof that B2's initial move changed: pure-search
probes after e4+d4 ...d5 choose e5,c4 at both3,200 and12,800. Later decisions,
sampling, reuse and opponent behavior remain possible contributors. Pure probes
disable early stop/finisher and are not identical to full-game searches.
No arbitrary tactical reward or rules patch is justified by these observations.

Test whether the stronger gen49 teacher and higher-search targets can transfer
more of the demonstrated search strength into fixed-budget gen50 play. This
is an iteration experiment, not an isolated causal test of target depth (teacher
also changes). No new architecture, repetition semantics or evaluator change.

## Frozen recipe

- Teacher gen49 epoch7, SHA256
  4bcc68a0219acf8c3dc53326d738789e6bd767fccba88fd4f471345567b4a647.
- 2,800 normal-start selfplay games at1,600, unchanged from gen49 formula.
- 400 parent-linked deeper continuations at12,800 (previously6,400).
- No forced openings, prefix pools or league opponents in the new increment.
- Reanalysis24,000 sampled /12,000 retained,60%Black,12,800 sims
  (previously6,400), existing coverage and family limits unchanged.
- Eight-generation rolling replay: seven accepted sources throughgen49 plus
  the new increment. Historical mixed replay is retained, not relabeled pure.
- Same scratch attention CNN, training seed3173, LR.002,batch256,EMA.999,
  warmup3,max30epochs,patience10. No offline-only rejection. Same checkpoint
  probe/screen regimen; do not play-test every epoch. Separate generation seeds.

## Chain and independent evaluation

First run existing full Python suite, focused configuration tests and a complete
isolated28-game/one-epoch tiny-search rehearsal through selection and every test
branch. Test rehearsal resume as well. Then allocate canonicalgen50 and train.

Freeze selected checkpoint before evaluation. Always run every block, including
after a measured gate failure:

1. Standard sampled normal-start gate vsgen49 at3,200:400bar selfplay games,
   400H2H plus400confirmation. Existing thresholds, no changes.
2. 200each at3,200: candidate vsB2, v27, and candidate selfplay.
3. 160each at12,800: candidate vsgen49, B2, and candidate selfplay.

Total2,280 postselection games. Report actual-color selfplay using both roles,
not model-A's arbitrary role score. Deeper tests are research, not substitutes
for the standard gate. B2 is a studied diagnostic opponent, no longer blind.
No search-budget cherry-picking or extension triggered by favorable scores.
All candidate results retained. No automatic release promotion or next training.

## Safety and runtime

Reuse existing generation, reanalysis, training, selection and audited match
machinery. New root-level launcher and recipes leave previous pinned studies
resumable. Pin code/runtime, models, recipes, plan and replay registry entries
through49; exclude newly appendedgen50 from frozen replay identity. Refuse
changed inputs and interrupted-training overwrites. Atomic evidence/receipts,
campaign lock, eight workers, sequential heavy stages, target<=12GiB VRAM.
Tiny rehearsal is plumbing evidence, not strength evidence.

Allow roughly12–18hours including deeper target generation and full checks;
actual runtime depends on game lengths/contention. Long interruptible waits,
milestone checks rather than incessant polls. Finish the declared chain even
if it runs into tomorrow. If it fails, preserve progress and repair only when
compatible with evidence identity; never silently overwrite an interrupted fit.
