# September 15: resume and extend counterplay study

User authorized more testing after the Windows update interrupted production.
Keep all original code, cases, seeds, receipts and models frozen. Resume the
original v2 schedule first; do not pool the discarded original study into it.
The root-level launcher avoids changing the original tools/tests source pin.

After original completion, run the following fixed diagnostic extension:

1. 256 conditional games at 12,800 simulations per side after e4+d4 ...d5:
   both White move orders, 32 sampled continuations per order for each pairing:
   gen49 White/gen49 Black, B2 White/gen49 Black, gen49 White/gen48 Black,
   gen48 White/gen49 Black. Fresh seed block 2,260,000,000. These compare
   White defense and Black conversion with the opening held fixed.
2. 160 normal-start games gen49 vs gen48, 12,800 each, seed 2,270,000,000.
3. 160 normal-start B2 selfplay games, 12,800 each, seed 2,271,000,000.
   The latter supplies a same-search color-skew baseline for B2's defense.

Total addition: 576 full games, no new training, search changes or promotion.
These are explicitly follow-up diagnostics selected after the 3,200 results,
not untouched holdouts. Report all pairings, both colors, all budgets, and
separate conditional results from normal starts. Prefixes remain correlated;
more samples measure sampling variability, not more independent structures.

Rehearse the extension first: eight conditional games at eight simulations,
and four games for each normal match at eight simulations. Separate seed blocks
2,280,000,000 / 2,290,000,000 / 2,291,000,000. Verify summaries, legal replay,
hashes, completion and resume receipts. Original rehearsal already passed.

Run sequentially with existing eight-game/four-probe worker limits, one heavy
stage at a time; retain the <=12 GiB VRAM target. Original deeper work remains
the priority. Allow roughly 8–14 hours remaining including the extension,
subject to contention and deep-game length; not a hard deadline or a promise.
The existing original timetable was an estimate, not measured completion time.

All outputs publish through existing atomic task/journal and receipt machinery.
Resume the same launcher after another interruption; no automatic OS-startup
installation or Windows update setting change. Fail closed on identity drift
or missing evidence. Do not continue to the extension after a failed base run.
