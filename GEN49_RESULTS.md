# Gen49 results and qualification — September 15, 2026

The mainline-only new-data campaign completed September14 at15:13Eastern,
about12h40m after launch. Selected epoch7:
`models/candidates/bootstrap_main_gen_0049/arena_selected.pt`, SHA256
`4bcc68a0219acf8c3dc53326d738789e6bd767fccba88fd4f471345567b4a647`.
No promotion; public release remains v27/gen46. No production failure/restart.

## Independent normal-start results

All3,200simulations, temperature.5 for the first16search half-plies, then zero.

| Opponent | Games | Gen49 overall | White | Black |
|---|---:|---:|---:|---:|
| Gen48 first leg |400|95.25%|96.25%|94.25%|
| Gen48 confirmation |400|92.75%|93.25%|92.25%|
| Gen48 combined |800|94%|94.75%|93.25%|
| B2 |200|83.75%|96%|71.5%|
| Public v27 |200|98.25%|96.5%|100%|

Combined gen48 WDL:720wins64draws16losses. Gen48 fresh400game self-par was
White61.25%,Black38.75%; both gen49 colors exceeded their corresponding par.
The standard sampled gate passed both fixed legs. This is separately recorded
research gate evidence; canonical `iterations/gen_0049/state.json` deliberately
remains partial after checkpoint selection, not an automatically promoted run.

Gen49 selfplay:14Whitewins136Blackwins50repetitiondraws/200games. Actual White
score19.5%,Black80.5%. Arbitrary model-A role score51% is NOT the self-skew metric.
Gen49 vsB2: White96W0D4L; Black43W57D0L. Gen48's earlier B2 Black score93%
was86W14D0L, while its White score59.5% was54W11D35L.

All3,000pre/post-training independent campaign games were replay-audited;
1,160additional checkpoint-selection games are not independent confirmation.
All9production receipts and final summary hash checked on September15.
Full prelaunch suite954tests+3subtests passed. Tiny first rehearsal stopped on
default-vs-explicit opponent simulation metadata; fixed before production,
preserved separately. Second full rehearsal passed every branch, including
continuing other tests after a measured FAIL.

Evidence: `benchmarks/gen49_mainline_20260914/production/summary.json`,
`gen48_free_results.json`, `play/*.jsonl`, and the nested normal-start gate.
Recipe/implementation provenance: `GEN49_PLAN.md`, production manifest and
`iterations/gen_0049/stateful_recipe.json`.

## What the headline does and does not establish

The owner reports gen49 feels much stronger in human play. The independent
scores agree that this is substantially better against the tested policies,
not merely a favorable saved-epoch selection result. But the800gen48 games
contained31White-role and36Black-role opening endpoints. Giving endpoints
equal weight yields about74.84%, not94%; this is a secondary breadth diagnostic,
not a replacement for the owner's frequency-weighted normal-start instrument.
Repeated endpoints are legitimate random samples, not independent strategic
families; different prefix histories can also lead to the same recorded state.

The opening-branch breakdown is especially important. After White plays e4+d4
(either order), gen49 White scored369W1D3L against gen48's ...e5, but1W14D0L
against gen48's ...d5. Gen49 selfplay after ...d5 yielded0Whitewins48draws131Black
wins. All96gen49 White wins against B2 came after e4+d4 ...e5. A different
architecture is therefore not necessarily testing a different defensive idea.

The lower Black score against B2 is primarily an opening-policy shift:
gen48 chose ...e5 in53of59e4+d4 encounters and ...d5 in6; gen49 chose ...d5 in57of58
and ...e5 once. Both models drew every observed ...d5 game.56of57new draws reached
one exact opening endpoint and shared one continuation; gen48 also reached that
endpoint6times and drew, via a different72-ply continuation (gen49:74plies).
This is not evidence of57different technical conversion errors, nor proof that
...d5 is objectively drawn. A defense that scores less against B2 could still
be safer against stronger White play; the alternatives need direct counterplay.

No causal claim that variety hurts training is justified: teacher changed from
gen47 togen48, data-source composition changed, and replay rolled forward.
The next bounded investigation is `MAINLINE_COUNTERPLAY_PLAN.md`: actual ...e5/
...d5 cross-play, the genuine-history B2 draw, deeper search, then free-play checks.
