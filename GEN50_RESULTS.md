# Gen50 epoch14 — September16 completed results

Campaign `benchmarks/gen50_20260916/production/summary.json`, completed18:08.
No promotion. Selected existing epoch14, SHA256
`51b5ddb01db51ae9023eaaf8ccbd896b48a805a52b2707633dc1d7e3f8067f25`.
All8production receipts verified;982tests+3subtests before launch.
Training stopped at23epochs. Source generation3,200games;24kpositions searched
at12,800,12kretained,60%Black. New teacher and target depth changed together,
so this is not an isolated causal test of deeper reanalysis.

## Independent normal-start results

Scores count draws ashalf. Gen50 always modelA. Sampled opening temp.5 until
primitive ply16; native defaults unchanged. No book.2,280postselection games.

| Match | Games | Overall | White | Black |
|---|---:|---:|---:|---:|
| gen49 @3200 leg1 |400|76.375%|59.75%|93%|
| gen49 @3200 confirmation |400|73.625%|53.75%|93.5%|
| gen49 @3200 combined |800|75%|56.75%|93.25%|
| gen49 @12800 |160|54.6875%|46.875%|62.5%|
| B2 @3200 |200|82%|91.5%|72.5%|
| B2 @12800 |160|99.375%|98.75%|100%|
| v27 @3200 |200|92.25%|84.5%|100%|

Gate PASS. Combined Black vsgen49:347W52D1L. Selfplay actual colors:
3200:33Whitewins97Blackwins70draw, White34%;
12800:13Whitewins57Blackwins90draw, White36.25%.
Selfplay modelA scores are NOT color skew.

Strong ordinary-budget improvement vsgen49, especially Black. Deep advantage
is smaller. White older-opponent results decline from gen49's earlier samples:
v27 96.5% ->84.5%, B2 96% ->91.5%. Different samples, not matched causal deltas.

## Read-only investigation after completion

Replayed all400 old/new v27/B2 games legally and checked recorded state/outcomes.
Against B2, gen49 won96/96 e4+d4 games andgen50 won86/86; alternative first-turn
plans grew from4/100 to14/100. Allgen50 B2 White losses/draws are in alternatives.
This is not evidence of worse continuation in the dominant B2 mainline.

After e4+d4 ...d5, gen50's c4+c5 continuation scores1W1D6L vsv27 and4W4D59L
across bothgen49 legs. The latter is67/400Whitegames vs1/80 at12,800.
After e4+d4 ...d5 e5+c4 ...f6, gen49's king routes toe3 scored36W1D0L;
gen50's Kd2,Kc3 route scored4W0D3L. Small, correlated subgroups, not solved lines.

Fresh-tree diagnostic searches (normal early-stop settings, temp0) afterc4:
gen49@3200 c5 policy2.62%,gen50@3200 35.407%,gen50@12800 17.828%.
Greedygen50@3200 still chosee5 (39.642%); greater c5 probability affects the
sampled opening test and isn't proof it always playsc5 in human temp0 games.
At12,800 greedygen50 choseKe2. Fresh-tree probes differ from in-game cachedtrees.

Gen50 ordinary teacher data at matching early branches averaged2.949%c5
policy over444 matching afterc4 records;29matching deeper reanalysis searches
all gave2.586%c5. The branch isn't simply a heavily favored teacher target.
This does not yet isolate raw prior error vsvalue error vsPUCT interactions.
Read-only probes left weights, training data and engine settings unchanged.

Next authorized work: `GEN50_RECOVERY_PLAN.md`, matched-root checkpoint
comparison and diagnostic policy/value crossover, then independent confirmation.
No automatic promotion or training. All prior evidence retained.
