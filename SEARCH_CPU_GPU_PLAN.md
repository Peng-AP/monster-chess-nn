# CPU search and GPU cooperation — September12 overnight

Owner authorized roughly eight hours from02:26 Eastern on September12. Implement
and test this plan; the duration is a guide. Use sequential managed campaigns,
receipt checks, and milestone updates. Preserve existing models and measurements.

## Implementation checkpoint —02:43 Eastern

Stages1–3 implemented.26Rust tests pass, including PVS against exhaustive search;
full Python suite886tests+3subtests passes. Default fixed-node results reproduce
the prior runtime. All four tiny complete-game rehearsals and their replay audit
pass. Active managed job: `search_cpu_gpu_campaign`; it has entered `cpu_vs_cpu`.

Final profile: `benchmarks/search_cpu_gpu_20260912/workload_final.json`.
Three interleaved repetitions on23states, fixed target depths3..8: baseline
aggregate median time7.8665s; PVS7.3807s; freshTT7.3498s; combined6.8190s;
combined131,072-entryTT6.6346s. Nominate the last configuration,1.186x baseline
throughput at equal depth (15.7% less elapsed time). Exact values match.
The nine timed probes averaged identical completed depths at300ms and2s; a
strength/depth improvement has not yet been established. GPU root-ordering
fixed-depth parity also passes. The small White census found201continuations
with130unique final states across4states; this is too small to generalize and
does not measure how much the existing TT already avoids.

Runtime hash `efb01f58461dee6b0c532b80d3042f38a940a1277b76589295bf8767bfa33fb8`.
Reusable adapter: src/cpu_search_engine.py. Match and campaign drivers:
tools/search_cpu_gpu_match.py and tools/search_cpu_gpu_campaign.py. The full
campaign's pinned code/runtime must remain unchanged until completion. No
retraining, deletion or promotion performed.

## Evidence and objective

The absolute512 leaf-trained evaluator, epoch3, scored43.75% against GPU gen47
at2seconds per half-move over64games; the original evaluator scored32.81% on the
same32 starts. Leaf scores were32.81%White/54.69%Black. Against CPU-only gen47,
it scored71.875% over64games at300ms (65.625%White/78.125%Black). Both CPU engines
used one search thread. These results support continuing CPU-engine work; they
do not establish superiority over the GPU reference or explain the color gap.

Primary objective: reduce time to the same completed-turn depth, then determine
whether the resulting extra search improves actual games. Secondary objective:
test a small, measurable cooperation path between the existing GPU and CPU engines.
Fixed models throughout; no new training/architecture family during this run.

## Stage1 — exact CPU search improvements (30–60min)

1. Snapshot installed native runtime and relevant sources. Record baseline
   fixed-node results on representative saved states before edits.
2. Integrate optional principal-variation search (PVS): full window for first
   child, minimal floating-point scout window for later children, full-depth
   re-search for any in-window improvement. White is max/max, Black min; never
   assume alternating negamax. Count scout searches and re-searches.
3. Add optional clearing of searched bounds between iterative-deepening passes.
   Current bound keys include exact depth, completed-turn count and settled path;
   earlier-depth entries consume the32k table despite their restricted reuse.
   Retain move-ordering and static-evaluation caches. Measure the effect rather
   than assuming clearing or a larger table is better.
4. Expose bounded TT capacity (default unchanged), saturation telemetry, and
   per-phase expansion/generated-move counts. Track White-turn-equivalent final
   states in a separate offline diagnostic, including rawEP, rights, phase and
   turn count. Existing search already reuses some equivalent pair continuations
   via its bound cache; a second pair enumerator may add overhead.
5. Fixed-depth parity against ordinary alpha-beta and small exhaustive positions;
   promotion, EP, White pending phase, repetition, terminal/cap, and interrupted
   iteration coverage. Require equal values; tied optimal actions may differ.

## Stage2 — bounded GPU cooperation (30–60min)

Implement a reusable engine adapter and explicit match options:

- **GPU ordering:** one root policy inference orders legal CPU root moves. Retain
  all legal moves and exact CPU minimax values. Previous-iteration best move should
  retain precedence so GPU guidance does not defeat iterative deepening. Charge
  inference, encoding and sorting to the same total per-halfmove clock. If the
  clock is exhausted before search, return a legal fallback and mark depth0.
- **Color routing:** existing GPU PUCT plays White; CPU alpha-beta plays Black.
  This tests whether their observed strengths are complementary. It is an explicit
  experimental mode with per-decision backend telemetry, not a claimed universal
  architecture improvement. Give each move the same configured clock.

Both approaches reuse gen47 as the GPU model. A GPU call at every depth-first CPU
leaf is deferred: its latency and batching requirements would demand a different
search scheduler. Also defer quantization, incremental accumulators, multi-thread
search, null moves and depth reductions until the simpler controlled changes are
measured. These remain future options, not promised overnight deliverables.

## Stage3 — correctness and performance nomination (20–40min)

Profile baseline, PVS, refreshed TT, and their combination on saved positions in
all phases, at identical fixed depths and timed300ms/2s budgets. Interleave arm
order; report total time-to-depth, node counts, phase counts, TT saturation/hits,
and re-search overhead. No selection by nodes/second alone. Verify GPU ordering
preserves fixed-depth value despite different move order. Run focused Rust/Python
checks and full suite once after integration; repeat only for subsequent changes.

Nominate one CPU configuration by aggregate fixed-depth elapsed time, with a
per-position regression report. If all changes lose, retain baseline and report
that result. GPU ordering/routing still get their own actual games. Warm GPU
startup outside measured moves; include all recurring overhead inside the clock.

## Stage4 — play screen and confirmation (roughly4–6h)

One resident match worker; one heavy job at a time; up to12GB VRAM. Models fixed:
leaf epoch3 for CPU, gen47epoch17 for GPU, B2epoch8 as independent opponent.

1. CPU nominee versus unchanged CPU search,32games at300ms, paired openings.
2. CPU baseline/nominee, GPU-guided CPU, and color routing versus GPU gen47,
   32games per distinct arm on common development starts. Skip redundant baseline
   reruns when the nominated configuration is unchanged.
3. Freeze the best development arm, then test on disjoint fresh paired starts at
   2seconds:64games versus gen47. Include a matched CPU baseline control when the
   chosen arm changes CPU search or guidance; report both colors and paired delta.
4.32games versus B2 and16common-start selfgames per selected engine/gen47, budget
   permitting. CPU-only comparison is relevant for a CPU nominee and must remain
   separate from GPU-backed scores. Do not call a GPU-assisted mode CPU-only.
5. Replay every completed game, verify legal moves/phase/outcome, classify captures,
   repetition and turn-cap draws, check proof consistency, and write final receipt.

Development starts288..303; confirmation304..335 from the existing400-start book,
new to this search experiment. Book is reused project evidence, not newly generated
random openings. Common-start selfplay measures skew; it is not an independent
strength test. Human diagnostic positions remain diagnostic and never training.

Each stage records code/runtime/model hashes and exits on invalid evidence. A
failed stage prevents dependent stages from running. Estimates depend on game
length and machine contention; avoid automatically repeating weak arms just to
fill time. No automatic release promotion. v27 remains the official release.

## Interpretation and next decisions

Accept a speed result only with fixed-depth parity. Accept a strength claim only
with actual play evidence and uncertainty; fewer nodes or a deeper reported line
alone is insufficient. Depth counts complete player turns: White's pair counts1,
Black's move counts1. Score50% means parity against the stated opponent/setup.

If efficient CPU search improves both colors, keep that configuration and consider
incremental evaluation next. If GPU guidance helps, expand batched ordering only
near the root. If routing helps, retain the explicit backend choice while studying
which positions benefit, rather than inventing tactical rules from a few games.
If neither helps, inspect move ordering, evaluation instability and branching
telemetry before another architecture or training change.

Background references consulted: Stockfish's current search uses scout/full-window
search and iterative deepening; LC0 combines policy/value networks with tree search.
Their conventional-chess pruning constants are not transferred to Monster Chess.
[Stockfish search source](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)
and [LC0 overview](https://lczero.org/dev/overview/).
