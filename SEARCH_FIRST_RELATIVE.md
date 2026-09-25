# King-relative evaluator control — September 11

Owner authorized implementation after the initial search-first prototype lost
to gen47 and B2. This is one controlled representation comparison, not a new
production engine or a broad architecture campaign. No promotion is authorized.

## Current work

Both training runs are now complete and development games are active. Full
suite:866 tests+3subtests pass. Source/split/teacher hashes match exactly between
arms. Absolute sparse-path training reproduces the original epoch7 binary hash
exactly. Relative nominates epoch18; valMSE.0321970/test.0319613 versus absolute
.0313142/.0319879. Offline errors do not establish an improvement; both play.
Native/Torch max errors <=2.99e-7. Relative inference costs1.20x the absolute
baseline on the122-state profile (8.98us vs7.47us), meeting the <=1.5x target;
sample whole-search throughput327k versus361k nodes/sec. Relative peak allocated
VRAM1.75GB. The gain, if any, must come from better decisions, not more nodes.

Update: native integration built and verified (22Rust tests,20focused Python).
Old128/512 model predictions remain exactly equal on122 saved states. New
`search_relative_campaign` is launched; do not change its pinned runtime. It
executes the staged plan below automatically. Development confirmation trigger
is relative score minus absolute score>=0.10 on16paired starts; this is only a
screening trigger, not a promotion criterion. Every nominee receives32games.
CPU baseline completed at50% over16games (White62.5%,Black37.5%); tiny sample,
separate from GPU-backed strength evidence. The earlier preparation notes below
describe work that is now integrated.

- All completed overnight match logs passed legal replay and outcome/phase audit:
  `benchmarks/search_first_20260911/replay_audit.json`.
- `search_first_cpu_control` runs16 games at300ms/half-move on book indices32..39,
  old absolute512 evaluator versus CPU-only gen47. One CPU inference thread;
  separate hardware receipt records FP32 versus GPU baseline FP16. Do not pool
  this with GPU results. Native binary remains frozen until this run finishes.
- Python relative encoder implemented; initial geometry, padding uniqueness,
  original-feature parity and missing-king tests pass.
- Native relative encoder prepared, currently unlinked. Do not rebuild the
  loaded native binary during the CPU comparison.

## Fixed experiment

Retain all840 original inputs. Add two blocks of2700 inputs: twelve piece
type/color channels, each with15x15 signed file/rank offsets from one king.
White and Black kings are separate centers. No rotation, color symmetry,
hand-written move preferences, or human-line training. Absolute inputs retain
board-edge location, phase, castling rights, raw EP and remaining turn budget.
Missing kings add no relative-center features; terminal rules remain in search.

Compare absolute840->512->32->1 against relative6240->512->32->1. Same hidden
widths, nonlinearity and raw gen47 epoch17 teacher targets; the relative model
necessarily has more first-layer parameters. The intended trade is richer sparse
inputs at similar evaluation cost, not equal parameter count. Measure actual
inference and full-search cost; target <=1.5x inference cost, report any miss.
Do not conceal a slowdown or reject a trained nominee without any play test.

Both use the original913,960-row B2 corpus, original family splits, value weights,
seed3173, AdamW lr.002/wd.0001, batch4096,30epochs, and validation-MSE nomination.
Test split used once at the end. No per-epoch play sweep. Sparse index storage on
GPU reconstructs ordinary dense minibatches; no nondeterministic sparse-gradient
training or full6240-wide corpus allocation. Train the absolute control through
the same input path to avoid confounding representation with training machinery.
Keep total VRAM below12GB. Save every small checkpoint in new directories.

## Checks and sequence

1. Finish CPU control; snapshot current runtime before integration.
2. Add backward-compatible MCSV002 loading/native relative input parity. MCSV001
   predictions must remain unchanged. Test promotions, EP, missing kings, phases,
   turn budget, random legal walks, and native/Torch numerical agreement.
3. Run full tests, train both arms, verify nominated trained-model numerical
   parity and profile inference/full search. No search/pruning changes.
4. Both nominees play32 games each against GPU-backed gen47 on shared
   development starts64..79 at300ms/half-move. These are new for this experiment,
   not globally unseen starts. Record per-color WDL, actual clocks and proofs.
5. If relative representation materially improves the development comparison,
   confirm on fresh-for-this-experiment indices128+ against gen47 and B2, with
   matched-clock common-start selfplay and known human-line diagnostics. Do not
   promote on a development result. Otherwise pause engine-replacement work and
   report whether search-leaf training is a justified next hypothesis.

The present bottleneck assessment is implementation plus evaluator quality,
not an established hardware ceiling. Initial absolute512 search averaged
W5.04/B4.62 completed turns at300ms and W6.13/B5.41 at2s. These are not Stockfish
plies: White's complete turn contains two piece moves. Our search is single-thread
and conservative, with sparse recomputation rather than incremental/quantized
NNUE and without Stockfish's extensive selective-search machinery.
