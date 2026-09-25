# Shippable engine plan — proposed September 25, 2026

Status: **proposed, not started.** Owner focus after promoting v28: turn the
research engine into something other people can play, on a website and as a
download. This plan covers what to build, in what order, how each step is
verified, and which decisions belong to the owner. It does not queue any job.

## 1. Goal and definition of done

A person with no Python, CUDA or repository checkout can play v28 at a
predictable strength:

1. **Website:** open a URL and play either colour in the browser. The engine
   runs client-side, so there is no server cost, no accounts and no game data
   leaves the machine.
2. **Download:** a single Windows executable (later macOS/Linux) that plays the
   same engine faster on the local CPU, with an optional GPU.
3. **Same engine, provably:** the shipped engine uses the rules, search and
   network that the gates measured, not a lookalike. Section 5 defines that.
4. **Repeatable:** every future release (v29, …) produces its shippable
   artifacts with one command and a parity certificate.

Out of scope for now: accounts, online multiplayer between humans, and ratings
we host ourselves. An external rating comes from the optional playstrategy.org
track (Phase 2b).

## 2. What exists today (measured September 25)

| Piece | State | Consequence for shipping |
|---|---|---|
| Rules, move generation, encoding | Rust (`native/src/bitboard.rs`, `monster.rs`, `game.rs`, `encoding.rs`), with parity against the Python reference | Reusable as-is |
| Search | Rust PUCT (`mcts.rs`): factorized White half-moves, tree reuse, early stop, king-safety overrides in `best_action`, time budget (`seconds=`) | Reusable, but it is written against PyO3: the network is a Python callback (`eval_fn(bytes, n, channels)`) |
| Exact finisher | Rust solver (`solver.rs`), invoked from Python (`benchmark._build_engine` wrapper, Black-to-move with low White material) | Must move into the Rust engine loop |
| Network | `DualHeadNet`, 15 input planes, 1,909,699 parameters, 7.7 MB fp32; attention policy head over 4096 moves plus a scalar value | Small enough for the web. ONNX export is untested (no `onnx`/`onnxruntime` installed) |
| Play UI | `src/play.ipynb` widgets at 3,200 simulations, temperature 0.1 | Reference behaviour, not shippable |
| Toolchain | cargo 1.97.1 (MSVC target only), Node 24 | Needs `wasm32-unknown-unknown` + `wasm-bindgen` |

**CPU inference, v28 in PyTorch fp32 on this Ryzen 5700X:** about 360 evals/s
on 1 thread (batch 16–64) and 1,144–1,464 evals/s on 8 threads (batch 16–64);
238/377 at batch 1. That is one measurement of one runtime. ONNX Runtime,
WASM and WebGPU rates are **unknown until Phase 0 measures them**; do not
extrapolate these numbers to the browser.

The research engine plays 3,200 simulations per half-move on the GPU. White
searches twice per turn, reusing the tree for the second half.

## 3. Architecture

```
            +---------------------------+
            | monster-core (Rust crate) |  rules, encoding, PUCT, overrides,
            |  no Python, no I/O        |  finisher, time manager
            +-------------+-------------+
                          | trait Evaluator { fn eval(&mut self, planes, n) -> (values, logits) }
      +-------------------+--------------------+------------------+
      |                   |                    |                  |
 native/ (PyO3)     engine CLI           wasm module         (optional)
 research runtime   ort / tract backend  JS evaluator:       playstrategy
 PyTorch callback   text protocol        onnxruntime-web     bot adapter
 (behaviour frozen) desktop + bot        (WebGPU / WASM)
```

Key choices, each with the reason:

- **Extract a Python-free `monster-core` crate that `native/` depends on.**
  This avoids a fork: the research runtime and the shipped engine share one
  search. `native/` keeps its PyO3 API and becomes a thin wrapper that passes
  a Python-callback evaluator. Refactoring `native/` changes the `.pyd` hash
  that past campaigns pin, so the refactor must reproduce existing behaviour
  exactly (§5, check S1) before it replaces the research runtime.
- **The network ships as ONNX.** One exported file serves ONNX Runtime
  (desktop, via the `ort` crate), onnxruntime-web (browser) and, as a fallback,
  `tract` (pure Rust, compiles to WASM). Phase 0 picks between
  onnxruntime-web and tract for the browser by measurement.
- **Batched evaluator callback, the same shape as today's bridge.** Leaf
  batches of encoded planes go out and values/logits come back in side-to-move
  perspective. Keeping this contract is what lets search parity be tested
  exactly (§5, check S2).
- **One text protocol for every host.** The `monster-engine` binary speaks a
  small UCI-like protocol: `position` (FEN plus `white_half_pending`,
  turn count and recent history), `go nodes N` or `go movetime MS`, `stop`,
  and replies with `bestmove <uci>`, value and visits. A state is FEN **plus**
  pending-half flag, turn count and history (see `CONTEXT.md` §8); the
  protocol carries all of it.
- **The browser runs the engine in a Web Worker** so the UI never blocks.
  Search is cancellable, and time-budgeted moves use the existing `seconds`
  path.

## 4. Phases

Effort sizes are relative (S ≈ one session, M ≈ two or three, L ≈ a week of
sessions). They are not promises; re-estimate from actual progress. Each phase
ends with its acceptance checks passing and a short results note in
`docs/plans/`.

### Phase 0 — measurements and spikes (S–M; one GPU job)

1. **ONNX export spike.** Use a separate virtual environment (never the pinned
   research environment) with CPU torch, `onnx` and `onnxruntime`. Export v28
   at opset ≥ 17 with a dynamic batch dimension. Run check N1 (§5).
2. **Inference rates.** For batch sizes 1/8/16/32/64, measure evals/s for:
   ORT CPU native (1 and 8 threads), tract native, onnxruntime-web WASM
   (SIMD, 1 and 4 threads) and WebGPU in Chrome and Firefox, plus tract-wasm.
   Record the hardware; add a mid-range laptop if one is available.
3. **Strength per budget (the one GPU job, run when the owner is not playing).**
   On the sampled normal-start instrument, v28 at 200, 400, 800 and 1,600
   simulations plays v28 at 3,200, 200 games each (100 per colour), with
   actual-colour self-play par at each budget. This turns "the browser manages
   N sims per second" into "the browser plays at X% of the gated engine". It
   also sets the difficulty ladder (§6). Existing evidence covers only 3,200
   vs 12,800 simulations.
4. **Bot API spike (read-only, no account yet).** From the playstrategy API docs
   and bridge source, establish how a Monster turn is sent: two API moves or
   one pair, and how their event stream shows White's pending half.

**Exit:** a table of evals/s by runtime, a strength-vs-simulations curve for
v28, the chosen browser backend, and a yes/no on the bot route.

### Phase 1 — `monster-core` extraction and search parity (L)

1. Create a Cargo workspace. Move rules, encoding, arena/PUCT, overrides,
   solver and time management into `monster-core`, generic over `Evaluator`.
   Leave the experimental alpha-beta, cheap-value, label-tree and leaf-recorder
   modules in `native/` behind their existing API; they are not shipped.
2. Move the finisher's decision rule (Black to move, White material ≤
   `FINISHER_WHITE_MATERIAL_MAX`, 3 Black moves, 200k-node budget) from
   `benchmark.py` into the core, as an option that is on by default, as in play.
3. Keep `native/` as the PyO3 wrapper and rebuild it. Old `.pyd` files stay as
   rollbacks; the old binary's hash remains valid for historical resumes.
4. Pass checks S1 and S2 (§5) before the rebuilt `.pyd` becomes the research
   runtime.

**Exit:** a `cargo test` suite in `monster-core` with no Python present; the
research suite passes on the rebuilt `.pyd`; S1 and S2 recorded.

### Phase 2 — native engine binary (M)

1. `monster-engine` CLI: the protocol from §3, an ORT CPU backend (optional
   DirectML/CUDA execution providers), and `tract` as a zero-dependency
   fallback build.
2. Time management: fixed nodes, fixed movetime, and clock mode
   (remaining + increment → per-half-move budget; White's two halves split one
   turn's allocation).
3. Python harness adapter: `tools/match.py` can play the CLI as an external
   engine, so every existing instrument can measure it.
4. Pass checks N2 and G1 (§5).

**Exit:** a signed-off Windows `.exe` of about 20 MB plus the model file, and
G1 recorded.

### Phase 3 — website (L)

1. Build `monster-core` to `wasm32` with `wasm-bindgen`, plus a JS evaluator
   on the Phase 0 backend. The engine runs in a Web Worker and the model
   downloads once, then is cached.
2. **Board UI.** Double-move White turns (show the pending half and allow
   either order, since the halves transpose), king capture as the win, draws by
   repetition and the 150-turn cap, per-side clocks, flip board, move list in
   comma-pair notation (`12. Ke6,f5`), undo in casual mode, and PGN-like
   export/import. Use a permissively licensed board component (e.g.
   cm-chessboard or chessboard.js, MIT); chessground is GPL-3.0.
3. Settings: difficulty (§6), side, time control, "show engine evaluation"
   off by default.
4. Static hosting (GitHub Pages or Cloudflare Pages). Serving the model and
   WASM with COOP/COEP headers enables WASM threads, so choose a host that
   supports them.
5. Pass checks W1–W3 (§5).

**Exit:** a public URL (only after the owner approves publication, §8).

### Phase 4 — downloadable app (M)

Either the Phase 2 CLI plus the Phase 3 UI served on localhost, or a Tauri
wrapper around the same web UI calling the native engine (faster than WASM).
Recommendation: Tauri, because the UI is built once and native speed comes
free. The installer bundles the model; no Python, no CUDA required.

### Phase 5 — release pipeline integration (S)

`tools/ship_release.py vNN` does the following: export ONNX, run N1/S2/G1 at a
reduced size, write `ship_manifest.json` (checkpoint hash, ONNX hash, parity
results, engine version), and build the web and desktop bundles. Promotion
stays an owner decision; shipping is a separate, explicit step after it.

### Phase 2b — optional playstrategy.org bot (S–M, after Phase 2)

This gives real opponents and an **external rating**, the one strength signal
that doesn't depend on our own models (the owner can no longer beat them;
see memory "playtest no longer measures strength"). The bridge accepts custom
Python engines (`MinimalEngine.search(board, time_limit, …)`) or UCI
executables; the site requires at least 3+2 with increment for its own bot.
Their rules differ from ours, and the adapter must handle each difference
without crashing:

| Rule | Ours | playstrategy.org | Adapter requirement |
|---|---|---|---|
| En passant | only after White's **last** half-move | after either half-move; White ep only on first move | Accept opponent ep captures our generator omits (rebuild state from their FEN) |
| White ends turn in check | illegal (except forced blunder) | allowed if it mates | Accept and continue; our engine will not anticipate it (rare: 1 of 2,971 human games) |
| Game end | king capture | checkmate (the site also describes king capture) | Map terminal detection to the site's result stream |
| Draws | repetition, 150-turn cap | standard chess: threefold, 50-move, stalemate | Let the site adjudicate; engine keeps its own draw values |

Any bot account and any public play is an owner decision (§8).

## 5. Verification — "the shipped engine is the gated engine"

Bit-identical play with the GPU research engine is impossible, because it
evaluates in fp16/CUDA graphs. Parity is therefore checked in layers, each
isolating one source of difference:

| Check | What is compared | Pass condition (fixed before running) |
|---|---|---|
| **N1** network | ONNX (ORT CPU fp32) vs PyTorch CPU fp32, 10,000 positions sampled from v28 gate journals (both colours, both White halves) | max \|Δvalue\| ≤ 1e-4; policy argmax identical on ≥ 99.9% of positions, with every mismatch a near-tie (top-2 logit gap < 1e-3) |
| **S1** refactor | rebuilt `native` vs old `.pyd`, same PyTorch evaluator, same seeds | identical moves, visit counts and values on the existing 400-position agreement set and the 117-ply parity game; full research suite passes |
| **S2** core search | `monster-core` with a **recorded evaluator** (replays logged network outputs) vs the research engine on 200 roots | identical visit distributions, exactly. This isolates search from floating point |
| **N2** backend | native CLI (ORT) vs `native` + PyTorch CPU at 800 sims, 200 roots, temperature 0 | same move on ≥ 97% of roots; each difference inspected, and its root values agree within 0.02 |
| **G1** games | CLI at 3,200 sims vs research engine at 3,200 sims, sampled normal start | 400 games: score inside 50% ± 2 SE, and each colour inside its own self-par ± 5 pp |
| **W1** browser parity | wasm engine vs native CLI, same backend family, 100 roots | same as N2 |
| **W2** browser strength | browser default level vs research engine at the budget §6 assigns it | within the Phase 0 curve's interval for that budget |
| **W3** robustness | 200 complete browser games driven headless (Playwright): legal moves only, correct end-of-game detection, memory flat across games | zero illegal moves, zero crashes, no leak above a set threshold |

Rules parity is covered by running the existing Rust/Python rules parity
tests (`tests/test_king_capture_rules.py`, `test_ruleset_divergences.py`,
native differential tests) against `monster-core`.

## 6. Difficulty levels

The owner cannot beat v28, and neither will most visitors, so a single level
ships a frustrating product. Two independent knobs:

- **Search budget** (e.g. 50 → 3,200 simulations), mapped to strength by the
  Phase 0 curve.
- **Older releases.** v20–v27 are measured rungs, 7.7 MB each, and each is a
  genuinely different, weaker player, not just a noisier one. The round robins
  already order them (`CONTEXT.md` §2). Ship perhaps four levels, e.g. v20 @
  200, v23 @ 400, v27 @ 800, v28 @ max. Pick them from Phase 0 data, not now.

Random-move handicaps teach nothing and feel wrong; avoid them. Opening
variety should reuse the gate's rule (temperature 0.5 for the first 16
primitive plies, then 0), not the notebook's flat 0.1.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Browser too slow for a meaningful budget | Phase 0 measures before any UI work; WebGPU first, WASM-SIMD fallback; lower levels still ship |
| Attention policy head exports poorly to ONNX, or WebGPU lacks an op | N1 in Phase 0; rewrite the head with export-friendly ops if needed, and prove equivalence with N1 |
| Refactor silently changes the research engine | S1 gate; old `.pyd` rollbacks retained; research campaigns keep pinning the old binary until S1 passes |
| fp16/int8 quantization changes play | Ship fp32 first. Quantize only if the model size or speed needs it, and only after N2/G1 pass on the quantized model |
| Licensing | The Rust core does not use python-chess (GPL-3.0); keep it out of shipped code. Pick an MIT-licensed board UI. The owner chooses the project licence before publication |
| A public page is mistaken for a strength claim | UI copy says "v28, trained by self-play", with no Elo or "perfect play" claims |
| Scope creep into research | GPU use in this plan is Phase 0's single job, plus G1 in Phase 2. Research can resume in parallel, one heavy GPU job at a time |

## 8. Owner decisions needed

1. **Surface order.** Recommended: Phase 0 → 1 → 2 → 3 (website) → 4 (desktop),
   with the playstrategy bot optional after Phase 2.
2. **Publication.** Public GitHub repository or private with only the site
   public; project licence; hosting account.
3. **Rules on our own site:** ours (king capture, last-move en passant,
   repetition and 150-turn cap), which is recommended because it is what the
   engine was trained and gated on.
4. **Difficulty ladder:** how many levels, and whether older releases may ship.
5. **playstrategy.org bot:** yes/no and account name (after the Phase 0 spike).

## 9. First session, if approved

Phase 0 items 1, 2 (native rows) and 4 need no GPU and no environment changes
beyond a scratch virtual environment. Item 3 is the only heavy job: write its
fixed schedule, rehearse it at a tiny scale, then launch it with
`tools/runs.py start` at a time the owner isn't playing.
