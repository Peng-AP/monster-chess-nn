# Monster Chess NN — handoff (current state, October 6, 2026)

This is the entry point for the next developer or agent. It describes what is
true now. The dated history is in `docs/history/`:

- `HANDOFF_20260925_20261005.md`: the previous handoff, with every update from
  September 25 to October 5;
- `HANDOFF_LOG.md`: the operational log through September 25;
- `CONTEXT_LOG.md`: the old status sections of `CONTEXT.md`.

`docs/README.md` indexes plans, protocols and results.

## 1. State

- **Release: v29** (`models/bootstrap_v29/best_value_net.pt`, gen51
  deep-value, promoted September 29). The site defaults to it, and
  `models/bootstrap/champion.json` and `tools/gate.py`'s `BAR` name it.
- **Strongest model: gen53** (`models/candidates/bootstrap_main_gen_0053/arena_selected.pt`).
  - +105 Elo vs v29 in the top-group round robin, and it beats every other
    model head-to-head.
  - It is the first candidate to pass gate v4 against v29.
  - **It is not eligible for promotion.** Its held-out mean is 88.3%,
    against the 91.7% the rule requires, because of one White endgame hole
    that only v27 exploits (32 of 32 losses come from a single position).
  - It is on the site as "experimental".
  - See `docs/experiments/gen53/`.
- **Running: gen54**. This is one arm:
  - gen53 is the main teacher, with extra self-play from gen52 Arms R and LR;
  - a hole-scan opponent pool (v29 and gen52 Arms A, B, C, L, LR and R);
  - ramped game-result labels.

  Managed runs `gen54_production`, then `gen54_followup` (gate v4 vs v29,
  the v27-position probe, the value audit). Plan:
  `docs/plans/GEN54_PLAN.md`. Results will go to
  `docs/experiments/gen54/GEN54_RESULTS.md`.
- **Nothing is ever promoted automatically.** The owner decides, using
  `docs/protocols/PROMOTION_RULE.md`:
  - gate v4 PASS against the release;
  - a held-out mean (B2, v27) no worse than the release's minus 1 pp
    (currently 91.7%);
  - the owner's sign-off on a one-page evidence summary.

## 2. What the evidence says

- **Search stops paying above about 3,200 simulations.** v29 gains about 62
  Elo per doubling up to 3,200, then about +19 and +9. Strength has to come
  from the network (`docs/experiments/elo_rr/LADDER_RESULTS.md`).
- **The data recipe has moved strength more than architecture.**
  - The 2× wider network helped only narrowly: Arm L scored 55.25% against
    Arm B, or 50.5% counting distinct games once.
  - Its gain did not stack with the label change: Arm LR lost to Arm R
    45.1%.
- **White pessimism tracks the deep-value share under undiscounted labels.**
  Ramped game-result labels removed it (gen52 Arm R, value bias −0.005).
  Every value target is already a game result.
- **Teacher choice matters.** The pre-declared teacher-selection rule picked
  Arm R (`docs/experiments/gen53_prep/TEACHER_SELECTION.md`), and gen53, its
  student, beat it.
- **Only strong, recent opponents expose holes.** Old models expose nothing
  (`docs/experiments/gen53/MORNING_20261005.md`, hole scan). That is why
  B2 and v27 stay held out.
- **Playtesting no longer measures strength.** The owner cannot beat these
  models with either colour. Round robins and the gates are the instruments.

## 3. Models

| Model | Path | Role | SHA256 (prefix) |
|---|---|---|---|
| v29 | `models/bootstrap_v29/best_value_net.pt` | Release, gate bar | `dbf26b9e` |
| gen53 | `models/candidates/bootstrap_main_gen_0053/arena_selected.pt` | Strongest; gen54 teacher | `8f1b8803` |
| gen52 Arm R | `models/candidates/bootstrap_main_gen_0052_ramp/arena_selected.pt` | gen53's teacher; gen54 extra teacher | `bfb1857d` |
| gen52 Arm LR | `models/candidates/bootstrap_main_gen_0052_large_ramp/arena_selected.pt` | Wide tower; gen54 extra teacher | |
| gen52 Arms A, B, C, L | `models/candidates/bootstrap_main_gen_0052{,_pool,_poolcap,_large}/` | gen54 pool | |
| v28 | `models/bootstrap_v28/best_value_net.pt` | Previous release | `b651e740` |
| B2, v27 | `models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt`, `models/bootstrap_v27/best_value_net.pt` | **Held out**: never teachers or training opponents | `fc23076a`, `976294da` |

The site's Elo levels (v21 = 1600) are in `web/server.py` `ENGINES`. They come
from the joint ladder fit: v29 2450 and gen53 2555. Use `arena_selected.pt`,
the play-selected epoch, not `best_value_net.pt`, for a candidate's identity.

## 4. The generation recipe (gen53/gen54)

- **Self-play:** 2,800 teacher games at 1,600 simulations with 30-ply
  exploration, 400 forks, and reanalysis of 24k → 12k positions at 12,800.
- **Deep-value data:** 768 disagreement roots × 2 continuations at 6,400,
  relabelled with ramped game results (`tools/process_linked_extra_ramped.py`).
- **Training:** a rolling replay of the last 8 generations, the 1.9M-parameter
  tower from scratch, seed 3173.
- **Selection:** saved-epoch screens and 12,800 probes.
- **Evaluation:** gate v4 against the teacher (`tools/gate_depth.py`: two
  400-game legs at 3,200 plus a 160-game guard at 12,800), then held-out
  B2/v27, gen49, v28 and self-play diagnostics.
- **Drivers:** `tools/gen5x_campaign.py`, each with a rehearsal namespace and
  receipt-checked resume. Recipes are in `tools/recipes/`; decisions are in
  `docs/plans/gen5x_teacher_decision.json`.

## 5. Operating

- **Jobs:** always launch with
  `py -3 -B tools/runs.py start --name <name> -- <command>`. Use `status`,
  `tail --name <name>` and `stop --name <name>` (which kills the children
  too). Run one heavy GPU job at a time, at most eight workers. Never block a
  shell on a match or a gate.
- **Pinned inputs:** a running campaign re-hashes its `PINNED` files (tools,
  tests, plans, recipes) and `src/*.py`, the match tools and the native
  `.pyd` before and after every stage. Editing any of them fails the stage
  with "Inputs changed during stage". Read the live driver's `PINNED` before
  editing tools/ or tests/. Even comment-only edits to `src/*.py` change the
  runtime identity, so later gates must re-measure their par.
- **Resume:** relaunch the identical command. Completed stages are skipped by
  receipt. An interrupted training stage is refused: move its output to
  `models/archive/<name>_<timestamp>` first.
- **After a reboot nothing auto-starts.** Restart `web_server`
  (`py -3 -u web/server.py --port 8765`) and `web_tunnel`
  (`%LOCALAPPDATA%\Programs\cloudflared\cloudflared.exe tunnel run monster-chess`),
  then resume any campaign.
- **Website:** <https://chess.aaronpeng.dev> through a Cloudflare tunnel; the
  guide is `web/README.md`. It has Play (eleven Elo-rated levels) and Watch
  engines (engine vs engine with per-side search settings and both engines'
  evaluations). Games are recorded without IPs to `data/raw/web_games/`. The
  tunnel credentials JSON is secret and never goes in git.
- **Disk:** composed replays (`data/processed/bootstrap_replay_*`, about
  20 GB each) are disposable and can be rebuilt from the per-generation
  increments. **Never delete `bootstrap_new_*` or `bootstrap_extra_*`
  increments.** Retired replay manifests are in
  `data/processed/retired_replay_manifests/`.
- **Tests:** `py -3 -m pytest tests -q`. GitHub CI runs
  `-m "not local_artifacts"`; a test that reads weights, receipts, journals
  or `play.ipynb` must carry `@pytest.mark.local_artifacts`.
- **Git:** commit as `Peng-AP` with one-line messages and no co-author
  trailer. Stage explicit paths and never `git add -A`. Never commit
  `src/play.ipynb`, weights, arrays, JSONL journals or per-game `tasks/`;
  `.gitignore` encodes this for `benchmarks/`. Pushing is authorized.

## 6. Owner rules (binding)

- Plan, rehearse, then run. Pre-declare criteria; never weaken a gate after
  seeing results. A measured FAIL still runs the remaining diagnostics, while
  an execution error stops the chain.
- No automatic promotion or new generation. The owner approves.
- B2 and v27 are held out: never teachers, never training opponents.
- No hand-coded opening bans or tactical patches. Do not extend the scripted
  oracle past bare-king endings.
- A win requires capturing the king (since August 3; earlier numbers are not
  comparable).
- Report measured numbers, not narratives, and say "unknown" rather than
  extrapolating a rate.

The full list is `CONTEXT.md` §4. The rules of the game are `CONTEXT.md` §1.

## 7. Repository map

| Path | Contents |
|---|---|
| `src/` | Rules (`monster_chess.py`), encoding, search bridge (`native_mcts.py`), training, match evidence |
| `native/` | Rust rules and MCTS (`monster_native.pyd`; rollback copies kept) |
| `tools/` | Campaign drivers, `match.py`, `gate_depth.py`, Elo tools, `runs.py`, recipes |
| `web/` | Browser server and static pages |
| `tests/` | Contract tests |
| `campaigns/` | Frozen finished drivers (resume via a worktree at `b46ce1c`) |
| `docs/` | Plans, protocols, experiment results, history (`docs/README.md`) |
| `benchmarks/` | Evidence: reports and receipts committed; journals and arrays local only |
| `data/`, `iterations/`, `models/` | Games, increments, replays, generation state and weights (gitignored) |
| `logs/` | Live run logs; finished runs in `logs/archive/<date>/` |

## 8. Reading order

1. This file, then `docs/plans/GEN54_PLAN.md` and, once written, the gen54
   results.
2. `docs/experiments/gen53/` (results and the hole scan), then
   `docs/experiments/gen52/` (arms, value audit, round robins).
3. `docs/protocols/PROMOTION_RULE.md`, then `SAMPLED_GATE_PROTOCOL.md`.
4. `docs/experiments/elo_rr/` (Elo ladder and depth scaling).
5. `CONTEXT.md` (rules, laws, hazards), then `docs/history/` for anything
   older. Date-check every "current" claim in history files.
