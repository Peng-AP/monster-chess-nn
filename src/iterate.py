"""Self-improvement loop: generate -> process -> train -> gate -> archive.

REWORK_PLAN Phase 5 replacement for the old 3,000-line iterate.py. One
generation per cycle:

  1. generate  — hybrid self-play with the incumbent as policy generator
                 (heuristic-only when no incumbent exists), plus optional
                 black-focus games from backward-chained Black-won starts.
  2. process   — flat conversion (data_processor.py) over data/raw with a
                 bounded generation-age window.
  3. train     — game_result target, WDL head (train.py).
  4. gate      — candidate must (a) score >= --gate-threshold against the
                 incumbent over --arena-games at temperature 0, both colors,
                 no noise, and (b) not regress against the heuristic anchor
                 (benchmark.py score >= incumbent's anchor score - epsilon).
  5. archive   — promoted candidate becomes models/best_value_net.pt; every
                 candidate and its gate report are kept under
                 models/candidates/gen_<N>/.

Run:  py -3 src/iterate.py --generations 1
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

from config import (
    PROJECT_ROOT, MODEL_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    ITERATE_GAMES, ITERATE_SIMS, ITERATE_BLACKFOCUS_GAMES,
    ITERATE_ARENA_GAMES, ITERATE_ARENA_SIMS, ITERATE_GATE_THRESHOLD,
    ITERATE_ANCHOR_GAMES, ITERATE_ANCHOR_EPSILON,
    ITERATE_MAX_GENERATION_AGE, ITERATE_EPOCHS, RANDOM_SEED,
)

PY = sys.executable
INCUMBENT_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
HISTORY_PATH = os.path.join(MODEL_DIR, "iterate_history.json")


def _resolve_project_path(path):
    """Absolute path anchored at the project root for relative inputs."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _run(name, cmd):
    print(f"[{time.strftime('%H:%M:%S')}] START {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run([PY, "-u"] + cmd, cwd=PROJECT_ROOT)
    dt = (time.time() - t0) / 60
    print(f"[{time.strftime('%H:%M:%S')}] END {name} (exit={proc.returncode}, {dt:.1f} min)",
          flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed (exit {proc.returncode})")


def _next_generation(raw_dir):
    """1 + highest nn_gen<N> subdir number (1 if none)."""
    best = 0
    for path in glob.glob(os.path.join(raw_dir, "nn_gen*")):
        name = os.path.basename(path)
        digits = ""
        for ch in name[len("nn_gen"):]:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            best = max(best, int(digits))
    return best + 1


def run_arena(candidate_path, incumbent_path, games, sims, seed,
              opening_temp_plies=16):
    """Head-to-head candidate vs incumbent, both colors, no noise.

    The first opening_temp_plies plies are sampled at temperature (benchmark.
    play_one) — two NN engines at pure temp 0 are deterministic, so every
    seeded "game" replays identically and the arena silently scores a single
    game N times. Temp-diversified openings restore a real sample.

    Returns candidate score in [0, 1].
    """
    import random
    from benchmark import play_one, _build_engine

    candidate, _ = _build_engine(candidate_path, sims)
    incumbent, _ = _build_engine(incumbent_path, sims)

    n_white = games // 2
    n_black = games - n_white
    wins = draws = 0
    for i in range(n_white):
        random.seed(seed + i)
        result, _plies, _dec = play_one(candidate, incumbent,
                                        opening_temp_plies=opening_temp_plies)
        if result > 0:
            wins += 1
        elif result == 0:
            draws += 1
    for i in range(n_black):
        random.seed(seed + 1000 + i)
        result, _plies, _dec = play_one(incumbent, candidate,
                                        opening_temp_plies=opening_temp_plies)
        if result < 0:
            wins += 1
        elif result == 0:
            draws += 1
    return (wins + 0.5 * draws) / games if games else 0.0


def run_anchor(model_path, games, sims, seed):
    """Benchmark a model against the heuristic anchor; returns (score, report)."""
    from benchmark import run_benchmark
    result = run_benchmark(model_path=model_path, games=games, sims=sims,
                           anchor_sims=sims, seed=seed)
    return result["candidate_score"], result


def gate_passes(arena_score, candidate_anchor_score, incumbent_anchor_score,
                threshold, anchor_epsilon):
    """Promotion gate: beat the incumbent AND don't regress vs the anchor.

    incumbent_anchor_score may be None (no incumbent baseline recorded yet) —
    then only the arena test applies.
    """
    if arena_score < threshold:
        return False, f"arena {arena_score:.3f} < threshold {threshold:.3f}"
    if incumbent_anchor_score is not None:
        floor = incumbent_anchor_score - anchor_epsilon
        if candidate_anchor_score < floor:
            return False, (f"anchor {candidate_anchor_score:.3f} < incumbent "
                           f"{incumbent_anchor_score:.3f} - eps {anchor_epsilon:.3f}")
    return True, "passed"


def _load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"generations": []}


def _save_history(history):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _latest_incumbent_anchor_score(history):
    for entry in reversed(history.get("generations", [])):
        if entry.get("promoted"):
            return entry.get("candidate_anchor_score")
    return None


def run_generation(args):
    history = _load_history()
    gen = _next_generation(args.raw_dir)
    gen_dir = os.path.join(args.raw_dir, f"nn_gen{gen}")
    bf_dir = os.path.join(args.raw_dir, f"nn_gen{gen}_blackfocus")
    candidate_dir = os.path.join(MODEL_DIR, "candidates", f"gen_{gen}")
    candidate_pt = os.path.join(candidate_dir, "best_value_net.pt")
    incumbent = INCUMBENT_PT if os.path.exists(INCUMBENT_PT) else None

    print(f"=== Generation {gen} ===")
    print(f"Incumbent: {incumbent or 'none (heuristic-only generation)'}")

    # 1. generate self-play
    gen_cmd = ["src/data_generation.py",
               "--num-games", str(args.games),
               "--simulations", str(args.sims),
               "--curriculum", "--curriculum-live-results",
               "--record-all-plies",
               "--seed", str(args.seed + gen * 101),
               "--output-dir", gen_dir]
    if incumbent:
        gen_cmd += ["--use-model", incumbent, "--hybrid-eval"]
    _run("generate", gen_cmd)

    # 2. optional black-focus games from backward-chained Black-won starts
    if args.blackfocus_games > 0:
        bf_starts = os.path.join(
            os.path.dirname(args.raw_dir), "start_fens", f"blackfocus_gen{gen}.jsonl")
        _run("bf-starts", ["src/make_blackfocus_starts.py",
                           "--input-dir", gen_dir,
                           "--output", bf_starts])
        if os.path.getsize(bf_starts) > 0:
            bf_cmd = ["src/data_generation.py",
                      "--num-games", str(args.blackfocus_games),
                      "--simulations", str(args.sims),
                      "--start-fen-file", bf_starts,
                      "--start-fen-side", "black",
                      "--record-all-plies",
                      "--seed", str(args.seed + gen * 101 + 7),
                      "--output-dir", bf_dir]
            if incumbent:
                bf_cmd += ["--use-model", incumbent, "--hybrid-eval"]
            _run("generate-blackfocus", bf_cmd)
        else:
            print("No Black-won games to backward-chain; skipping black-focus")

    # 3. process
    _run("process", ["src/data_processor.py",
                     "--raw-dir", args.raw_dir,
                     "--output-dir", args.processed_dir,
                     "--max-generation-age", str(args.max_generation_age),
                     "--seed", str(args.seed)])

    # 4. train candidate
    _run("train", ["src/train.py",
                   "--data-dir", args.processed_dir,
                   "--model-dir", candidate_dir,
                   "--target", "game_result", "--value-head", "wdl",
                   "--epochs", str(args.epochs),
                   "--seed", str(args.seed)])

    # 5. gate
    arena_score = None
    if incumbent:
        print(f"Arena: candidate vs incumbent, {args.arena_games} games "
              f"@{args.arena_sims} sims")
        arena_score = run_arena(candidate_pt, incumbent,
                                games=args.arena_games, sims=args.arena_sims,
                                seed=args.seed + gen * 977)
        print(f"Arena score: {arena_score:.3f}")
    print(f"Anchor benchmark: {args.anchor_games} games @{args.arena_sims} sims")
    anchor_score, anchor_report = run_anchor(candidate_pt,
                                             games=args.anchor_games,
                                             sims=args.arena_sims,
                                             seed=args.seed + gen * 331)
    print(f"Anchor score: {anchor_score:.3f} "
          f"(black share {anchor_report['candidate_black_win_share']:.2f})")

    incumbent_anchor = _latest_incumbent_anchor_score(history)
    if incumbent:
        promoted, reason = gate_passes(arena_score, anchor_score, incumbent_anchor,
                                       args.gate_threshold, args.anchor_epsilon)
    else:
        promoted, reason = True, "no incumbent — first model promotes unconditionally"

    # 6. archive + promote
    entry = {
        "generation": gen,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate_path": candidate_pt,
        "incumbent_path": incumbent,
        "arena_score": arena_score,
        "candidate_anchor_score": anchor_score,
        "candidate_anchor_report": anchor_report,
        "incumbent_anchor_score": incumbent_anchor,
        "gate_threshold": args.gate_threshold,
        "anchor_epsilon": args.anchor_epsilon,
        "promoted": bool(promoted),
        "gate_reason": reason,
    }
    history["generations"].append(entry)
    _save_history(history)

    with open(os.path.join(candidate_dir, "gate_report.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)

    if promoted:
        shutil.copy(candidate_pt, INCUMBENT_PT)
        print(f"PROMOTED: gen {gen} -> {INCUMBENT_PT} ({reason})")
    else:
        print(f"REJECTED: gen {gen} stays in {candidate_dir} ({reason})")
    return promoted


def main():
    p = argparse.ArgumentParser(description="Self-improvement loop (generate/process/train/gate)")
    p.add_argument("--generations", type=int, default=1)
    p.add_argument("--games", type=int, default=ITERATE_GAMES,
                   help=f"Self-play games per generation (default: {ITERATE_GAMES})")
    p.add_argument("--blackfocus-games", type=int, default=ITERATE_BLACKFOCUS_GAMES,
                   help=f"Black-focus games per generation, 0 disables (default: {ITERATE_BLACKFOCUS_GAMES})")
    p.add_argument("--sims", type=int, default=ITERATE_SIMS,
                   help=f"MCTS simulations for generation (default: {ITERATE_SIMS})")
    p.add_argument("--arena-games", type=int, default=ITERATE_ARENA_GAMES,
                   help=f"Candidate-vs-incumbent gate games (default: {ITERATE_ARENA_GAMES})")
    p.add_argument("--arena-sims", type=int, default=ITERATE_ARENA_SIMS,
                   help=f"Simulations for arena + anchor games (default: {ITERATE_ARENA_SIMS})")
    p.add_argument("--gate-threshold", type=float, default=ITERATE_GATE_THRESHOLD,
                   help=f"Min candidate score vs incumbent (default: {ITERATE_GATE_THRESHOLD})")
    p.add_argument("--anchor-games", type=int, default=ITERATE_ANCHOR_GAMES,
                   help=f"Anchor benchmark games (default: {ITERATE_ANCHOR_GAMES})")
    p.add_argument("--anchor-epsilon", type=float, default=ITERATE_ANCHOR_EPSILON,
                   help=f"Allowed anchor-score regression (default: {ITERATE_ANCHOR_EPSILON})")
    p.add_argument("--max-generation-age", type=int, default=ITERATE_MAX_GENERATION_AGE,
                   help=f"Processing window in generations (default: {ITERATE_MAX_GENERATION_AGE})")
    p.add_argument("--epochs", type=int, default=ITERATE_EPOCHS,
                   help=f"Training epochs per generation (default: {ITERATE_EPOCHS})")
    p.add_argument("--raw-dir", type=str, default=RAW_DATA_DIR)
    p.add_argument("--processed-dir", type=str, default=PROCESSED_DATA_DIR)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = p.parse_args()

    if args.generations <= 0:
        raise ValueError("--generations must be > 0")
    if not (0.5 <= args.gate_threshold <= 1.0):
        raise ValueError("--gate-threshold must be in [0.5, 1.0]")
    args.raw_dir = _resolve_project_path(args.raw_dir)
    args.processed_dir = _resolve_project_path(args.processed_dir)

    for _ in range(args.generations):
        run_generation(args)


if __name__ == "__main__":
    main()
