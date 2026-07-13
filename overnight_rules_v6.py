"""Autonomous overnight pipeline: rebuild everything under the corrected
unconditional-king-capture rules (2026-07-04) and train fresh_start_v6.

Steps (each logged to OVERNIGHT_REPORT.md as it happens):
  1. Re-verify ScriptedMate vs MCTS White (16 games x 2 seeds) — informative,
     does not block the chain.
  2. Regenerate fork-safe demo start FENs (old file used the pre-fix
     generator).
  3. Generate corrected-rules corpora:
       heuristic_v2   800 games @400 sims (curriculum, live results)
       hybrid_v2      150 games @600 sims (v5 policy + heuristic values)
       mate_demos_v2  300 games @400 sims (defer hook -> ScriptedMate)
  4. Merge (demos filtered to Black wins) + human games -> combined_v4.
  5. Process -> data/processed/combined_v4.
  6. Train models/fresh_start_v6 (game_result, WDL, 30 epochs).
  7. D2 diagnostic.
  8. Benchmark v6 (20 games @400) and v5 (16 games @400, corrected-rules
     baseline) vs the heuristic anchor.

Run from the repo root:  py -3 -u overnight_rules_v6.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

REPORT = os.path.join(ROOT, "OVERNIGHT_REPORT.md")
PY = sys.executable

HEUR_DIR = os.path.join("data", "raw", "heuristic_v2")
HYBRID_DIR = os.path.join("data", "raw", "hybrid_v2")
DEMO_DIR = os.path.join("data", "raw", "mate_demos_v2")
MERGED_DIR = os.path.join("data", "raw", "combined_v4")
PROCESSED_DIR = os.path.join("data", "processed", "combined_v4")
MODEL_DIR = os.path.join("models", "fresh_start_v6")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
V5_PT = os.path.join("models", "fresh_start_v5", "best_value_net.pt")
DEMO_FENS = os.path.join("data", "start_fens", "mate_demo_starts_v2.jsonl")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args, optional=False):
    log(f"START {name}: {' '.join(args)}")
    t0 = time.time()
    proc = subprocess.run([PY, "-u"] + args, capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-12:])
    log(f"END {name} (exit={proc.returncode}, {dt:.1f} min)\n```\n{tail}\n```")
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        log(f"STDERR {name}:\n```\n{err}\n```")
        if not optional:
            raise RuntimeError(f"{name} failed (exit {proc.returncode})")
    return proc.stdout or ""


def emit_demo_fens():
    import random
    from verify_scripted_mate import random_start_fen
    rng = random.Random(20260704)
    os.makedirs(os.path.dirname(DEMO_FENS), exist_ok=True)
    seen = set()
    with open(DEMO_FENS, "w", encoding="utf-8") as f:
        while len(seen) < 300:
            fen = random_start_fen(rng)
            if fen not in seen:
                seen.add(fen)
                f.write(json.dumps({"fen": fen}) + "\n")
    log(f"Wrote {len(seen)} fork-safe demo start FENs to {DEMO_FENS}")


def merge():
    os.makedirs(MERGED_DIR, exist_ok=True)
    n_h = n_y = n_d = n_hu = dropped = 0
    for src in glob.glob(os.path.join(HEUR_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(MERGED_DIR, os.path.basename(src)))
        n_h += 1
    for src in glob.glob(os.path.join(HYBRID_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(MERGED_DIR, "game_3" + os.path.basename(src)[5:]))
        n_y += 1
    for src in glob.glob(os.path.join(DEMO_DIR, "game_*.jsonl")):
        with open(src, encoding="utf-8") as fh:
            first = fh.readline()
        if not first or json.loads(first).get("game_result", 0) >= 0:
            dropped += 1
            continue
        shutil.copy(src, os.path.join(MERGED_DIR, "game_4" + os.path.basename(src)[5:]))
        n_d += 1
    for src in glob.glob(os.path.join("data", "raw", "human_games", "**", "game_*.jsonl"),
                         recursive=True):
        shutil.copy(src, os.path.join(MERGED_DIR, f"game_5{n_hu:04d}.jsonl"))
        n_hu += 1
    log(f"Merged: heuristic={n_h} hybrid={n_y} demos={n_d} (dropped {dropped} "
        f"non-wins) human={n_hu} -> {MERGED_DIR}")


def d2():
    import numpy as np
    import torch
    from train import load_model_for_inference

    pos = np.load(os.path.join(PROCESSED_DIR, "positions.npy"), mmap_mode="r")
    res = np.load(os.path.join(PROCESSED_DIR, "game_results.npy"))
    test_idx = np.load(os.path.join(PROCESSED_DIR, "splits.npz"))["test"]
    model, _ = load_model_for_inference(MODEL_PT, torch.device("cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_idx), 512):
            idx = test_idx[i:i + 512]
            x = torch.from_numpy(np.ascontiguousarray(pos[idx])).float()
            out = model(x.permute(0, 3, 1, 2))
            v = out[0] if isinstance(out, tuple) else out
            preds.append(v.squeeze(-1).numpy())
    preds = np.concatenate(preds)
    r = res[test_idx]
    side = np.array([pos[i][0, 0, 12] for i in test_idx])
    pw = preds * side
    for name, mask, bar in [("White-win", r > 0, +0.15), ("Black-win", r < 0, -0.15)]:
        avg = float(pw[mask].mean())
        ok = avg > bar if bar > 0 else avg < bar
        log(f"D2 {name}: n={int(mask.sum())} avg={avg:+.3f} "
            f"{'PASS' if ok else 'MISS'} (bar {bar:+.2f})")


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n# Overnight corrected-rules v6 run — started "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Verification runs AFTER the training-critical chain (it blocked the
    # whole first night when a depth-2 chase search ground for hours).
    emit_demo_fens()

    run("gen-heuristic", ["src/data_generation.py", "--num-games", "800",
                          "--simulations", "400", "--curriculum",
                          "--curriculum-live-results", "--seed", "601",
                          "--output-dir", HEUR_DIR])
    run("gen-hybrid", ["src/data_generation.py", "--num-games", "150",
                       "--simulations", "600", "--curriculum",
                       "--curriculum-live-results",
                       "--use-model", V5_PT, "--hybrid-eval",
                       "--seed", "602", "--output-dir", HYBRID_DIR])
    run("gen-demos", ["src/data_generation.py", "--num-games", "300",
                      "--simulations", "400",
                      "--start-fen-file", DEMO_FENS,
                      "--record-all-plies", "--seed", "603",
                      "--output-dir", DEMO_DIR])

    merge()
    run("process", ["src/data_processor.py", "--raw-dir", MERGED_DIR,
                    "--output-dir", PROCESSED_DIR, "--seed", "42"])
    run("train", ["src/train.py", "--data-dir", PROCESSED_DIR,
                  "--model-dir", MODEL_DIR, "--target", "game_result",
                  "--value-head", "wdl", "--epochs", "30", "--seed", "42"])
    d2()
    run("benchmark-v6", ["src/benchmark.py", "--model", MODEL_PT,
                         "--games", "20", "--sims", "400",
                         "--seed", "20260704"])
    run("benchmark-v5-corrected", ["src/benchmark.py", "--model", V5_PT,
                                   "--games", "16", "--sims", "400",
                                   "--seed", "20260705"], optional=True)
    run("verify-seed7", ["src/verify_scripted_mate.py", "--games", "16",
                         "--white-sims", "200", "--max-turns", "150",
                         "--seed", "7"], optional=True)
    run("verify-seed99", ["src/verify_scripted_mate.py", "--games", "16",
                          "--white-sims", "200", "--max-turns", "150",
                          "--seed", "99"], optional=True)
    log("ALL STEPS COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"CHAIN ABORTED: {e!r}")
        raise
