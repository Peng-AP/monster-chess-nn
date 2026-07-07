"""Autonomous pipeline: loop-repeatability cycle -> fresh_start_v9.

Gate test (REWORK_PLAN Phase 5 unlock): with v8 as the hybrid policy
generator, does the retrained candidate still beat the heuristic anchor?
One new lever only — hybrid games from v8; everything else is reused from
the v8 cycle so the comparison isolates the loop effect.

Reused: heuristic_v4 (all plies), mate_demos_v3, v8_blackfocus, human games.
New:    hybrid_v3 — 150 games @600 sims, v8 policy + heuristic values
        (defer hook keeps endings real).

Steps (logged to OVERNIGHT_REPORT.md):
  1. hybrid_v3 generation
  2. merge -> combined_v7 (subdir-preserving) ; process ; train v9
  3. D2 ; benchmark v9 (20 games @400, seed 20260704 — comparable with
     v8's 0.70)

Run from the repo root:  py -3 -u overnight_hybrid_v9.py
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

HEUR_DIR = os.path.join("data", "raw", "heuristic_v4")
DEMO_DIR = os.path.join("data", "raw", "mate_demos_v3")
BF_DIR = os.path.join("data", "raw", "v8_blackfocus")
HYBRID_DIR = os.path.join("data", "raw", "hybrid_v3")
MERGED_DIR = os.path.join("data", "raw", "combined_v7")
PROCESSED_DIR = os.path.join("data", "processed", "combined_v7")
MODEL_DIR = os.path.join("models", "fresh_start_v9")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
V8_PT = os.path.join("models", "fresh_start_v8", "best_value_net.pt")


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


def merge():
    """Merge into combined_v7, preserving source-tagging subdir names."""
    os.makedirs(MERGED_DIR, exist_ok=True)
    bf_out = os.path.join(MERGED_DIR, "v8_blackfocus")
    human_out = os.path.join(MERGED_DIR, "human_games")
    os.makedirs(bf_out, exist_ok=True)
    os.makedirs(human_out, exist_ok=True)

    n_h = n_y = n_d = n_bf = n_hu = dropped = 0
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
    for src in glob.glob(os.path.join(BF_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(bf_out, os.path.basename(src)))
        n_bf += 1
    for src in glob.glob(os.path.join("data", "raw", "human_games", "**", "game_*.jsonl"),
                         recursive=True):
        shutil.copy(src, os.path.join(human_out, f"game_{n_hu:04d}.jsonl"))
        n_hu += 1
    log(f"Merged: heuristic={n_h} hybrid={n_y} demos={n_d} (dropped {dropped} "
        f"non-wins) blackfocus={n_bf} human={n_hu} -> {MERGED_DIR}")


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
        f.write(f"\n# Hybrid v9 run (loop-repeatability gate, v8 as generator) "
                f"— started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for d in (HEUR_DIR, DEMO_DIR, BF_DIR):
        if not os.path.isdir(d):
            raise RuntimeError(f"missing reused corpus: {d}")
    if not os.path.exists(V8_PT):
        raise RuntimeError(f"missing v8 model: {V8_PT}")

    run("gen-hybrid", ["src/data_generation.py", "--num-games", "150",
                       "--simulations", "600", "--curriculum",
                       "--curriculum-live-results",
                       "--use-model", V8_PT, "--hybrid-eval",
                       "--record-all-plies",
                       "--seed", "901", "--output-dir", HYBRID_DIR])

    merge()
    run("process", ["src/data_processor.py", "--raw-dir", MERGED_DIR,
                    "--output-dir", PROCESSED_DIR, "--seed", "42"])
    run("train", ["src/train.py", "--data-dir", PROCESSED_DIR,
                  "--model-dir", MODEL_DIR, "--target", "game_result",
                  "--value-head", "wdl", "--epochs", "30", "--seed", "42"])
    d2()
    run("benchmark-v9", ["src/benchmark.py", "--model", MODEL_PT,
                         "--games", "20", "--sims", "400",
                         "--seed", "20260704"])
    log("ALL STEPS COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"CHAIN ABORTED: {e!r}")
        raise
