"""Autonomous overnight Phase 4 driver (REWORK_PLAN.md Phase 4).

Waits for the already-running hybrid generation (data/raw/hybrid_v1), then:
  1. merges hybrid + heuristic corpora
  2. processes to tensors
  3. trains models/fresh_start_v4 (game_result target, WDL head)
  4. runs the D2 calibration diagnostic
  5. probes pure-NN self-play (20 games, 800 sims, standard opening)
  6. benchmarks vs the heuristic anchor (20 games, 400 sims)

Everything is logged to OVERNIGHT_REPORT.md as it happens, so a crash of any
step still leaves a readable partial report. Run from the repo root:

    py -3 overnight_phase4.py
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
HYBRID_DIR = os.path.join("data", "raw", "hybrid_v1")
HEUR_DIR = os.path.join("data", "raw", "heuristic_all")
MERGED_DIR = os.path.join("data", "raw", "combined_v2")
PROCESSED_DIR = os.path.join("data", "processed", "combined_v2")
MODEL_DIR = os.path.join("models", "fresh_start_v4")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
PROBE_DIR = os.path.join("data", "raw", "probe_nn_v4")
PY = sys.executable


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args):
    log(f"START {name}: {' '.join(args)}")
    t0 = time.time()
    proc = subprocess.run([PY, "-u"] + args, capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-15:])
    log(f"END {name} (exit={proc.returncode}, {dt:.1f} min)\n```\n{tail}\n```")
    if proc.returncode != 0:
        err_tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        log(f"STDERR {name}:\n```\n{err_tail}\n```")
        raise RuntimeError(f"{name} failed (exit {proc.returncode})")
    return proc.stdout or ""


def wait_for_generation(max_hours=10):
    marker = os.path.join(HYBRID_DIR, "generation_summary.json")
    log(f"Waiting for hybrid generation to finish ({marker})...")
    deadline = time.time() + max_hours * 3600
    while not os.path.exists(marker):
        if time.time() > deadline:
            raise RuntimeError("hybrid generation did not finish within "
                               f"{max_hours} h — aborting chain")
        time.sleep(60)
    time.sleep(10)  # let the writer finish
    with open(marker, encoding="utf-8") as f:
        summary = json.load(f)
    log(f"Hybrid generation done: {json.dumps(summary)[:400]}")


def merge_corpora():
    os.makedirs(MERGED_DIR, exist_ok=True)
    n = 0
    for src in glob.glob(os.path.join(HEUR_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(MERGED_DIR, os.path.basename(src)))
        n += 1
    # hybrid games get a "game_3" prefix so ids never collide with the
    # heuristic corpus (game_0xxxx and game_1xxxx are taken)
    m = 0
    for src in glob.glob(os.path.join(HYBRID_DIR, "game_*.jsonl")):
        base = os.path.basename(src)
        shutil.copy(src, os.path.join(MERGED_DIR, "game_3" + base[5:]))
        m += 1
    log(f"Merged corpora: {n} heuristic + {m} hybrid -> {MERGED_DIR}")


def d2_diagnostic():
    import numpy as np
    import torch
    from train import load_model_for_inference

    pos = np.load(os.path.join(PROCESSED_DIR, "positions.npy"), mmap_mode="r")
    res = np.load(os.path.join(PROCESSED_DIR, "game_results.npy"))
    test_idx = np.load(os.path.join(PROCESSED_DIR, "splits.npz"))["test"]

    model, _meta = load_model_for_inference(MODEL_PT, torch.device("cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_idx), 512):
            idx = test_idx[i:i + 512]
            x = torch.from_numpy(np.ascontiguousarray(pos[idx])).float()
            x = x.permute(0, 3, 1, 2)
            out = model(x)
            v = out[0] if isinstance(out, tuple) else out
            preds.append(v.squeeze(-1).numpy())
    preds = np.concatenate(preds)
    r = res[test_idx]
    side = np.array([pos[i][0, 0, 12] for i in test_idx])
    pw = preds * side  # white-perspective predictions

    lines = ["D2 diagnostic (white-perspective avg prediction by actual result):"]
    ok_white = ok_black = False
    for name, mask, bar in [("White-win", r > 0, +0.15), ("Black-win", r < 0, -0.15)]:
        avg = float(pw[mask].mean())
        passed = avg > bar if bar > 0 else avg < bar
        if bar > 0:
            ok_white = passed
        else:
            ok_black = passed
        lines.append(f"  {name}: n={int(mask.sum())} avg={avg:+.3f} "
                     f"bar={'>' if bar > 0 else '<'}{bar:+.2f} "
                     f"{'PASS' if passed else 'MISS'}")
    log("\n".join(lines))
    return ok_white, ok_black


def probe_stats():
    lengths = []
    results = []
    for f in glob.glob(os.path.join(PROBE_DIR, "game_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            recs = fh.read().strip().splitlines()
        if not recs:
            continue
        lengths.append(len(recs))
        results.append(json.loads(recs[0])["game_result"])
    if not results:
        log("PROBE: no games saved — FAIL")
        return
    black_wins = sum(1 for x in results if x < 0)
    mean_len = sum(lengths) / len(lengths)
    verdict = "PASS" if black_wins >= 1 and mean_len > 15 else "FAIL"
    log(f"PROBE pure-NN self-play: games={len(results)} black_wins={black_wins} "
        f"mean_recorded_plies={mean_len:.1f} -> {verdict} "
        f"(criteria: >=1 Black win, mean length > 15)")


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n# Overnight Phase 4 run — started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    wait_for_generation()
    merge_corpora()
    run("process", ["src/data_processor.py", "--raw-dir", MERGED_DIR,
                    "--output-dir", PROCESSED_DIR, "--seed", "42"])
    run("train", ["src/train.py", "--data-dir", PROCESSED_DIR,
                  "--model-dir", MODEL_DIR, "--target", "game_result",
                  "--value-head", "wdl", "--epochs", "30", "--seed", "42"])
    d2_diagnostic()
    run("probe", ["src/data_generation.py", "--num-games", "20",
                  "--simulations", "800",
                  "--use-model", MODEL_PT,
                  "--seed", "500", "--output-dir", PROBE_DIR])
    probe_stats()
    run("benchmark", ["src/benchmark.py", "--model", MODEL_PT,
                      "--games", "20", "--sims", "400", "--seed", "20260703"])
    log("ALL STEPS COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"CHAIN ABORTED: {e!r}")
        raise
