"""Autonomous cycle: v15 recipe + new curriculum batch + engine king-safety -> v16.

Process rules now enforced (2026-07-12 pre-mortem):
  - FROZEN base corpora: heuristic_v8 + mate_demos_v7 are REUSED, not
    regenerated — one variable per cycle. New variables: the latest human
    curriculum games, fresh focus sources (generated with the engine-level
    king-safety override now in MCTS), and nothing else.
  - WIN-FILTERED focus merges: blackfocus AND whitefocus keep only games won
    by their side (the v14 lesson — 73%-pure blackfocus regressed Black).
  - MANDATORY pretrain gate (tools/pretrain_check.py) before training; the
    chain aborts on FAIL.
  - Head-to-head match vs the incumbent (v15) is part of the chain — the
    anchor is saturated for White measurement.

Steps (logged to OVERNIGHT_REPORT.md):
  1. bf starts  <- black_2026_07 + curriculum + probe_human_v2 (offsets 4-16)
     wf starts <- curriculum + probe_whitefocus (offsets 4-22, --side white)
  2. human_blackfocus_v16 300 games ; v16_whitefocus 250 games (@400 sims)
  3. merge -> combined_v12 (win-filtered focus, human x6 in-file)
  4. pretrain gate (reference: combined_v11f) ; process ; train v16
  5. D2 ; benchmark v16 (60 games @400, seed 20260704)
  6. match v16 vs v15 (20 games, tools/match.py)

Run from the repo root:  py -3 -u overnight_human_v16.py
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

DEMO_DIR = os.path.join("data", "raw", "mate_demos_v7")      # frozen base
HEUR_DIR = os.path.join("data", "raw", "heuristic_v8")       # frozen base
HUMAN_B_DIR = os.path.join("data", "raw", "human_games", "black_2026_07")
HUMAN_CUR_DIR = os.path.join("data", "raw", "human_games", "curriculum_2026_07")
PROBE_B_DIR = os.path.join("data", "raw", "probe_human_v2")
PROBE_W_DIR = os.path.join("data", "raw", "probe_whitefocus")
BF_STARTS = os.path.join("data", "start_fens", "human_bf_starts_v4.jsonl")
WF_STARTS = os.path.join("data", "start_fens", "human_wf_starts_v3.jsonl")
BF_DIR = os.path.join("data", "raw", "human_blackfocus_v16")
WF_DIR = os.path.join("data", "raw", "v16_whitefocus")
MERGED_DIR = os.path.join("data", "raw", "combined_v12")
REFERENCE_DIR = os.path.join("data", "raw", "combined_v11f")
PROCESSED_DIR = os.path.join("data", "processed", "combined_v12")
MODEL_DIR = os.path.join("models", "fresh_start_v16")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
INCUMBENT_PT = os.path.join("models", "fresh_start_v15", "best_value_net.pt")

HUMAN_DUP = 6


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
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-14:])
    log(f"END {name} (exit={proc.returncode}, {dt:.1f} min)\n```\n{tail}\n```")
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        log(f"STDERR {name}:\n```\n{err}\n```")
        if not optional:
            raise RuntimeError(f"{name} failed (exit {proc.returncode})")
    return proc.stdout or ""


def _win_filtered_copy(src_dir, dst_dir, want_result):
    """Copy only games whose result matches the side the source teaches."""
    os.makedirs(dst_dir, exist_ok=True)
    kept = dropped = 0
    for src in glob.glob(os.path.join(src_dir, "game_*.jsonl")):
        with open(src, encoding="utf-8") as fh:
            first = fh.readline()
        if not first or json.loads(first).get("game_result", 0) != want_result:
            dropped += 1
            continue
        shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))
        kept += 1
    return kept, dropped


def merge():
    os.makedirs(MERGED_DIR, exist_ok=True)
    human_out = os.path.join(MERGED_DIR, "human_games")
    os.makedirs(human_out, exist_ok=True)

    n_h = n_d = dropped_d = n_hu = 0
    for src in glob.glob(os.path.join(HEUR_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(MERGED_DIR, os.path.basename(src)))
        n_h += 1
    for src in glob.glob(os.path.join(DEMO_DIR, "game_*.jsonl")):
        with open(src, encoding="utf-8") as fh:
            first = fh.readline()
        if not first or json.loads(first).get("game_result", 0) >= 0:
            dropped_d += 1
            continue
        shutil.copy(src, os.path.join(MERGED_DIR, "game_7" + os.path.basename(src)[5:]))
        n_d += 1

    # Win-filtered amplification sources (v14 lesson, now permanent).
    n_bf, drop_bf = _win_filtered_copy(BF_DIR, os.path.join(MERGED_DIR, "human_blackfocus"), -1)
    n_wf, drop_wf = _win_filtered_copy(WF_DIR, os.path.join(MERGED_DIR, "whitefocus"), 1)

    dup_dirs = {os.path.normpath(HUMAN_B_DIR), os.path.normpath(HUMAN_CUR_DIR)}
    for src in glob.glob(os.path.join("data", "raw", "human_games", "**", "game_*.jsonl"),
                         recursive=True):
        dst = os.path.join(human_out, f"game_{n_hu:04d}.jsonl")
        if os.path.normpath(os.path.dirname(src)) in dup_dirs:
            with open(src, encoding="utf-8") as fh:
                content = fh.read()
            if not content.endswith("\n"):
                content += "\n"
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(content * HUMAN_DUP)
        else:
            shutil.copy(src, dst)
        n_hu += 1
    log(f"Merged: heuristic={n_h} (frozen) demos={n_d} (dropped {dropped_d}) "
        f"blackfocus={n_bf} (dropped {drop_bf} non-wins) "
        f"whitefocus={n_wf} (dropped {drop_wf} non-wins) "
        f"human={n_hu} (x{HUMAN_DUP} fresh dirs) -> {MERGED_DIR}")


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
        f.write(f"\n# v16 run (frozen base + new curriculum + engine king-safety, "
                f"gated) — started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for req in (HEUR_DIR, DEMO_DIR, HUMAN_B_DIR, HUMAN_CUR_DIR, INCUMBENT_PT):
        if not os.path.exists(req):
            raise RuntimeError(f"missing input: {req}")

    run("make-bf-starts", ["src/make_blackfocus_starts.py",
                           "--input-dir", HUMAN_B_DIR,
                           "--input-dir", HUMAN_CUR_DIR,
                           "--input-dir", PROBE_B_DIR,
                           "--offsets", "4,8,12,16",
                           "--output", BF_STARTS])
    run("make-wf-starts", ["src/make_blackfocus_starts.py", "--side", "white",
                           "--input-dir", HUMAN_CUR_DIR,
                           "--input-dir", PROBE_W_DIR,
                           "--offsets", "4,8,12,16,22",
                           "--output", WF_STARTS])
    run("gen-blackfocus", ["src/data_generation.py", "--num-games", "300",
                           "--simulations", "400",
                           "--start-fen-file", BF_STARTS,
                           "--start-fen-side", "black",
                           "--record-all-plies", "--seed", "1502",
                           "--output-dir", BF_DIR])
    run("gen-whitefocus", ["src/data_generation.py", "--num-games", "250",
                           "--simulations", "400",
                           "--start-fen-file", WF_STARTS,
                           "--start-fen-side", "white",
                           "--record-all-plies", "--seed", "1503",
                           "--output-dir", WF_DIR])

    merge()
    run("pretrain-gate", ["tools/pretrain_check.py", MERGED_DIR,
                          "--reference", REFERENCE_DIR])   # NOT optional
    run("process", ["src/data_processor.py", "--raw-dir", MERGED_DIR,
                    "--output-dir", PROCESSED_DIR, "--seed", "42"])
    run("train", ["src/train.py", "--data-dir", PROCESSED_DIR,
                  "--model-dir", MODEL_DIR, "--target", "game_result",
                  "--value-head", "wdl", "--epochs", "30", "--seed", "42"])
    d2()
    run("benchmark-v16", ["src/benchmark.py", "--model", MODEL_PT,
                          "--games", "60", "--sims", "400",
                          "--seed", "20260704"])
    run("match-v16-vs-v15", ["tools/match.py", "--model-a", MODEL_PT,
                             "--model-b", INCUMBENT_PT,
                             "--games", "20", "--workers", "6"])
    log("ALL STEPS COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"CHAIN ABORTED: {e!r}")
        raise
