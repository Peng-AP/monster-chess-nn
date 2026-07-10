"""Autonomous cycle: v12 recipe + ~40-game curriculum batch + ply-discounted value targets -> v13.

New vs v12: VALUE_TARGET_GAMMA=0.999 (processor) gives the scalar value head an
urgency gradient (anti-shuffle/anti-suicide in won positions); curriculum dir
has doubled (owner games vs v12, both sides).

Over the v11 cycle (first Black gain in four cycles):
  1. Fresh human data: 19 curriculum-session games (data/raw/human_games/
     curriculum_2026_07) — 10 White wins (the corpus's first coherent White-plan
     play) + 9 Black wins, served from pawn-phase deck positions. Duplicated x6
     in-file at merge (same split-safe trick as black_2026_07).
  2. NEW whitefocus source: restarts chained from the owner's White-won games
     (offsets <=22, measured 95% White conversion — probe_whitefocus 57/60).
     Mirrors what blackfocus does for Black conversion.
  3. blackfocus starts now also chain from curriculum Black wins.

Steps (logged to OVERNIGHT_REPORT.md):
  1. mate_demos_v7    300 games @400 sims (defer hook -> ScriptedMate Black)
  2. heuristic_v8     800 games @400 sims (curriculum, all plies)
  3. black starts  <- black_2026_07 + curriculum + probe_human_v2 (offsets 4-16)
     white starts <- curriculum + probe_whitefocus (offsets 4-22, --side white)
  4. human_blackfocus 300 games ; v13_whitefocus 250 games (@400 sims)
  5. merge -> combined_v11 ; corpus audit ; process ; train v12
  6. D2 ; benchmark v12 (60 games @400, seed 20260704) vs the aggressive anchor

Run from the repo root:  py -3 -u overnight_human_v13.py
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

DEMO_DIR = os.path.join("data", "raw", "mate_demos_v7")
HEUR_DIR = os.path.join("data", "raw", "heuristic_v8")
HUMAN_B_DIR = os.path.join("data", "raw", "human_games", "black_2026_07")
HUMAN_CUR_DIR = os.path.join("data", "raw", "human_games", "curriculum_2026_07")
PROBE_B_DIR = os.path.join("data", "raw", "probe_human_v2")
PROBE_W_DIR = os.path.join("data", "raw", "probe_whitefocus")
BF_STARTS = os.path.join("data", "start_fens", "human_bf_starts_v3.jsonl")
WF_STARTS = os.path.join("data", "start_fens", "human_wf_starts_v2.jsonl")
BF_DIR = os.path.join("data", "raw", "human_blackfocus_v13")
WF_DIR = os.path.join("data", "raw", "v13_whitefocus")
MERGED_DIR = os.path.join("data", "raw", "combined_v11")
PROCESSED_DIR = os.path.join("data", "processed", "combined_v11")
MODEL_DIR = os.path.join("models", "fresh_start_v13")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
DEMO_FENS = os.path.join("data", "start_fens", "mate_demo_starts_v2.jsonl")

HUMAN_DUP = 6  # x-factor for fresh human dirs (black_2026_07 + curriculum)


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
    os.makedirs(MERGED_DIR, exist_ok=True)
    bf_out = os.path.join(MERGED_DIR, "human_blackfocus")
    wf_out = os.path.join(MERGED_DIR, "whitefocus")
    human_out = os.path.join(MERGED_DIR, "human_games")
    for d in (bf_out, wf_out, human_out):
        os.makedirs(d, exist_ok=True)

    n_h = n_d = n_bf = n_wf = n_hu = dropped = 0
    for src in glob.glob(os.path.join(HEUR_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(MERGED_DIR, os.path.basename(src)))
        n_h += 1
    for src in glob.glob(os.path.join(DEMO_DIR, "game_*.jsonl")):
        with open(src, encoding="utf-8") as fh:
            first = fh.readline()
        if not first or json.loads(first).get("game_result", 0) >= 0:
            dropped += 1
            continue
        shutil.copy(src, os.path.join(MERGED_DIR, "game_6" + os.path.basename(src)[5:]))
        n_d += 1
    for src in glob.glob(os.path.join(BF_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(bf_out, os.path.basename(src)))
        n_bf += 1
    for src in glob.glob(os.path.join(WF_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(wf_out, os.path.basename(src)))
        n_wf += 1

    # Human games. Fresh dirs duplicated x HUMAN_DUP by repeating records
    # INSIDE one file per game (game-level split => all copies share a split;
    # separate files would leak identical positions across train/test).
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
    log(f"Merged: heuristic={n_h} demos={n_d} (dropped {dropped} non-wins) "
        f"blackfocus={n_bf} whitefocus={n_wf} human={n_hu} "
        f"(x{HUMAN_DUP} on black_2026_07+curriculum) -> {MERGED_DIR}")


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
        f.write(f"\n# Human curriculum v13 run (white+black amplification, x{HUMAN_DUP} "
                f"human) — started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for req in (DEMO_FENS, HUMAN_B_DIR, HUMAN_CUR_DIR):
        if not os.path.exists(req):
            raise RuntimeError(f"missing input: {req}")

    run("gen-demos", ["src/data_generation.py", "--num-games", "300",
                      "--simulations", "400", "--start-fen-file", DEMO_FENS,
                      "--record-all-plies", "--seed", "1404",
                      "--output-dir", DEMO_DIR])
    run("gen-heuristic", ["src/data_generation.py", "--num-games", "800",
                          "--simulations", "400", "--curriculum",
                          "--curriculum-live-results", "--record-all-plies",
                          "--seed", "1401", "--output-dir", HEUR_DIR])
    # Behind-cliff Black starts (offsets <=16, measured 71-85% Black conversion).
    run("make-bf-starts", ["src/make_blackfocus_starts.py",
                           "--input-dir", HUMAN_B_DIR,
                           "--input-dir", HUMAN_CUR_DIR,
                           "--input-dir", PROBE_B_DIR,
                           "--offsets", "4,8,12,16",
                           "--output", BF_STARTS])
    # White starts from owner White wins (offsets <=22, measured 95% conversion).
    run("make-wf-starts", ["src/make_blackfocus_starts.py", "--side", "white",
                           "--input-dir", HUMAN_CUR_DIR,
                           "--input-dir", PROBE_W_DIR,
                           "--offsets", "4,8,12,16,22",
                           "--output", WF_STARTS])
    run("gen-blackfocus", ["src/data_generation.py", "--num-games", "300",
                           "--simulations", "400",
                           "--start-fen-file", BF_STARTS,
                           "--start-fen-side", "black",
                           "--record-all-plies", "--seed", "1402",
                           "--output-dir", BF_DIR])
    run("gen-whitefocus", ["src/data_generation.py", "--num-games", "250",
                           "--simulations", "400",
                           "--start-fen-file", WF_STARTS,
                           "--start-fen-side", "white",
                           "--record-all-plies", "--seed", "1403",
                           "--output-dir", WF_DIR])

    merge()
    run("corpus-audit", ["tools/corpus_audit.py", MERGED_DIR], optional=True)
    run("process", ["src/data_processor.py", "--raw-dir", MERGED_DIR,
                    "--output-dir", PROCESSED_DIR, "--seed", "42"])
    run("train", ["src/train.py", "--data-dir", PROCESSED_DIR,
                  "--model-dir", MODEL_DIR, "--target", "game_result",
                  "--value-head", "wdl", "--epochs", "30", "--seed", "42"])
    d2()
    run("benchmark-v13", ["src/benchmark.py", "--model", MODEL_PT,
                          "--games", "60", "--sims", "400",
                          "--seed", "20260704"])
    log("ALL STEPS COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"CHAIN ABORTED: {e!r}")
        raise
