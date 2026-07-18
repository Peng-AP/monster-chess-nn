"""Autonomous cycle: human pawn-phase demos + behind-cliff amplification -> v11.

Grounded in the 2026-07-09 diagnosis (tools/corpus_audit.py + restart probes):
Black's bottleneck is the PAWN PHASE (play while White still has pawns).
Heuristic-Black converts owner-won pawn-phase positions at ~7-14%; once White's
pawns are gone conversion is 71-100%. Prior blackfocus curricula never touched
this phase (their starts average 3.6 Black men — endgame amplification only).

Levers over the v10 cycle:
  1. Fresh human corpus: 10/10 owner Black wins vs the aggressive anchor
     (data/raw/human_games/black_2026_07), each traversing the pawn phase.
     Duplicated x6 at merge time by repeating records WITHIN one file per game
     (keeps game-level split integrity — no train/test leakage) to counter the
     flat processor's dilution of human data to ~0.4% of train mass.
  2. human_blackfocus: amplification restarts chained ONLY from behind the
     cliff (offsets <=16, ~0-0.5 White pawns, measured 71% Black conversion)
     from the fresh human wins + the probe's Black-won games. Restarts at or
     before the cliff measured 14-33% Black — poison, excluded.
  3. Benchmark at 60 games (30 per side): n=10 per side cannot resolve deltas
     < ~0.3 (the v10 "Black collapse" was Fisher p~0.21).

Steps (logged to OVERNIGHT_REPORT.md):
  1. mate_demos_v5   300 games @400 sims (defer hook -> ScriptedMate Black)
  2. heuristic_v6    800 games @400 sims (curriculum, all plies)
  3. human-bf starts <- black_2026_07 + probe_human_v2 Black wins (offsets 4-16)
  4. human_blackfocus 300 games @400 sims from those starts
  5. merge -> combined_v9 ; corpus audit ; process ; train v11
  6. D2 ; benchmark v11 (60 games @400, seed 20260704) vs the aggressive anchor

Run from the repo root:  py -3 -u overnight_human_v11.py
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

DEMO_DIR = os.path.join("data", "raw", "mate_demos_v5")
HEUR_DIR = os.path.join("data", "raw", "heuristic_v6")
HUMAN_NEW_DIR = os.path.join("data", "raw", "human_games", "black_2026_07")
PROBE_DIR = os.path.join("data", "raw", "probe_human_v2")
BF_STARTS = os.path.join("data", "start_fens", "human_bf_starts_v1.jsonl")
BF_DIR = os.path.join("data", "raw", "human_blackfocus")
MERGED_DIR = os.path.join("data", "raw", "combined_v9")
PROCESSED_DIR = os.path.join("data", "processed", "combined_v9")
MODEL_DIR = os.path.join("models", "fresh_start_v11")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
DEMO_FENS = os.path.join("data", "start_fens", "mate_demo_starts_v2.jsonl")

HUMAN_NEW_DUP = 6  # x-factor for the fresh pawn-phase wins


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
    human_out = os.path.join(MERGED_DIR, "human_games")
    os.makedirs(bf_out, exist_ok=True)
    os.makedirs(human_out, exist_ok=True)

    n_h = n_d = n_bf = n_hu = dropped = 0
    for src in glob.glob(os.path.join(HEUR_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(MERGED_DIR, os.path.basename(src)))
        n_h += 1
    for src in glob.glob(os.path.join(DEMO_DIR, "game_*.jsonl")):
        with open(src, encoding="utf-8") as fh:
            first = fh.readline()
        if not first or json.loads(first).get("game_result", 0) >= 0:
            dropped += 1
            continue
        shutil.copy(src, os.path.join(MERGED_DIR, "game_5" + os.path.basename(src)[5:]))
        n_d += 1
    for src in glob.glob(os.path.join(BF_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(bf_out, os.path.basename(src)))
        n_bf += 1

    # Human games. Fresh pawn-phase wins are duplicated x HUMAN_NEW_DUP by
    # repeating the records INSIDE one file per game: the processor splits at
    # game (=file) level, so all copies land in the same train/val/test split
    # (separate files would leak identical positions across splits).
    new_set = set(glob.glob(os.path.join(HUMAN_NEW_DIR, "game_*.jsonl")))
    for src in glob.glob(os.path.join("data", "raw", "human_games", "**", "game_*.jsonl"),
                         recursive=True):
        dst = os.path.join(human_out, f"game_{n_hu:04d}.jsonl")
        if src in new_set or os.path.normpath(src) in {os.path.normpath(p) for p in new_set}:
            with open(src, encoding="utf-8") as fh:
                content = fh.read()
            if not content.endswith("\n"):
                content += "\n"
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(content * HUMAN_NEW_DUP)
        else:
            shutil.copy(src, dst)
        n_hu += 1
    log(f"Merged: heuristic={n_h} demos={n_d} (dropped {dropped} non-wins) "
        f"human_blackfocus={n_bf} human={n_hu} (x{HUMAN_NEW_DUP} on black_2026_07) "
        f"-> {MERGED_DIR}")


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
        f.write(f"\n# Human pawn-phase v11 run (behind-cliff amplification, x{HUMAN_NEW_DUP} "
                f"human) — started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    if not os.path.exists(DEMO_FENS):
        raise RuntimeError(f"missing demo start FENs: {DEMO_FENS}")
    if not glob.glob(os.path.join(HUMAN_NEW_DIR, "game_*.jsonl")):
        raise RuntimeError(f"missing fresh human wins: {HUMAN_NEW_DIR}")

    run("gen-demos", ["src/data_generation.py", "--num-games", "300",
                      "--simulations", "400", "--start-fen-file", DEMO_FENS,
                      "--record-all-plies", "--seed", "1204",
                      "--output-dir", DEMO_DIR])
    run("gen-heuristic", ["src/data_generation.py", "--num-games", "800",
                          "--simulations", "400", "--curriculum",
                          "--curriculum-live-results", "--record-all-plies",
                          "--seed", "1201", "--output-dir", HEUR_DIR])
    # Behind-cliff starts ONLY (offsets <=16 from game end; measured 71% Black
    # conversion). make_blackfocus_starts keeps result == -1 games only.
    run("make-human-bf-starts", ["src/make_blackfocus_starts.py",
                                 "--input-dir", HUMAN_NEW_DIR,
                                 "--input-dir", PROBE_DIR,
                                 "--offsets", "4,8,12,16",
                                 "--output", BF_STARTS])
    run("gen-human-blackfocus", ["src/data_generation.py", "--num-games", "300",
                                 "--simulations", "400",
                                 "--start-fen-file", BF_STARTS,
                                 "--start-fen-side", "black",
                                 "--record-all-plies", "--seed", "1202",
                                 "--output-dir", BF_DIR])

    merge()
    run("corpus-audit", ["tools/corpus_audit.py", MERGED_DIR], optional=True)
    run("process", ["src/data_processor.py", "--raw-dir", MERGED_DIR,
                    "--output-dir", PROCESSED_DIR, "--seed", "42"])
    run("train", ["src/train.py", "--data-dir", PROCESSED_DIR,
                  "--model-dir", MODEL_DIR, "--target", "game_result",
                  "--value-head", "wdl", "--epochs", "30", "--seed", "42"])
    d2()
    run("benchmark-v11", ["src/benchmark.py", "--model", MODEL_PT,
                          "--games", "60", "--sims", "400",
                          "--seed", "20260704"])
    log("ALL STEPS COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"CHAIN ABORTED: {e!r}")
        raise
