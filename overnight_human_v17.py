"""v17: frozen v16 recipe plus White-runner promotion contrast.

Safety rules:
  - Only ``promo_white_runner`` starts are generated. Black-runner starts are
    diagnostic/human-only and never enter the corpus from engine generation.
  - All promotion-game values remain outcome-grounded. Failed Black defenses
    have Black policy weight 0, so the value head sees failure without the
    policy head imitating it.
  - Every output path must be new; reruns cannot silently retain stale files.
  - Corpus, promotion, anchor, and head-to-head gates are mandatory.

Run only on explicit owner go:  py -3 -u overnight_human_v17.py
"""
import glob
import hashlib
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

BF_STARTS = os.path.join("data", "start_fens", "human_bf_starts_v5.jsonl")
WF_STARTS = os.path.join("data", "start_fens", "human_wf_starts_v4.jsonl")
PROMO_STARTS = os.path.join("data", "start_fens", "promo_races_probe.jsonl")

BF_DIR = os.path.join("data", "raw", "human_blackfocus_v17")
WF_DIR = os.path.join("data", "raw", "v17_whitefocus")
PROMO_RAW_DIR = os.path.join("data", "raw", "promo_races_v17_raw")
PROMO_DIR = os.path.join("data", "raw", "promo_races_v17")
MERGED_DIR = os.path.join("data", "raw", "combined_v13")
REFERENCE_DIR = os.path.join("data", "raw", "combined_v12")
PROCESSED_DIR = os.path.join("data", "processed", "combined_v13")
MODEL_DIR = os.path.join("models", "fresh_start_v17")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
INCUMBENT_PT = os.path.join("models", "fresh_start_v16", "best_value_net.pt")

HUMAN_DUP = 6
PROMO_GAMES = 160


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
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-18:])
    log(f"END {name} (exit={proc.returncode}, {dt:.1f} min)\n```\n{tail}\n```")
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").strip().splitlines()[-16:])
        log(f"STDERR {name}:\n```\n{err}\n```")
        raise RuntimeError(f"{name} failed (exit {proc.returncode})")
    return proc.stdout or ""


def _require_new_outputs(paths):
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        raise RuntimeError(
            "v17 output paths already exist; refusing stale merge: "
            + ", ".join(existing)
        )


def _win_filtered_copy(src_dir, dst_dir, want_result):
    os.makedirs(dst_dir, exist_ok=False)
    kept = dropped = 0
    for src in glob.glob(os.path.join(src_dir, "game_*.jsonl")):
        with open(src, encoding="utf-8") as f:
            first = f.readline()
        if not first or json.loads(first).get("game_result", 0) != want_result:
            dropped += 1
            continue
        shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))
        kept += 1
    return kept, dropped


def _directory_digest(path):
    digest = hashlib.sha256()
    for file_path in sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)):
        digest.update(os.path.relpath(file_path, path).replace(os.sep, "/").encode())
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def merge():
    os.makedirs(MERGED_DIR, exist_ok=False)
    human_out = os.path.join(MERGED_DIR, "human_games")
    promo_out = os.path.join(MERGED_DIR, "promo_races")
    os.makedirs(human_out, exist_ok=False)
    os.makedirs(promo_out, exist_ok=False)

    n_h = n_d = dropped_d = n_hu = n_promo = 0
    for src in glob.glob(os.path.join(HEUR_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(MERGED_DIR, os.path.basename(src)))
        n_h += 1
    for src in glob.glob(os.path.join(DEMO_DIR, "game_*.jsonl")):
        with open(src, encoding="utf-8") as f:
            first = f.readline()
        if not first or json.loads(first).get("game_result", 0) >= 0:
            dropped_d += 1
            continue
        shutil.copy(src, os.path.join(MERGED_DIR, "game_7" + os.path.basename(src)[5:]))
        n_d += 1

    n_bf, drop_bf = _win_filtered_copy(
        BF_DIR, os.path.join(MERGED_DIR, "human_blackfocus"), -1)
    n_wf, drop_wf = _win_filtered_copy(
        WF_DIR, os.path.join(MERGED_DIR, "whitefocus"), 1)

    for src in glob.glob(os.path.join(PROMO_DIR, "game_*.jsonl")):
        shutil.copy(src, os.path.join(promo_out, os.path.basename(src)))
        n_promo += 1

    dup_dirs = {os.path.normpath(HUMAN_B_DIR), os.path.normpath(HUMAN_CUR_DIR)}
    for src in glob.glob(
        os.path.join("data", "raw", "human_games", "**", "game_*.jsonl"),
        recursive=True,
    ):
        dst = os.path.join(human_out, f"game_{n_hu:04d}.jsonl")
        if os.path.normpath(os.path.dirname(src)) in dup_dirs:
            with open(src, encoding="utf-8") as f:
                content = f.read()
            if not content.endswith("\n"):
                content += "\n"
            with open(dst, "w", encoding="utf-8") as f:
                f.write(content * HUMAN_DUP)
        else:
            shutil.copy(src, dst)
        n_hu += 1

    manifest = {
        "recipe": "v16 frozen base + White-runner contrast",
        "heuristic": n_h,
        "demos": n_d,
        "demos_dropped": dropped_d,
        "blackfocus": n_bf,
        "blackfocus_dropped": drop_bf,
        "whitefocus": n_wf,
        "whitefocus_dropped": drop_wf,
        "promo_white_runner_games": n_promo,
        "generated_black_runner_games": 0,
        "human_games": n_hu,
        "human_duplication": HUMAN_DUP,
    }
    manifest["sha256"] = _directory_digest(MERGED_DIR)
    with open(os.path.join(MERGED_DIR, "corpus_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"Merged v17 corpus: {json.dumps(manifest, sort_keys=True)}")


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
            preds.append(model(x.permute(0, 3, 1, 2))[0].squeeze(-1).numpy())
    preds = np.concatenate(preds)
    r = res[test_idx]
    side = np.array([pos[i][0, 0, 12] for i in test_idx])
    pw = preds * side
    for name, mask, bar in (("White-win", r > 0, 0.15), ("Black-win", r < 0, -0.15)):
        avg = float(pw[mask].mean())
        ok = avg > bar if bar > 0 else avg < bar
        log(f"D2 {name}: n={int(mask.sum())} avg={avg:+.3f} "
            f"{'PASS' if ok else 'MISS'} (bar {bar:+.2f})")


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n# v17 White-runner contrast run — started "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for req in (
        HEUR_DIR, DEMO_DIR, HUMAN_B_DIR, HUMAN_CUR_DIR, PROBE_B_DIR,
        PROBE_W_DIR, PROMO_STARTS, REFERENCE_DIR, INCUMBENT_PT,
    ):
        if not os.path.exists(req):
            raise RuntimeError(f"missing input: {req}")
    _require_new_outputs((
        BF_STARTS, WF_STARTS, BF_DIR, WF_DIR, PROMO_RAW_DIR, PROMO_DIR,
        MERGED_DIR, PROCESSED_DIR, MODEL_DIR,
    ))

    run("make-bf-starts", [
        "src/make_blackfocus_starts.py",
        "--input-dir", HUMAN_B_DIR, "--input-dir", HUMAN_CUR_DIR,
        "--input-dir", PROBE_B_DIR, "--offsets", "4,8,12,16",
        "--output", BF_STARTS,
    ])
    run("make-wf-starts", [
        "src/make_blackfocus_starts.py", "--side", "white",
        "--input-dir", HUMAN_CUR_DIR, "--input-dir", PROBE_W_DIR,
        "--offsets", "4,8,12,16,22", "--output", WF_STARTS,
    ])
    run("gen-blackfocus", [
        "src/data_generation.py", "--num-games", "300", "--simulations", "400",
        "--start-fen-file", BF_STARTS, "--start-fen-side", "black",
        "--record-all-plies", "--seed", "1602", "--output-dir", BF_DIR,
    ])
    run("gen-whitefocus", [
        "src/data_generation.py", "--num-games", "250", "--simulations", "400",
        "--start-fen-file", WF_STARTS, "--start-fen-side", "white",
        "--record-all-plies", "--seed", "1603", "--output-dir", WF_DIR,
    ])
    run("gen-promo-white-runner", [
        "src/data_generation.py", "--num-games", str(PROMO_GAMES),
        "--simulations", "400", "--start-fen-file", PROMO_STARTS,
        "--start-fen-source", "promo_white_runner", "--record-all-plies",
        "--seed", "1604", "--output-dir", PROMO_RAW_DIR,
    ])
    run("prepare-promo-policy", [
        "src/promotion_data.py", PROMO_RAW_DIR, PROMO_DIR,
        "--expected-start-source", "promo_white_runner",
    ])

    merge()
    run("pretrain-gate", [
        "tools/pretrain_check.py", MERGED_DIR, "--reference", REFERENCE_DIR,
    ])
    run("process", [
        "src/data_processor.py", "--raw-dir", MERGED_DIR,
        "--output-dir", PROCESSED_DIR, "--seed", "42",
    ])
    run("train", [
        "src/train.py", "--data-dir", PROCESSED_DIR, "--model-dir", MODEL_DIR,
        "--target", "game_result", "--value-head", "wdl",
        "--epochs", "30", "--seed", "42",
    ])
    d2()

    # Targeted Black defense: strict no-regression on prevention and king safety.
    run("promotion-gate-black-defends", [
        "tools/promotion_probe.py", "--candidate", MODEL_PT,
        "--incumbent", INCUMBENT_PT, "--start-fen-file", PROMO_STARTS,
        "--source", "promo_white_runner", "--defender", "black",
        "--sims", "400", "--workers", "6", "--enforce",
        "--max-prevention-drop", "0", "--max-king-survival-drop", "0",
        "--max-score-drop", "0.05",
    ])
    # White defense receives no generated Black-runner teaching. Gate only
    # against material regression, with 3/30 prevention tolerance for noise.
    run("promotion-gate-white-defends", [
        "tools/promotion_probe.py", "--candidate", MODEL_PT,
        "--incumbent", INCUMBENT_PT, "--start-fen-file", PROMO_STARTS,
        "--source", "promo_black_runner", "--defender", "white",
        "--sims", "400", "--workers", "6", "--enforce",
        "--max-prevention-drop", "0.10", "--max-king-survival-drop", "0.05",
        "--max-score-drop", "0.10",
    ])

    run("benchmark-v17", [
        "src/benchmark.py", "--model", MODEL_PT, "--games", "60",
        "--sims", "400", "--seed", "20260704",
    ])
    run("match-v17-v16", [
        "tools/match.py", "--model-a", MODEL_PT, "--model-b", INCUMBENT_PT,
        "--games", "20", "--workers", "6",
    ])
    log("ALL STEPS COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"CHAIN ABORTED: {exc!r}")
        raise
