"""v18: general learning cleanup after human game 0015.

Changes from v17:
  - no promotion-specific generation, preparation, or gate;
  - all v17 focus outcomes are retained rather than winner-filtered;
  - each human game appears once, with policy trust decided centrally by
    data_processor.policy_weight_for_record;
  - symmetric 17-channel position encoding;
  - hybrid WDL + progress value training;
  - post-training matches are informational and never promotion gates.

Explicit owner go received 2026-07-13. Run with:
    py -3 -u overnight_human_v18.py
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

REPORT = os.path.join(ROOT, "V18_REPORT.md")
PY = sys.executable

HEUR_DIR = os.path.join("data", "raw", "heuristic_v8")
DEMO_DIR = os.path.join("data", "raw", "mate_demos_v7")
BF_DIR = os.path.join("data", "raw", "human_blackfocus_v17")
WF_DIR = os.path.join("data", "raw", "v17_whitefocus")
HUMAN_DIR = os.path.join("data", "raw", "human_games")

MERGED_DIR = os.path.join("data", "raw", "combined_v14")
REFERENCE_DIR = os.path.join("data", "raw", "combined_v12")
PROCESSED_DIR = os.path.join("data", "processed", "combined_v14")
MODEL_DIR = os.path.join("models", "fresh_start_v18")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
INCUMBENT_PT = os.path.join("models", "fresh_start_v17", "best_value_net.pt")

TRACE_GAME = os.path.join(
    HUMAN_DIR, "black_2026_07", "game_00015.jsonl")
VALUE_HORIZON = 225
VALUE_FLOOR = 0.5


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
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-24:])
    log(f"END {name} (exit={proc.returncode}, {dt:.1f} min)\n```\n{tail}\n```")
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").strip().splitlines()[-20:])
        log(f"STDERR {name}:\n```\n{err}\n```")
        raise RuntimeError(f"{name} failed (exit {proc.returncode})")
    return proc.stdout or ""


def _require_new_outputs(paths):
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        raise RuntimeError(
            "v18 output paths already exist; refusing stale merge: "
            + ", ".join(existing)
        )


def _game_result(path):
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    return float(json.loads(first).get("game_result", 0.0)) if first else 0.0


def _result_counts(paths):
    counts = {"white": 0, "black": 0, "draw": 0}
    for path in paths:
        result = _game_result(path)
        if result > 0:
            counts["white"] += 1
        elif result < 0:
            counts["black"] += 1
        else:
            counts["draw"] += 1
    return counts


def _directory_digest(path):
    digest = hashlib.sha256()
    for file_path in sorted(glob.glob(
            os.path.join(path, "**", "*.jsonl"), recursive=True)):
        digest.update(os.path.relpath(file_path, path).replace(os.sep, "/").encode())
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _copy_focus(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=False)
    paths = sorted(glob.glob(os.path.join(src_dir, "game_*.jsonl")))
    for src in paths:
        shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))
    return paths


def merge():
    os.makedirs(MERGED_DIR, exist_ok=False)
    human_out = os.path.join(MERGED_DIR, "human_games")
    os.makedirs(human_out, exist_ok=False)

    heur_paths = sorted(glob.glob(os.path.join(HEUR_DIR, "game_*.jsonl")))
    for src in heur_paths:
        shutil.copy(src, os.path.join(MERGED_DIR, os.path.basename(src)))

    demos_kept = []
    demos_dropped = 0
    for src in sorted(glob.glob(os.path.join(DEMO_DIR, "game_*.jsonl"))):
        if _game_result(src) >= 0:
            demos_dropped += 1
            continue
        shutil.copy(
            src,
            os.path.join(MERGED_DIR, "game_7" + os.path.basename(src)[5:]),
        )
        demos_kept.append(src)

    blackfocus = _copy_focus(
        BF_DIR, os.path.join(MERGED_DIR, "human_blackfocus"))
    whitefocus = _copy_focus(
        WF_DIR, os.path.join(MERGED_DIR, "whitefocus"))

    human_paths = sorted(glob.glob(
        os.path.join(HUMAN_DIR, "**", "game_*.jsonl"), recursive=True))
    for src in human_paths:
        rel = os.path.relpath(src, HUMAN_DIR)
        dst = os.path.join(human_out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    manifest = {
        "recipe": "general learning cleanup; no promotion injection",
        "heuristic_games": len(heur_paths),
        "demo_games": len(demos_kept),
        "demo_games_dropped": demos_dropped,
        "blackfocus_games": len(blackfocus),
        "blackfocus_results": _result_counts(blackfocus),
        "whitefocus_games": len(whitefocus),
        "whitefocus_results": _result_counts(whitefocus),
        "human_games": len(human_paths),
        "human_duplication": 1,
        "promotion_specific_games": 0,
        "position_encoding_channels": 17,
        "value_discount_mode": "progress",
        "value_horizon": VALUE_HORIZON,
        "value_floor": VALUE_FLOOR,
    }
    manifest["sha256"] = _directory_digest(MERGED_DIR)
    with open(os.path.join(MERGED_DIR, "corpus_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"Merged v18 corpus: {json.dumps(manifest, sort_keys=True)}")


def d2():
    import numpy as np
    import torch
    from train import load_model_for_inference

    pos = np.load(os.path.join(PROCESSED_DIR, "positions.npy"), mmap_mode="r")
    res = np.load(os.path.join(PROCESSED_DIR, "game_results.npy"))
    with np.load(os.path.join(PROCESSED_DIR, "splits.npz")) as split_file:
        test_idx = split_file["test"]
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
    white_values = preds * side
    for name, mask in (("White-result", r > 0), ("Black-result", r < 0)):
        avg = float(white_values[mask].mean())
        log(f"D2 {name}: n={int(mask.sum())} avg={avg:+.3f} (informational)")


def trace_game_0015():
    if not os.path.exists(TRACE_GAME):
        log("TRACE 0015 unavailable")
        return
    from evaluation import NNEvaluator
    from monster_chess import MonsterChessGame

    with open(TRACE_GAME, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    indices = [14, 22, 24]
    states = [MonsterChessGame(records[i]["fen"]) for i in indices]
    candidate = NNEvaluator(MODEL_PT).batch_evaluate(states)
    incumbent = NNEvaluator(INCUMBENT_PT).batch_evaluate(states)
    for i, v18, v17 in zip(indices, candidate, incumbent):
        log(f"TRACE 0015 record={i} raw_white_value v18={v18:+.3f} "
            f"v17={v17:+.3f} (informational)")


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n# v18 general learning cleanup - started "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for req in (
        HEUR_DIR, DEMO_DIR, BF_DIR, WF_DIR, HUMAN_DIR,
        REFERENCE_DIR, INCUMBENT_PT,
    ):
        if not os.path.exists(req):
            raise RuntimeError(f"missing input: {req}")
    _require_new_outputs((MERGED_DIR, PROCESSED_DIR, MODEL_DIR))

    merge()
    run("pretrain-audit", [
        "tools/pretrain_check.py", MERGED_DIR,
        "--reference", REFERENCE_DIR,
        "--min-purity", "0",
        "--max-dup", "2",
        "--value-discount-mode", "progress",
        "--value-horizon", str(VALUE_HORIZON),
        "--value-floor", str(VALUE_FLOOR),
        "--bias-fail", "0.25",
    ])
    run("process", [
        "src/data_processor.py",
        "--raw-dir", MERGED_DIR,
        "--output-dir", PROCESSED_DIR,
        "--seed", "42",
        "--value-discount-mode", "progress",
        "--value-horizon", str(VALUE_HORIZON),
        "--value-floor", str(VALUE_FLOOR),
    ])
    run("train", [
        "src/train.py",
        "--data-dir", PROCESSED_DIR,
        "--model-dir", MODEL_DIR,
        "--target", "game_result",
        "--value-head", "hybrid",
        "--epochs", "30",
        "--seed", "42",
    ])
    d2()
    trace_game_0015()

    run("anchor-v18", [
        "tools/match.py", "--model-a", MODEL_PT,
        "--games", "6", "--sims", "400", "--workers", "6",
    ])
    run("match-v18-v17", [
        "tools/match.py", "--model-a", MODEL_PT,
        "--model-b", INCUMBENT_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    log("ALL STEPS COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"CHAIN ABORTED: {exc!r}")
        raise
