"""v18 recovery candidate: v17's proven recipe, minimally extended.

Both rejected v18 candidates bundled many changes and weakened the corpus
gates; this run is the controlled recovery. Exactly two things change vs the
run that produced v17 — everything else is frozen:

  1. Human games recorded since (black_2026_07 now includes 00015-00017,
     the games that exposed v17's pawn-run blindness) enter with the same
     6x duplication as before.
  2. Checkpoint selection uses the decisive metric (min-over-sides policy
     top-1 + winner-sign) instead of aggregate validation loss, which hid
     the WDL-v18 policy regression.

Frozen from v17: 15-plane encoding, WDL + near-mate value target
(horizon 10, floor 0.97), win-filtered focus games (reusing the v17
generation, no regeneration), promo_races_v17 with outcome-masked defense
policy (part of the approved incumbent's corpus — removing it would be a
second variable), heuristic_v8 + mate_demos_v7 base.

Gates run at DEFAULT thresholds — pretrain_check prints a WEAKENED GATE
banner if any driver ever loosens them again. New: tools/model_diff.py
enforces incumbent no-regression on decisive metrics BEFORE any match runs.
Anchor evidence is 20 games, not 6.

The trained model lands in models/candidates/ — it does not consume the
v18 name. Promotion needs: model_diff pass, anchor >= 0.55 without color
collapse, h2h vs v17 >= 0.55, then owner play (final gate).

Run only on explicit owner go:  py -3 -u overnight_human_v18b.py
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

REPORT = os.path.join(ROOT, "V18_RECOVERY_REPORT.md")
PY = sys.executable

DEMO_DIR = os.path.join("data", "raw", "mate_demos_v7")
HEUR_DIR = os.path.join("data", "raw", "heuristic_v8")
HUMAN_B_DIR = os.path.join("data", "raw", "human_games", "black_2026_07")
HUMAN_CUR_DIR = os.path.join("data", "raw", "human_games", "curriculum_2026_07")
BF_DIR = os.path.join("data", "raw", "human_blackfocus_v17")
WF_DIR = os.path.join("data", "raw", "v17_whitefocus")
PROMO_DIR = os.path.join("data", "raw", "promo_races_v17")

MERGED_DIR = os.path.join("data", "raw", "combined_v15")
REFERENCE_DIR = os.path.join("data", "raw", "combined_v13")  # incumbent's corpus
PROCESSED_DIR = os.path.join("data", "processed", "combined_v15")
MODEL_DIR = os.path.join("models", "candidates", "fresh_start_v18_recovery")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
INCUMBENT_PT = os.path.join("models", "fresh_start_v17", "best_value_net.pt")

TRACE_GAME = os.path.join(HUMAN_B_DIR, "game_00015.jsonl")
HUMAN_DUP = 6
CHANNELS = 15
VALUE_HORIZON = 10
VALUE_FLOOR = 0.97


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
            "recovery output paths already exist; refusing stale merge: "
            + ", ".join(existing)
        )


def _win_filtered_copy(src_dir, dst_dir, want_result):
    os.makedirs(dst_dir, exist_ok=False)
    kept = dropped = 0
    for src in sorted(glob.glob(os.path.join(src_dir, "game_*.jsonl"))):
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
    for src in sorted(glob.glob(os.path.join(HEUR_DIR, "game_*.jsonl"))):
        shutil.copy(src, os.path.join(MERGED_DIR, os.path.basename(src)))
        n_h += 1
    for src in sorted(glob.glob(os.path.join(DEMO_DIR, "game_*.jsonl"))):
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

    for src in sorted(glob.glob(os.path.join(PROMO_DIR, "game_*.jsonl"))):
        shutil.copy(src, os.path.join(promo_out, os.path.basename(src)))
        n_promo += 1

    dup_dirs = {os.path.normpath(HUMAN_B_DIR), os.path.normpath(HUMAN_CUR_DIR)}
    for src in sorted(glob.glob(
        os.path.join("data", "raw", "human_games", "**", "game_*.jsonl"),
        recursive=True,
    )):
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
        "recipe": "v17 frozen recipe + new human games + decisive checkpoint selection",
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
        "position_encoding_channels": CHANNELS,
        "value_discount_mode": "near_mate",
        "value_horizon": VALUE_HORIZON,
        "value_floor": VALUE_FLOOR,
    }
    manifest["sha256"] = _directory_digest(MERGED_DIR)
    with open(os.path.join(MERGED_DIR, "corpus_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"Merged recovery corpus: {json.dumps(manifest, sort_keys=True)}")


def trace_game_0015():
    """v17 mis-valued these positions near +1; the candidate should not."""
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
    for i, cand, inc in zip(indices, candidate, incumbent):
        log(f"TRACE 0015 record={i} raw_white_value candidate={cand:+.3f} "
            f"v17={inc:+.3f} (informational)")


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n# v18 recovery candidate — started "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for req in (
        HEUR_DIR, DEMO_DIR, HUMAN_B_DIR, HUMAN_CUR_DIR, BF_DIR, WF_DIR,
        PROMO_DIR, REFERENCE_DIR, INCUMBENT_PT,
    ):
        if not os.path.exists(req):
            raise RuntimeError(f"missing input: {req}")
    _require_new_outputs((MERGED_DIR, PROCESSED_DIR, MODEL_DIR))

    merge()
    # DEFAULT thresholds — no overrides. A recipe that cannot pass them is
    # the gate doing its job, not an obstacle (v13/v14/v18 lessons).
    run("pretrain-gate", [
        "tools/pretrain_check.py", MERGED_DIR, "--reference", REFERENCE_DIR,
    ])
    run("process", [
        "src/data_processor.py", "--raw-dir", MERGED_DIR,
        "--output-dir", PROCESSED_DIR, "--seed", "42",
        "--channels", str(CHANNELS),
        "--value-discount-mode", "near_mate",
        "--value-horizon", str(VALUE_HORIZON),
        "--value-floor", str(VALUE_FLOOR),
    ])
    run("train", [
        "src/train.py", "--data-dir", PROCESSED_DIR, "--model-dir", MODEL_DIR,
        "--target", "game_result", "--value-head", "wdl",
        "--select-metric", "decisive",
        "--epochs", "30", "--seed", "42",
    ])
    trace_game_0015()

    # Cheap offline gate BEFORE any match: the candidate must not regress
    # vs the incumbent on decisive metrics over shared held-out positions.
    run("model-diff-gate", [
        "tools/model_diff.py", "--candidate", MODEL_PT,
        "--incumbent", INCUMBENT_PT, "--data-dir", PROCESSED_DIR,
        "--enforce",
    ])

    run("anchor-20", [
        "tools/match.py", "--model-a", MODEL_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    run("match-vs-v17", [
        "tools/match.py", "--model-a", MODEL_PT, "--model-b", INCUMBENT_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    log("ALL STEPS COMPLETE — candidate awaits owner play "
        "(models/candidates/fresh_start_v18_recovery)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"CHAIN ABORTED: {exc!r}")
        raise
