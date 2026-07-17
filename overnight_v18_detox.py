"""v18 candidate 3 "detox": v17's recipe with the poison removed.

Owner directive 2026-07-17: "start something overnight... without any of the
poisoning components. Even if it's just a copy of what v17 has, or maybe just
a bit more on top of it."

Copy of v17: same frozen base (heuristic_v8, win-filtered mate_demos_v7),
same focus recipe (starts from owner games, heuristic generation @400,
win-filtered at merge), same promo_races_v17 slice, same 6x human
duplication, 15 planes, WDL + near-mate (10 / 0.97).

The poison removed: human-game AI moves and losing-human moves never teach
policy (mask_human_ai default in processing). Every game the owner wins adds
AI-failure demonstrations; unmasked they taught each generation its
predecessor's losing play (the post-v16 echo — recovery-2 froze, masked
recovery-1 attacked).

The bit more on top:
  - focus games REGENERATED tonight: the generator engine now carries the
    m1 king-safety guard, the engine-wide king-safety override, and the soft
    oscillation penalty, so generated games contain fewer blunder/shuffle
    plies than the v16-era focus dirs;
  - focus starts drawn from ALL owner games including 00015+ (the pawn-run
    losses) — restart attempts at defending pawn runs, win-filtered so only
    successful defenses enter;
  - all current human games included (value signal from every position;
    policy only from winning-side moves).

Gates: pretrain check at DEFAULT thresholds; leakage-clean model-diff vs v17
built from games held out by BOTH corpora, enforced before matches;
anchor 20 + h2h 20. Candidate lands in models/candidates/ — promotion still
requires the owner.

Runs detached (WMI, hidden console, keep-awake, SIGINT-immune).
"""
import ctypes
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

REPORT = os.path.join(ROOT, "OVERNIGHT_V18D_REPORT.md")
PY = sys.executable

DEMO_DIR = os.path.join("data", "raw", "mate_demos_v7")
HEUR_DIR = os.path.join("data", "raw", "heuristic_v8")
HUMAN_B_DIR = os.path.join("data", "raw", "human_games", "black_2026_07")
HUMAN_CUR_DIR = os.path.join("data", "raw", "human_games", "curriculum_2026_07")
PROBE_B_DIR = os.path.join("data", "raw", "probe_human_v2")
PROBE_W_DIR = os.path.join("data", "raw", "probe_whitefocus")
PROMO_SRC = os.path.join("data", "raw", "promo_races_v17")

BF_STARTS = os.path.join("data", "start_fens", "human_bf_starts_v6.jsonl")
WF_STARTS = os.path.join("data", "start_fens", "human_wf_starts_v5.jsonl")
BF_DIR = os.path.join("data", "raw", "human_blackfocus_v18d")
WF_DIR = os.path.join("data", "raw", "v18d_whitefocus")
MERGED_DIR = os.path.join("data", "raw", "combined_v16")
REFERENCE_DIR = os.path.join("data", "raw", "combined_v13")  # incumbent's corpus
PROCESSED_DIR = os.path.join("data", "processed", "combined_v16")
CLEAN_RAW_DIR = os.path.join("data", "raw", "eval_clean_v13v16")
CLEAN_PROCESSED_DIR = os.path.join("data", "processed", "eval_clean_v13v16")
MODEL_DIR = os.path.join("models", "candidates", "fresh_start_v18_detox")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
INCUMBENT_PT = os.path.join("models", "fresh_start_v17", "best_value_net.pt")

TRACE_GAME = os.path.join(HUMAN_B_DIR, "game_00015.jsonl")
HUMAN_DUP = 6
CHANNELS = 15
VALUE_HORIZON = 10
VALUE_FLOOR = 0.97

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


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
            "detox output paths already exist; refusing stale reuse: "
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

    for src in sorted(glob.glob(os.path.join(PROMO_SRC, "game_*.jsonl"))):
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
        "recipe": "v17 detox: frozen recipe, masked human-AI policy, "
                  "guard-regenerated focus, all current human games",
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
        "human_ai_policy_masked": True,
        "generator_guards": ["king_safety", "white_first_half", "oscillation_penalty"],
        "position_encoding_channels": CHANNELS,
        "value_discount_mode": "near_mate",
        "value_horizon": VALUE_HORIZON,
        "value_floor": VALUE_FLOOR,
    }
    manifest["sha256"] = _directory_digest(MERGED_DIR)
    with open(os.path.join(MERGED_DIR, "corpus_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"Merged detox corpus: {json.dumps(manifest, sort_keys=True)}")


def build_clean_eval():
    """Games held out (test) by BOTH the incumbent's split and tonight's."""
    def test_ids(processed):
        with open(os.path.join(processed, "split_game_ids.json"), encoding="utf-8") as f:
            return set(json.load(f)["test"])

    clean = sorted(test_ids(os.path.join("data", "processed", "combined_v13"))
                   & test_ids(PROCESSED_DIR))
    os.makedirs(CLEAN_RAW_DIR, exist_ok=False)
    copied = 0
    for gid in clean:
        rel = gid.replace("/", os.sep)
        src = os.path.join(MERGED_DIR, rel + ".jsonl")
        if not os.path.exists(src):
            src = os.path.join(MERGED_DIR, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(CLEAN_RAW_DIR, os.path.relpath(src, MERGED_DIR))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        copied += 1
    log(f"Clean eval set: {copied}/{len(clean)} games held out by both splits")
    if copied < 40:
        raise RuntimeError(f"clean eval set too small ({copied} games)")


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
    for i, cand, inc in zip(indices, candidate, incumbent):
        log(f"TRACE 0015 record={i} raw_white_value candidate={cand:+.3f} "
            f"v17={inc:+.3f} (informational)")


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n# DETOX: v18 candidate 3 — started "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for req in (
        HEUR_DIR, DEMO_DIR, HUMAN_B_DIR, HUMAN_CUR_DIR, PROBE_B_DIR,
        PROBE_W_DIR, PROMO_SRC, REFERENCE_DIR, INCUMBENT_PT,
        os.path.join("data", "processed", "combined_v13", "split_game_ids.json"),
    ):
        if not os.path.exists(req):
            raise RuntimeError(f"missing input: {req}")
    _require_new_outputs((
        BF_STARTS, WF_STARTS, BF_DIR, WF_DIR, MERGED_DIR, PROCESSED_DIR,
        CLEAN_RAW_DIR, CLEAN_PROCESSED_DIR, MODEL_DIR,
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
        "--record-all-plies", "--seed", "1702", "--output-dir", BF_DIR,
    ])
    run("gen-whitefocus", [
        "src/data_generation.py", "--num-games", "250", "--simulations", "400",
        "--start-fen-file", WF_STARTS, "--start-fen-side", "white",
        "--record-all-plies", "--seed", "1703", "--output-dir", WF_DIR,
    ])

    merge()
    # DEFAULT thresholds — a recipe that cannot pass them is the gate working.
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

    build_clean_eval()
    run("process-clean-eval", [
        "src/data_processor.py", "--raw-dir", CLEAN_RAW_DIR,
        "--output-dir", CLEAN_PROCESSED_DIR, "--seed", "42",
        "--channels", str(CHANNELS),
        "--value-discount-mode", "near_mate",
        "--value-horizon", str(VALUE_HORIZON),
        "--value-floor", str(VALUE_FLOOR),
    ])
    run("model-diff-gate-clean", [
        "tools/model_diff.py", "--candidate", MODEL_PT,
        "--incumbent", INCUMBENT_PT, "--data-dir", CLEAN_PROCESSED_DIR,
        "--split", "all", "--max-positions", "8192", "--enforce",
    ])
    run("model-diff-standard-informational", [
        "tools/model_diff.py", "--candidate", MODEL_PT,
        "--incumbent", INCUMBENT_PT, "--data-dir", PROCESSED_DIR,
    ])

    run("anchor-20", [
        "tools/match.py", "--model-a", MODEL_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    run("match-vs-v17", [
        "tools/match.py", "--model-a", MODEL_PT, "--model-b", INCUMBENT_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    log("ALL STEPS COMPLETE — detox candidate awaits owner play "
        "(models/candidates/fresh_start_v18_detox)")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        main()
    except BaseException as exc:
        log(f"CHAIN ABORTED: {exc!r}")
        sys.exit(1)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
