"""v18 candidate: hybrid value head — WDL sharpness + ramp gradient.

Owner playtest of fresh_start_v18_ramp (2026-07-18): late-game progress
behavior appeared (first attempts at stopping passed pawns) but flat early
values released the heuristic prior's pawn-chucking — two central pawns
gifted in the opening. The composition: value = 0.7 * WDL expectation
(sharp win/loss and material discrimination, the property every WDL model
had in openings) + 0.3 * scalar head trained on the end-anchored ramp
(floor 0.5 / horizon 60 — the gradient that took the same corpus from
anchor 0.30 to 0.80). Blend weight lives in config
VALUE_HYBRID_PROGRESS_WEIGHT and is applied at checkpoint load.

Everything else frozen: combined_v16 corpus (gate-passed), 15 planes,
ramp-processed tensors from the marathon, decisive selection, engine guards.

Chain: train -> trace -> clean diff (informational) -> anchor 20 ->
h2h vs v17 20 -> auto promotion evidence (anchor 60) if anchor >= 0.55
without color collapse. Owner playtest remains the final gate.

Runs detached (WMI, hidden console, keep-awake, SIGINT-immune).
"""
import ctypes
import glob
import json
import os
import signal
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

REPORT = os.path.join(ROOT, "HYBRID_REPORT.md")
PY = sys.executable

RAMP_PROCESSED = os.path.join("data", "processed", "combined_v16_r50h60")
CLEAN_EVAL = os.path.join("data", "processed", "eval_clean_v13v16")
MODEL_DIR = os.path.join("models", "candidates", "fresh_start_v18_hybrid")
MODEL_PT = os.path.join(MODEL_DIR, "best_value_net.pt")
INCUMBENT = os.path.join("models", "fresh_start_v17", "best_value_net.pt")
TRACE_GAME = os.path.join("data", "raw", "human_games", "black_2026_07",
                          "game_00015.jsonl")
MATCH_ROOT = os.path.join("benchmarks", "hybrid")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args):
    log(f"START {name}: {' '.join(args)}")
    t0 = time.time()
    proc = subprocess.run([PY, "-u"] + args, capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-22:])
    log(f"END {name} (exit={proc.returncode}, {dt:.1f} min)\n```\n{tail}\n```")
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").strip().splitlines()[-18:])
        log(f"STDERR {name}:\n```\n{err}\n```")
        raise RuntimeError(f"{name} failed (exit {proc.returncode})")
    return proc.stdout or ""


def match(name, model_a, model_b=None, games=20, sims=400):
    out_dir = os.path.join(MATCH_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    args = ["tools/match.py", "--model-a", model_a, "--games", str(games),
            "--sims", str(sims), "--workers", "6", "--out-dir", out_dir]
    if model_b:
        args += ["--model-b", model_b]
    run(f"match-{name}", args)
    paths = sorted(glob.glob(os.path.join(out_dir, "*.json")),
                   key=os.path.getmtime)
    with open(paths[-1], encoding="utf-8") as f:
        m = json.load(f)
    scores = {"overall": m["a_score"],
              "white": m["a_as_white"]["score"],
              "black": m["a_as_black"]["score"]}
    log(f"SCORE {name}: {scores}")
    return scores


def trace():
    from evaluation import NNEvaluator
    from monster_chess import MonsterChessGame
    with open(TRACE_GAME, encoding="utf-8") as f:
        records = [json.loads(ln) for ln in f if ln.strip()]
    states = [MonsterChessGame(records[i]["fen"]) for i in (14, 22, 24)]
    vals = NNEvaluator(MODEL_PT).batch_evaluate(states)
    log("TRACE 0015 hybrid: " + " ".join(f"{v:+.3f}" for v in vals)
        + " (v17: +0.993 +0.985 +0.942; positions are lost for White)")


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n# HYBRID — started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    for req in (RAMP_PROCESSED, CLEAN_EVAL, INCUMBENT):
        if not os.path.exists(req):
            raise RuntimeError(f"missing input: {req}")
    if os.path.exists(MODEL_DIR):
        raise RuntimeError(f"output already exists: {MODEL_DIR}")

    from config import VALUE_HYBRID_PROGRESS_WEIGHT
    log(f"hybrid inference blend weight: {VALUE_HYBRID_PROGRESS_WEIGHT} "
        "(scalar head trained on end-anchored ramp 0.5/60)")
    run("train-hybrid", [
        "src/train.py", "--data-dir", RAMP_PROCESSED, "--model-dir", MODEL_DIR,
        "--target", "game_result", "--value-head", "hybrid",
        "--select-metric", "decisive", "--epochs", "30", "--seed", "42",
    ])
    trace()
    run("diff-clean-informational", [
        "tools/model_diff.py", "--candidate", MODEL_PT,
        "--incumbent", INCUMBENT, "--data-dir", CLEAN_EVAL,
        "--split", "all", "--max-positions", "8192",
    ])
    anchor = match("hybrid_anchor", MODEL_PT)
    h2h = match("hybrid_h2h", MODEL_PT, INCUMBENT)
    if (anchor["overall"] >= 0.55 and anchor["white"] >= 0.40
            and anchor["black"] >= 0.40 and h2h["overall"] >= 0.55):
        log("EVIDENCE: both automated legs green — gathering 60-game anchor")
        match("hybrid_anchor60", MODEL_PT, games=60)
    log("ALL STEPS COMPLETE — hybrid candidate awaits owner play "
        "(models/candidates/fresh_start_v18_hybrid)")


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
