"""Sequential sharpening of the ramp checkpoint — dose-response overnight.

The joint-trained hybrid broke Black at every blend weight (sweep 2026-07-19:
Black 0.1-0.2 vs the pure ramp model's 0.5-0.6), so blending is dead. This
tests the sequential alternative: warm-start from the ramp checkpoint
(models/rejected/fresh_start_v18_ramp — engine-strongest model to date,
anchor 0.80/0.75, h2h 0.70, owner-rejected for opening pawn gifts) and
fine-tune BRIEFLY on sharp near-mate targets (10 / 0.97) at a tenth of base
LR. Hypothesis: a light sharpening dose restores opening material judgment
(the WDL-era property) while the ramp-built backbone keeps its late-game
gradient. Doses 1, 3, 8 epochs measure the trade-off curve; too much
sharpening should regress toward detox (anchor 0.30).

Each dose: fine-tune -> trace -> 20-game anchor. Best dose by anchor
(tiebreak: Black score) gets the 20-game h2h vs v17 and, if both promotion
legs are green without color collapse, the 60-game anchor evidence.
Owner playtest remains the final gate.

Runs detached (WMI, hidden console, keep-awake, SIGINT-immune). Waits for
the blend sweep's last match to finish before taking the machine.
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

REPORT = os.path.join(ROOT, "FINETUNE_REPORT.md")
PY = sys.executable

MERGED = os.path.join("data", "raw", "combined_v16")
NEARMATE_PROCESSED = os.path.join("data", "processed", "combined_v16")
CLEAN_EVAL = os.path.join("data", "processed", "eval_clean_v13v16")
RAMP_PT = os.path.join("models", "rejected", "fresh_start_v18_ramp",
                       "best_value_net.pt")
INCUMBENT = os.path.join("models", "fresh_start_v17", "best_value_net.pt")
TRACE_GAME = os.path.join("data", "raw", "human_games", "black_2026_07",
                          "game_00015.jsonl")
SWEEP_H2H_DIR = os.path.join("benchmarks", "hybrid", "sweep_h2h_w50")
DOSES = (1, 3, 8)

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
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-20:])
    log(f"END {name} (exit={proc.returncode}, {dt:.1f} min)\n```\n{tail}\n```")
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").strip().splitlines()[-16:])
        log(f"STDERR {name}:\n```\n{err}\n```")
        raise RuntimeError(f"{name} failed (exit {proc.returncode})")


def match(name, model_a, model_b=None, games=20):
    out_dir = os.path.join("benchmarks", "finetune", name)
    os.makedirs(out_dir, exist_ok=True)
    args = ["tools/match.py", "--model-a", model_a, "--games", str(games),
            "--sims", "400", "--workers", "6", "--out-dir", out_dir]
    if model_b:
        args += ["--model-b", model_b]
    run(f"match-{name}", args)
    latest = sorted(glob.glob(os.path.join(out_dir, "*.json")),
                    key=os.path.getmtime)[-1]
    with open(latest, encoding="utf-8") as f:
        m = json.load(f)
    scores = {"overall": m["a_score"], "white": m["a_as_white"]["score"],
              "black": m["a_as_black"]["score"]}
    log(f"SCORE {name}: {scores}")
    return scores


def trace(name, model_pt):
    from evaluation import NNEvaluator
    from monster_chess import MonsterChessGame
    with open(TRACE_GAME, encoding="utf-8") as f:
        records = [json.loads(ln) for ln in f if ln.strip()]
    states = [MonsterChessGame(records[i]["fen"]) for i in (14, 22, 24)]
    vals = NNEvaluator(model_pt).batch_evaluate(states)
    log(f"TRACE 0015 {name}: " + " ".join(f"{v:+.3f}" for v in vals)
        + " (ramp was saturated ~-1.0; graded negative is the goal)")


def wait_for_sweep():
    deadline = time.time() + 75 * 60
    while time.time() < deadline:
        if glob.glob(os.path.join(SWEEP_H2H_DIR, "*.json")):
            log("sweep h2h finished — machine is free")
            return
        time.sleep(60)
    log("sweep wait timed out after 75 min — proceeding anyway")


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n# FINETUNE — started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    for req in (MERGED, CLEAN_EVAL, RAMP_PT, INCUMBENT):
        if not os.path.exists(req):
            raise RuntimeError(f"missing input: {req}")
    for dose in DOSES:
        d = os.path.join("models", "candidates", f"fresh_start_v18_ft{dose}")
        if os.path.exists(d):
            raise RuntimeError(f"output already exists: {d}")

    wait_for_sweep()
    if not os.path.exists(NEARMATE_PROCESSED):
        run("process-nearmate", [
            "src/data_processor.py", "--raw-dir", MERGED,
            "--output-dir", NEARMATE_PROCESSED, "--seed", "42",
            "--channels", "15", "--value-discount-mode", "near_mate",
            "--value-horizon", "10", "--value-floor", "0.97",
        ])

    results = {}
    for dose in DOSES:
        model_dir = os.path.join("models", "candidates",
                                 f"fresh_start_v18_ft{dose}")
        model_pt = os.path.join(model_dir, "best_value_net.pt")
        run(f"finetune-{dose}ep", [
            "src/train.py", "--data-dir", NEARMATE_PROCESSED,
            "--model-dir", model_dir, "--resume-from", RAMP_PT,
            "--target", "game_result", "--value-head", "scalar",
            "--select-metric", "decisive", "--epochs", str(dose),
            "--lr", "0.0002", "--warmup-epochs", "0", "--seed", "42",
        ])
        trace(f"ft{dose}", model_pt)
        results[dose] = match(f"ft{dose}_anchor", model_pt)

    best = max(DOSES, key=lambda d: (results[d]["overall"], results[d]["black"]))
    best_pt = os.path.join("models", "candidates",
                           f"fresh_start_v18_ft{best}", "best_value_net.pt")
    log(f"DOSE CURVE: { {d: results[d] for d in DOSES} } — best ft{best}")
    run(f"diff-ft{best}-informational", [
        "tools/model_diff.py", "--candidate", best_pt,
        "--incumbent", INCUMBENT, "--data-dir", CLEAN_EVAL,
        "--split", "all", "--max-positions", "8192",
    ])
    h2h = match(f"ft{best}_h2h", best_pt, INCUMBENT)
    a = results[best]
    if (a["overall"] >= 0.55 and a["white"] >= 0.40 and a["black"] >= 0.40
            and h2h["overall"] >= 0.55):
        log("EVIDENCE: both automated legs green — 60-game anchor")
        match(f"ft{best}_anchor60", best_pt, games=60)
    log(f"ALL STEPS COMPLETE — best dose ft{best} awaits owner play "
        f"(models/candidates/fresh_start_v18_ft{best})")


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
