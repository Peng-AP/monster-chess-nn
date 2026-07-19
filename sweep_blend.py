"""Post-training blend-weight sweep on the hybrid checkpoint.

Engine strength tracked the ramp share monotonically (WDL-only anchor 0.30,
w=0.3 hybrid 0.50, pure ramp 0.80), so the interesting region is w in
[0.5, 1.0]. The blend is applied at checkpoint load (MONSTER_HYBRID_W), so
each point costs one 20-game anchor match, no retraining. h2h vs v17 for the
best point. Owner playtest still decides; this only locates the knob.
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

REPORT = os.path.join(ROOT, "HYBRID_REPORT.md")
PY = sys.executable
MODEL_PT = os.path.join("models", "candidates", "fresh_start_v18_hybrid",
                        "best_value_net.pt")
INCUMBENT = os.path.join("models", "fresh_start_v17", "best_value_net.pt")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def match(name, w, model_b=None, games=20):
    out_dir = os.path.join("benchmarks", "hybrid", name)
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ, MONSTER_HYBRID_W=str(w))
    args = [PY, "-u", "tools/match.py", "--model-a", MODEL_PT,
            "--games", str(games), "--sims", "400", "--workers", "6",
            "--out-dir", out_dir]
    if model_b:
        args += ["--model-b", model_b]
    log(f"START {name} (w={w})")
    proc = subprocess.run(args, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        log(f"FAILED {name}:\n```\n{err}\n```")
        raise RuntimeError(name)
    latest = sorted(glob.glob(os.path.join(out_dir, "*.json")),
                    key=os.path.getmtime)[-1]
    with open(latest, encoding="utf-8") as f:
        m = json.load(f)
    scores = {"overall": m["a_score"], "white": m["a_as_white"]["score"],
              "black": m["a_as_black"]["score"]}
    log(f"SCORE {name} (w={w}): {scores}")
    return scores


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n## Blend sweep — {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    results = {}
    for w in (0.5, 0.75, 1.0):
        results[w] = match(f"sweep_anchor_w{int(w * 100)}", w)
    best_w = max(results, key=lambda w: results[w]["overall"])
    log(f"SWEEP RESULTS: { {w: r['overall'] for w, r in results.items()} } "
        f"— best w={best_w}")
    if results[best_w]["overall"] >= 0.55:
        match(f"sweep_h2h_w{int(best_w * 100)}", best_w, model_b=INCUMBENT)
    log("SWEEP COMPLETE — set VALUE_HYBRID_PROGRESS_WEIGHT to the chosen w "
        "before owner play (or export MONSTER_HYBRID_W)")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        main()
    except BaseException as exc:
        log(f"SWEEP ABORTED: {exc!r}")
        sys.exit(1)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
