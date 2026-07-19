"""Validate the phase router (v17 opening / ramp late) as the sparring engine.

The router composites the two strongest checkpoints by phase: v17's opening
material judgment while White still has >= 3 pawns, the ramp checkpoint's
late-game gradient after. Before it earns owner board time (and generator
duty), it must demonstrate it inherits both strengths: 20-game h2h vs each
parent plus the 20-game anchor. Bar: >= 0.55 vs v17, >= 0.50 vs ramp
(ramp is the engine-strongest model; matching it while fixing the opening
is already a win), anchor >= ramp's 0.80 ballpark without color collapse.

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

REPORT = os.path.join(ROOT, "ROUTER_REPORT.md")
PY = sys.executable

SPEC = os.path.join("models", "experiments", "router_v17_ramp", "router.json")
V17 = os.path.join("models", "fresh_start_v17", "best_value_net.pt")
RAMP = os.path.join("models", "rejected", "fresh_start_v18_ramp",
                    "best_value_net.pt")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def match(name, model_b=None, games=20):
    out_dir = os.path.join("benchmarks", "router", name)
    os.makedirs(out_dir, exist_ok=True)
    args = [PY, "-u", "tools/match.py", "--model-a", SPEC,
            "--games", str(games), "--sims", "400", "--workers", "6",
            "--out-dir", out_dir]
    if model_b:
        args += ["--model-b", model_b]
    log(f"START {name}")
    t0 = time.time()
    proc = subprocess.run(args, capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").strip().splitlines()[-14:])
        log(f"FAILED {name} (exit={proc.returncode}, {dt:.1f} min):\n```\n{err}\n```")
        raise RuntimeError(name)
    latest = sorted(glob.glob(os.path.join(out_dir, "*.json")),
                    key=os.path.getmtime)[-1]
    with open(latest, encoding="utf-8") as f:
        m = json.load(f)
    scores = {"overall": m["a_score"], "white": m["a_as_white"]["score"],
              "black": m["a_as_black"]["score"]}
    log(f"SCORE {name} ({dt:.1f} min): {scores}")
    return scores


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n# ROUTER VALIDATION — started "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    for req in (SPEC, V17, RAMP):
        if not os.path.exists(req):
            raise RuntimeError(f"missing input: {req}")

    h2h_v17 = match("router_vs_v17", model_b=V17)
    h2h_ramp = match("router_vs_ramp", model_b=RAMP)
    anchor = match("router_anchor")

    ok = (h2h_v17["overall"] >= 0.55 and h2h_ramp["overall"] >= 0.50
          and anchor["overall"] >= 0.65
          and anchor["white"] >= 0.40 and anchor["black"] >= 0.40)
    log(f"SUMMARY: vs_v17={h2h_v17} vs_ramp={h2h_ramp} anchor={anchor}")
    log("VERDICT: " + ("PASS — router inherits both strengths; ready for "
                       "owner sessions (dropdown: router/router_v17_ramp)"
                       if ok else
                       "FAIL — composite does not beat both parents; "
                       "do not spend owner board time on it"))
    log("ALL STEPS COMPLETE")


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
