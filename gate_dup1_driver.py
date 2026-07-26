"""The two remaining gate legs for the dup1 control arm.

The anchor already ran (0.65, W 0.90, B 0.40 — benchmark_best_value_net_
20260726_043056.json).  What is missing is the pair that decides the
experiment: vs the incumbent and vs ramp.  Both v18 arms died on the vs-ramp
BLACK leg at 0.30, so that number is the whole question.

Protocol is the v18 driver's verbatim (git show 2d9bb1d:v18_driver.py):
20 games, 400 sims, seed 42, per-side floor 0.40, aggregate must beat 0.50.
Thresholds are fixed here and are never tuned to let an arm through.
"""
import ctypes
import datetime
import glob
import json
import os
import signal
import subprocess
import sys
import time

CAND = "models/candidates/v18_dup1/best_value_net.pt"
INCUMBENT = "models/fresh_start_v17/best_value_net.pt"
RAMP = "models/rejected/fresh_start_v18_ramp/best_value_net.pt"
GAMES, SIMS, SEED = 20, 400, 42
SIDE_FLOOR = 0.40
LOG = "gate_dup1.log"
BENCH = "benchmarks"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def newest(prefix, after):
    """Newest benchmarks/<prefix>*.json written after `after` (mtime floor:
    both legs write similar names and a stale file must never be read)."""
    best, best_m = None, after
    for path in glob.glob(os.path.join(BENCH, prefix + "*.json")):
        m = os.path.getmtime(path)
        if m > best_m:
            best, best_m = path, m
    return best


def read_match(path):
    if not path:
        return {"score": None, "white": None, "black": None, "path": None}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    # match.py's schema: a_score / a_as_white / a_as_black (NOT the
    # benchmark.py schema — they differ, and reading the wrong keys would
    # silently report None and fail every leg).
    return {"score": d.get("a_score"),
            "white": (d.get("a_as_white") or {}).get("score"),
            "black": (d.get("a_as_black") or {}).get("score"),
            "path": os.path.relpath(path)}


def leg(label, opponent):
    prefix = f"match_v18_dup1_vs_{os.path.basename(os.path.dirname(opponent))}"
    t0 = time.time()
    cmd = [sys.executable, "-u", "tools/match.py",
           "--model-a", CAND, "--model-b", opponent,
           "--games", str(GAMES), "--sims", str(SIMS), "--seed", str(SEED)]
    log(f"--- {label}: {' '.join(cmd[2:])}")
    rc = subprocess.run(cmd).returncode
    res = read_match(newest(prefix, t0))
    log(f"    {label}: rc={rc} {res} ({(time.time() - t0) / 60:.1f} min)")
    return res


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    log("=== dup1 gate: vs_ramp then vs_v17 ===")
    out = {"anchor": {"score": 0.65, "white": 0.90, "black": 0.40,
                      "path": "benchmarks/benchmark_best_value_net_20260726_043056.json"}}
    out["vs_ramp"] = leg("vs_ramp", RAMP)      # the deciding leg first
    out["vs_v17"] = leg("vs_v17", INCUMBENT)

    failures = []
    for label in ("vs_ramp", "vs_v17"):
        m = out[label]
        if m["score"] is None or m["score"] <= 0.50:
            failures.append(f"{label} aggregate {m['score']} <= 0.50")
        for side in ("white", "black"):
            if m[side] is None or m[side] < SIDE_FLOOR:
                failures.append(f"{label} {side} leg {m[side]} < floor {SIDE_FLOOR}")
    out["failures"] = failures
    out["verdict"] = "PASS" if not failures else "FAIL"
    with open("gate_dup1_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"VERDICT: {out['verdict']} {failures if failures else ''}")
    log("=== gate complete (a PASS is automated evidence only — owner gate decides) ===")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:                      # noqa: BLE001
        log(f"CHAIN ABORTED: {type(exc).__name__}: {exc}")
        raise
    sys.exit(code)
