"""Morning chain, October 3, 2026: evaluation-only follow-ups to Arm LR, ending by about 11:00.

Owner, 02:00: "Make sure it ends roughly around 11am. Do useful stuff up until
that time." Waits for the October 2 chain (Arm LR) to finish, then runs, one
at a time; each step starts only if it can plausibly finish by the cutoff:

  1. value colour audit of Arm LR (minutes)
  2. Arm LR added to the top-group round robin (8 x 100 games, ~1.4 h)
  3. Arm LR depth scaling, seed-matched to v29's ladder (~3.3 h)
  4. Arm R at 3,200 and 12,800 (~2 h)

Every step is resumable; rerunning this script continues where it stopped.
"""
import datetime as dt
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LR = "models/candidates/bootstrap_main_gen_0052_large_ramp/arena_selected.pt"
R = "models/candidates/bootstrap_main_gen_0052_ramp/arena_selected.pt"
CUTOFF = dt.datetime(2026, 10, 3, 11, 30)
STEPS = [  # (label, hours needed, command)
    ("audit_lr", 0.2, ["tools/value_colour_audit.py", "--models", "v29", "--model", f"gen52B=models/candidates/bootstrap_main_gen_0052_pool/arena_selected.pt",
                       "--model", f"gen52R={R}", "--model", f"gen52LR={LR}", "--out", "benchmarks/value_colour_audit_20261003_lr"]),
    ("extend_rr_lr", 1.6, ["tools/top_rr_extend.py", "--name", "gen52LR", "--model", LR]),
    ("depth_lr", 3.5, ["tools/elo_depth_scaling.py", "--name", "gen52LR", "--model", LR,
                       "--out", "benchmarks/elo_depth_scaling_20261003_armLR"]),
    ("depth_r", 2.2, ["tools/elo_depth_scaling.py", "--name", "gen52R", "--model", R, "--depths", "3200,12800",
                      "--out", "benchmarks/elo_depth_scaling_20261003_armR"]),
]


def running(name):
    out = subprocess.run([sys.executable, "-B", "tools/runs.py", "status"], cwd=ROOT, capture_output=True, text=True).stdout
    return any(line.split()[:2] == [name, "RUNNING"] for line in out.splitlines())


def main():
    while running("overnight_chain"):
        time.sleep(60)
    for label, hours, cmd in STEPS:
        if dt.datetime.now() + dt.timedelta(hours=hours) > CUTOFF:
            print(f"MORNING SKIP {label}: would end after {CUTOFF:%H:%M}", flush=True)
            continue
        print(f"MORNING STEP {label} ({time.strftime('%H:%M')}): {' '.join(cmd)}", flush=True)
        code = subprocess.call([sys.executable, "-u", *cmd], cwd=ROOT)
        if code:
            print(f"MORNING STOPPED: {label} exited with {code}", flush=True)
            sys.exit(code)
    print("MORNING CHAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
