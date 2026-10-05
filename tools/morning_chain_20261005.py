"""Morning chain, October 5, 2026 (owner: "Work until noon").

Waits for the gen53 release gate, then, one at a time, each only if it can end
by about 11:45: the v27 position probe, the gen53 hole scan (opponent-pool
input), and gen53 joining the top-group round robin (same seeds as Arm LR's).
All evaluation; resumable.

    py -3 -B tools/runs.py start --name morning_chain py -3 -u tools/morning_chain_20261005.py
"""
import datetime as dt
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GEN53 = "models/candidates/bootstrap_main_gen_0053/arena_selected.pt"
CUTOFF = dt.datetime(2026, 10, 5, 11, 45)
STEPS = [("position_probe", 0.6, ["tools/position_probe.py"]),
         ("hole_scan", 2.6, ["tools/hole_scan.py", "--name", "gen53", "--model", GEN53]),
         ("top_rr_gen53", 1.6, ["tools/top_rr_extend.py", "--name", "gen53", "--model", GEN53,
                                "--out", "benchmarks/top_rr_20261005_gen53"])]


def running(name):
    out = subprocess.run([sys.executable, "-B", "tools/runs.py", "status"], cwd=ROOT, capture_output=True, text=True).stdout
    return any(line.split()[:2] == [name, "RUNNING"] for line in out.splitlines())


def main():
    while running("gen53_release_gate"):
        time.sleep(60)
    for label, hours, cmd in STEPS:
        if dt.datetime.now() + dt.timedelta(hours=hours) > CUTOFF:
            print(f"MORNING SKIP {label}: would end after {CUTOFF:%H:%M}", flush=True)
            continue
        print(f"MORNING STEP {label} ({time.strftime('%H:%M')})", flush=True)
        if subprocess.call([sys.executable, "-u", *cmd], cwd=ROOT):
            print(f"MORNING STOPPED: {label}", flush=True)
            sys.exit(1)
    print("MORNING CHAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
