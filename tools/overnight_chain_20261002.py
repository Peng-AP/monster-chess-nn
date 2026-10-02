"""Overnight chain for October 2, 2026 (docs/plans/OVERNIGHT_20261002_PLAN.md).

Waits for gen52 Arm R, then runs, one at a time:
  1. tools/top_round_robin.py
  2. tools/gen52_variant_campaign.py --variant lr   if Arm R's verdict is labels_help
                                       --variant l2   otherwise
Every step is resumable, so rerunning this script after an interruption picks
up where it stopped. Launch it as a managed run:

    py -3 -B tools/runs.py start --name overnight_chain py -3 -u tools/overnight_chain_20261002.py
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RAMP = ROOT / "benchmarks/gen52_program/gen52_ramp_20261001/production"


def ramp_running():
    out = subprocess.run([sys.executable, "-B", "tools/runs.py", "status"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    return any(line.split()[:2] == ["ramp_production", "RUNNING"] for line in out.splitlines())


def step(label, args):
    print(f"CHAIN STEP {label}: {' '.join(args)} ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    code = subprocess.call([sys.executable, "-u", *args], cwd=ROOT)
    if code:
        print(f"CHAIN STOPPED: {label} exited with {code}", flush=True)
        sys.exit(code)


def main():
    while ramp_running():
        time.sleep(60)
    status = json.loads((RAMP / "status.json").read_text()) if (RAMP / "status.json").exists() else {}
    if status.get("status") != "complete":
        print(f"CHAIN STOPPED: Arm R did not complete ({status})", flush=True)
        sys.exit(1)
    call = json.loads((RAMP / "summary.json").read_text())["verdict"]["call"]
    variant = "lr" if call == "labels_help" else "l2"
    print(f"CHAIN: Arm R verdict {call} -> variant {variant}", flush=True)
    step("top_round_robin", ["tools/top_round_robin.py"])
    step(f"variant_{variant}", ["tools/gen52_variant_campaign.py", "--variant", variant])
    print("CHAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
