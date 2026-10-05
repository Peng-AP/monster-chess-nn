"""After gen54 completes: gate v4 vs the release (v29), the v27-position probe, the value audit.

Same follow-ups gen53 received (tools/gen53_release_gate_chain.py), so the two
generations are compared on identical instruments. The gate first tries to
reuse v29's par measured on October 5 for gen53's release gate; if the runtime
identity check rejects it, v29's par is measured afresh. Evaluation only.

    py -3 -B tools/runs.py start --name gen54_followup py -3 -u tools/gen54_followup_chain.py
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GEN54 = ROOT / "benchmarks/gen54_program/gen54_20261005/production"
OUT = ROOT / "benchmarks/gen54_program/gen54_20261005/vs_release"
BAR = "models/candidates/bootstrap_main_gen_0051_deepvalue/arena_selected.pt"   # = models/bootstrap_v29 (same SHA256)
PAR = ROOT / "benchmarks/gen53_program/gen53_20260928/vs_release/gate_fresh_par"
SEED = 4_110_000_000   # gate uses seed, +200,000, +400,000; clear of gen53's 4.10e9 block


def running(name):
    out = subprocess.run([sys.executable, "-B", "tools/runs.py", "status"], cwd=ROOT, capture_output=True, text=True).stdout
    return any(line.split()[:2] == [name, "RUNNING"] for line in out.splitlines())


def run(label, cmd):
    print(f"FOLLOWUP STEP {label} ({time.strftime('%H:%M')}): {' '.join(cmd)}", flush=True)
    return subprocess.call([sys.executable, "-u", *cmd], cwd=ROOT)


def main():
    while running("gen54_production"):
        time.sleep(60)
    status_path = GEN54 / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "missing"}
    if status.get("status") != "complete":
        print(f"FOLLOWUP STOPPED: gen54 did not complete ({status})", flush=True)
        sys.exit(1)
    nominee = json.loads((GEN54 / "selection/a_nominee.json").read_text())["path"]
    gate = ["tools/gate_depth.py", "--model", nominee, "--bar-model", BAR, "--seed", str(SEED)]
    if run("gate_vs_v29", gate + ["--run-dir", str(OUT / "gate"), "--par-dir", str(PAR)]):
        print("FOLLOWUP NOTE: reused par rejected; measuring v29's par afresh", flush=True)
        if run("gate_vs_v29_fresh_par", gate + ["--run-dir", str(OUT / "gate_fresh_par")]):
            print("FOLLOWUP STOPPED: gate failed to run", flush=True)
            sys.exit(1)
    run("position_probe", ["tools/position_probe.py", "--candidate", f"gen54={nominee}",
                           "--out", str(ROOT / "benchmarks/position_probe_gen54")])
    run("audit", ["tools/value_colour_audit.py", "--models", "v29",
                  "--model", "gen53=models/candidates/bootstrap_main_gen_0053/arena_selected.pt",
                  "--model", f"gen54={nominee}", "--out", "benchmarks/value_colour_audit_gen54"])
    print("FOLLOWUP CHAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
