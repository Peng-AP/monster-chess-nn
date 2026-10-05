"""After gen53 completes: gate v4 of the gen53 nominee against the release (v29).

Owner, October 4: "Sure" (chain the release gate after gen53's diagnostics).
PROMOTION_RULE.md requires gate v4 against the release plus held-out
non-regression; gen53's own gate was against its teacher (Arm R). The bar is
v29's byte-identical file (gen51 deep-value nominee, SHA256 dbf26b9e...), so
gen52's measured self-par of that file is reused, as gen52 Arm C's gate did.
Then the value-colour audit of the nominee. Evaluation only; nothing promoted.

    py -3 -B tools/runs.py start --name gen53_release_gate py -3 -u tools/gen53_release_gate_chain.py
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GEN53 = ROOT / "benchmarks/gen53_program/gen53_20260928/production"
OUT = ROOT / "benchmarks/gen53_program/gen53_20260928/vs_release"
BAR = "models/candidates/bootstrap_main_gen_0051_deepvalue/arena_selected.pt"   # = models/bootstrap_v29 (same SHA256)
PAR = ROOT / "benchmarks/gen52_program/gen52_20260927/production/par"
SEED = 4_100_000_000   # gate uses seed, +200,000 and +400,000; clear of every earlier block, below 2**32


def running(name):
    out = subprocess.run([sys.executable, "-B", "tools/runs.py", "status"], cwd=ROOT, capture_output=True, text=True).stdout
    return any(line.split()[:2] == [name, "RUNNING"] for line in out.splitlines())


def run(label, cmd):
    print(f"RELEASE STEP {label} ({time.strftime('%H:%M')}): {' '.join(cmd)}", flush=True)
    return subprocess.call([sys.executable, "-u", *cmd], cwd=ROOT)


def main():
    while running("gen53_production"):
        time.sleep(60)
    status = json.loads((GEN53 / "status.json").read_text())
    if status.get("status") != "complete":
        print(f"RELEASE CHAIN STOPPED: gen53 did not complete ({status})", flush=True)
        sys.exit(1)
    nominee = json.loads((GEN53 / "selection/a_nominee.json").read_text())["path"]
    gate = ["tools/gate_depth.py", "--model", nominee, "--bar-model", BAR, "--seed", str(SEED)]
    code = run("gate_vs_v29", gate + ["--run-dir", str(OUT / "gate"), "--par-dir", str(PAR)])
    if code:
        print("RELEASE NOTE: reused par rejected; measuring v29's par afresh", flush=True)
        code = run("gate_vs_v29_fresh_par", gate + ["--run-dir", str(OUT / "gate_fresh_par")])
        if code:
            print(f"RELEASE CHAIN STOPPED: gate exited with {code}", flush=True)
            sys.exit(code)
    run("audit", ["tools/value_colour_audit.py", "--models", "v29",
                  "--model", "gen52R=models/candidates/bootstrap_main_gen_0052_ramp/arena_selected.pt",
                  "--model", f"gen53={nominee}", "--out", "benchmarks/value_colour_audit_gen53"])
    print("RELEASE CHAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
