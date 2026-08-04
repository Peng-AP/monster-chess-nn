"""Run the PPC sims curve end to end, one point per simulation count.

A module rather than a shell loop so `tools/runs.py` can own it: one named log,
one pid, and progress visible in `runs.py status`. A shell loop launched with
`nohup` has neither, which is how two copies of this curve ended up racing each
other on 2026-08-04.

    py -3 tools/runs.py start --name native_curve --env MONSTER_ENGINE=native \
        -- py -3 -u tools/curve_runner.py
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join("models", "fresh_start_v19", "best_value_net.pt")
POINTS = [200, 400, 800, 1600, 3200, 6400]


def main():
    label = os.environ.get("CURVE_LABEL", "ppccurve_native")
    white_sims = os.environ.get("CURVE_WHITE_SIMS", "400")
    started = time.time()
    for sims in POINTS:
        print(f"=== black_sims={sims} start {time.strftime('%H:%M:%S')} ===",
              flush=True)
        subprocess.run(
            [sys.executable, "-u",
             os.path.join(ROOT, "tools", "promotion_defense_probe.py"),
             "--mode", "outcomes",
             "--playout-white", MODEL, "--playout-black", MODEL,
             "--white-sims", white_sims, "--black-sims", str(sims),
             "--limit", "100", "--label", f"{label}_s{sims}",
             "--workers", "8"],
            cwd=ROOT, check=False)
        print(f"=== black_sims={sims} done {time.strftime('%H:%M:%S')} "
              f"({(time.time() - started) / 60:.1f}m total) ===", flush=True)
    print(f"=== curve complete in {(time.time() - started) / 60:.1f}m ===",
          flush=True)


if __name__ == "__main__":
    main()
