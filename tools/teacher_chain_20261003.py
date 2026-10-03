"""Teacher-selection chain (docs/plans/TEACHER_SELECTION_PLAN.md): three depth
round robins in order, then the pre-declared selection. Resumable; rerun to continue.

    py -3 -B tools/runs.py start --name teacher_selection py -3 -u tools/teacher_chain_20261003.py
"""
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STEPS = [["tools/teacher_rr.py", "--sims", "1600", "--games", "100"],
         ["tools/teacher_rr.py", "--sims", "6400", "--games", "100"],
         ["tools/teacher_rr.py", "--sims", "12800", "--games", "60"],
         ["tools/teacher_select.py"]]


def main():
    for cmd in STEPS:
        print(f"TEACHER STEP ({time.strftime('%H:%M')}): {' '.join(cmd)}", flush=True)
        code = subprocess.call([sys.executable, "-u", *cmd], cwd=ROOT)
        if code:
            print(f"TEACHER CHAIN STOPPED: {' '.join(cmd)} exited with {code}", flush=True)
            sys.exit(code)
    print("TEACHER CHAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
