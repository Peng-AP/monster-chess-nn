"""Run the per-colour instrument on the current models, one after another (one GPU job).

gen53, gen54, v28 and gen49 against the references gen55 and v29. gen55 and v29
themselves are already measured by the census (phase 1b). Resumable.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("gen54", "models/candidates/bootstrap_main_gen_0054/arena_selected.pt"),
          ("gen53", "models/candidates/bootstrap_main_gen_0053/arena_selected.pt"),
          ("v28", "models/bootstrap_v28/best_value_net.pt"),
          ("gen49", "models/candidates/bootstrap_main_gen_0049/arena_selected.pt")]

for name, path in MODELS:
    subprocess.run([sys.executable, "-u", "tools/colour_instrument.py", "--name", name, "--model", path],
                   check=True, cwd=ROOT)
print("COLOUR BASELINES COMPLETE", flush=True)
