"""Depth or late sampling: which makes training self-play so Black-heavy? (owner, 2026-10-10)

The exploration test (tools/exploration_test_20261009.py) ruled out exploration
length: gen55's training-style self-play stays 17-19% White whether the opening
explores for 30, 16 or 8 half-moves, against 39% in match self-play. The two
remaining differences from match play are search depth (1,600 vs 3,200
simulations) and the temperature after the opening (0.1 vs 0). Arms, 300
games each, gen55, 16 exploration half-moves:

    sims3200_t0.1  depth only
    sims1600_t0    top move only
    sims3200_t0    both (match-like after the opening)

Baseline: the exploration test's plies_16 arm (1,600 simulations, 0.1): 18.7%.
Evidence: benchmarks/depth_temperature_test_20261010/. Produces no model.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from exploration_test_20261009 import MODEL, GAMES, tally  # noqa: E402

OUT = ROOT / "benchmarks/depth_temperature_test_20261010"
ARMS = (("sims3200_t0.1", 3200, None), ("sims1600_t0", 1600, 0.0), ("sims3200_t0", 3200, 0.0))
SEED = 2_710_000_000


def recipe(sims, late, i):
    r = dict(model=MODEL, free_games=GAMES, fresh_games=0, league_games=0, fork_games=0,
             sims=sims, fork_sims=sims, workers=8, seed=SEED + i * 1_000_000,
             coverage_reanalysis=False, temperature_plies=16, prefix_models=[], opponents=[])
    if late is not None:
        r["late_temperature"] = late
    return r


def main():
    os.chdir(ROOT)
    baseline = json.loads((ROOT / "benchmarks/exploration_test_20261009/report.json").read_text())["16"]
    report = {"baseline_sims1600_t0.1": baseline}
    for i, (name, sims, late) in enumerate(ARMS):
        arm = OUT / name
        arm.mkdir(parents=True, exist_ok=True)
        config, summary = arm / "recipe.json", arm / "summary.json"
        config.write_text(json.dumps(recipe(sims, late, i), indent=2))
        if not summary.exists():
            print(f"DEPTH/TEMP TEST {name}", flush=True)
            subprocess.run([sys.executable, "-u", "tools/stateful_generation.py", "--config", str(config),
                            "--raw", str(arm / "raw"), "--summary", str(summary)], check=True)
        report[name] = tally(arm / "raw")
        print(f"DEPTH/TEMP RESULT {name}: {report[name]}", flush=True)
        (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("DEPTH/TEMP TEST COMPLETE", flush=True)


if __name__ == "__main__":
    main()
