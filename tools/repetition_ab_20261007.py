"""Repetition awareness in the search (owner, 2026-10-07): does it help, and does it cost anything?

Runs sequentially (one GPU job at a time), each step resumable:

1. gen54 aware vs gen54 unaware, 400 games at 3,200 (same network: only the search differs).
2. gen54 vs v28, 160 games, the gen54 diagnostic's own seed: compare with its 71.9% (22 Black
   repetition draws, 17 distinct, while ahead on material).
3. gen54 vs B2, the same for its 92.2% (21 Black repetition draws, 4 distinct).
4. v29 aware vs v29 unaware, 400 games: the release under the new engine.

Evidence: benchmarks/repetition_search_20261007/. Nothing here changes a model or a release.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "benchmarks", "repetition_search_20261007")
GEN54 = "models/candidates/bootstrap_main_gen_0054/arena_selected.pt"
V29 = "models/bootstrap_v29/best_value_net.pt"
V28 = "models/bootstrap_v28/best_value_net.pt"
B2 = "models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt"

STEPS = [
    ("ab_gen54", GEN54, GEN54, 400, 4300000000, ["--no-repetition-search-b"]),
    ("gen54_vs_v28", GEN54, V28, 160, 4243000000, []),
    ("gen54_vs_b2", GEN54, B2, 160, 4240000000, []),
    ("ab_v29", V29, V29, 400, 4301000000, ["--no-repetition-search-b"]),
]


def complete(report, games):
    """match.py rewrites its report as a checkpoint during play; done means every game counted."""
    if not os.path.exists(report):
        return False
    with open(report, encoding="utf-8") as fh:
        r = json.load(fh)
    return r.get("a_as_white", {}).get("games", 0) + r.get("a_as_black", {}).get("games", 0) == games


def main():
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    for name, a, b, games, seed, extra in STEPS:
        report = os.path.join(OUT, f"{name}.json")
        if complete(report, games):
            print(f"REPETITION AB {name}: done, skipped", flush=True)
            continue
        cmd = [sys.executable, "-u", "tools/match.py", "--model-a", a, "--model-b", b,
               "--games", str(games), "--sims", "3200", "--sims-b", "3200", "--engine", "native",
               "--workers", "8", "--seed", str(seed), "--opening-temp-plies", "16",
               "--game-log", os.path.join(OUT, f"{name}.jsonl"), "--report-path", report,
               "--resume", *extra]
        print(f"REPETITION AB {name}: {' '.join(cmd[2:])}", flush=True)
        subprocess.run(cmd, check=True)
    print("REPETITION AB COMPLETE", flush=True)


if __name__ == "__main__":
    main()
