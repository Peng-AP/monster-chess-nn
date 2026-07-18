"""v18 recovery candidate 2: the actually-frozen v17 recipe.

Recovery 1 (2026-07-16) failed the model-diff gate at policy top-1 -4.7 vs
v17. Diagnosis (benchmarks/model_diff_20260716_*.json):
  - CONFIRMED: the v18-era human-AI policy masking rode silently into
    processing (9.4% masked vs v17's 1.7%), deleting ~10k augmented policy
    targets from the 6x-duplicated human slice. v17 never trained with it.
  - Quantified: ~2.6 of the 4.7 top-1 points were leakage inflation — 31/149
    of the gate set's games were in v17's TRAIN split. On the 109-game
    leakage-clean intersection (data/raw/eval_clean_v13v15) v17 is 29.4%,
    recovery-1 27.7%, and recovery-1's VALUE head already beats v17.

This run reuses the identical raw corpus, reprocessed with
--no-human-ai-mask (masked count 2340 == v17's exactly). The enforced gate
runs on the leakage-clean set — a bias fix, not a threshold change; the
biased standard-set diff is logged for comparability.

Runs detached (WMI, hidden console, keep-awake, SIGINT-immune).
"""
import ctypes
import signal
import sys

import overnight_human_v18b as d

PROCESSED_DIR = "data\\processed\\combined_v15_hpol"
CLEAN_EVAL_DIR = "data\\processed\\eval_clean_v13v15"
MODEL_DIR = "models\\candidates\\fresh_start_v18_recovery2"
MODEL_PT = MODEL_DIR + "\\best_value_net.pt"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def main():
    d.log("RECOVERY2: v17-faithful policy weights (mask 2340 == v17), "
          "gate on leakage-clean eval set")
    d.run("train-recovery2", [
        "src/train.py", "--data-dir", PROCESSED_DIR, "--model-dir", MODEL_DIR,
        "--target", "game_result", "--value-head", "wdl",
        "--select-metric", "decisive",
        "--epochs", "30", "--seed", "42",
    ])
    d.MODEL_PT = MODEL_PT  # trace helper reads the candidate path from d
    d.trace_game_0015()
    d.run("model-diff-gate-clean", [
        "tools/model_diff.py", "--candidate", MODEL_PT,
        "--incumbent", d.INCUMBENT_PT, "--data-dir", CLEAN_EVAL_DIR,
        "--split", "all", "--max-positions", "8192", "--enforce",
    ])
    d.run("model-diff-standard-informational", [
        "tools/model_diff.py", "--candidate", MODEL_PT,
        "--incumbent", d.INCUMBENT_PT, "--data-dir", PROCESSED_DIR,
    ])
    d.run("anchor-20", [
        "tools/match.py", "--model-a", MODEL_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    d.run("match-vs-v17", [
        "tools/match.py", "--model-a", MODEL_PT, "--model-b", d.INCUMBENT_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    d.log("ALL STEPS COMPLETE — recovery2 awaits owner play "
          "(models/candidates/fresh_start_v18_recovery2)")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        main()
    except BaseException as exc:
        d.log(f"CHAIN ABORTED: {exc!r}")
        sys.exit(1)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
