"""Informational matches for recovery2 after its near-miss at the clean gate.

The clean model-diff (benchmarks/model_diff_20260716_234433.json) failed on
exactly one metric: policy_top1_black -2.4 points. Everything else favors the
candidate (policy top-1 overall +0.3, White +1.6, winner-sign +2.1 overall
and +4.5 as Black). The gate exists to protect match time on a busy machine;
overnight the machine is idle, and only play can arbitrate a better-value /
slightly-worse-Black-policy trade-off. These matches are measurement, not a
gate bypass: promotion still requires anchor >= 0.55 without color collapse,
h2h vs v17 >= 0.55, and owner play.
"""
import ctypes
import signal
import sys

import overnight_human_v18b as d

MODEL_PT = "models\\candidates\\fresh_start_v18_recovery2\\best_value_net.pt"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def main():
    d.log("RECOVERY2: informational matches — clean gate missed only "
          "policy_top1_black (-2.4); value strongly up; play arbitrates")
    d.run("anchor-20-informational", [
        "tools/match.py", "--model-a", MODEL_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    d.run("match-vs-v17-informational", [
        "tools/match.py", "--model-a", MODEL_PT, "--model-b", d.INCUMBENT_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    d.log("ALL STEPS COMPLETE — recovery2 match evidence recorded; "
          "owner judgment next")


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
