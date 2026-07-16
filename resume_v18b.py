"""Resume the v18 recovery chain from the train step.

The 2026-07-16 03:17 launch completed merge + pretrain gate + processing,
then died with the Claude Code session ~04:26 during training (session-child
process, v12 lesson). Corpus and processed tensors are intact and gate-passed;
this script reruns training from scratch (clean optimizer state — no warm
start, the controlled-recovery claim stays intact) and continues the original
driver chain unchanged. Holds a keep-awake flag so the machine cannot
idle-sleep mid-chain; releases it on exit.

Launch detached (survives session/app closure):
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create ...
"""
import ctypes
import signal
import sys

import overnight_human_v18b as d

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def main():
    d.log("RESUME: rerunning train after 04:26 session-death; corpus/gate/"
          "process reused from the 03:17 launch")
    d.run("train", [
        "src/train.py", "--data-dir", d.PROCESSED_DIR, "--model-dir", d.MODEL_DIR,
        "--target", "game_result", "--value-head", "wdl",
        "--select-metric", "decisive",
        "--epochs", "30", "--seed", "42",
    ])
    d.trace_game_0015()
    d.run("model-diff-gate", [
        "tools/model_diff.py", "--candidate", d.MODEL_PT,
        "--incumbent", d.INCUMBENT_PT, "--data-dir", d.PROCESSED_DIR,
        "--enforce",
    ])
    d.run("anchor-20", [
        "tools/match.py", "--model-a", d.MODEL_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    d.run("match-vs-v17", [
        "tools/match.py", "--model-a", d.MODEL_PT, "--model-b", d.INCUMBENT_PT,
        "--games", "20", "--sims", "400", "--workers", "6",
    ])
    d.log("ALL STEPS COMPLETE — candidate awaits owner play "
          "(models/candidates/fresh_start_v18_recovery)")


if __name__ == "__main__":
    # 13:12 launch was killed by a console control event (^C / window close,
    # exit 0xC000013A). The wrapper ignores SIGINT so a stray event can't kill
    # the chain silently; a killed child still surfaces as a logged abort.
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
