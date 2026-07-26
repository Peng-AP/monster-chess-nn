"""Train the 1x-human-duplication control arm. TRAINING ONLY — no matches.

The v18 GAP arm's command verbatim (git show 2d9bb1d:v18_driver.py) with one
thing changed: --data-dir points at combined_v17_dup1_r50h60 instead of
combined_v17_r50h60.  Same 1,175 train games, same 80 human games, same split,
same seed, same ramp labels — only the human duplication multiple differs
(6x -> 1x).  The question it answers: did duplication cost Black the 0.30 leg
that killed both v18 arms?

Gate matches are deliberately NOT run here (owner's call 2026-07-25) — they eat
six workers.  When the checkpoint lands, run the three legs separately.
"""
import ctypes
import datetime
import signal
import subprocess
import sys
import time

DATA = "data/processed/combined_v17_dup1_r50h60"
MODEL_DIR = "models/candidates/v18_dup1"
SEED = 42
LOG = "dup1_driver.log"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    cmd = [sys.executable, "-u", "src/train.py",
           "--data-dir", DATA, "--model-dir", MODEL_DIR,
           "--target", "game_result", "--value-head", "scalar",
           "--epochs", "30", "--seed", str(SEED)]
    log("=== dup1 control arm: training only ===")
    log(" ".join(cmd))
    started = time.time()
    # Inherit stdout rather than PIPE: subprocess.run would hold every line
    # until the child exits, leaving a 4.5-hour run unobservable.
    proc = subprocess.run(cmd)
    mins = (time.time() - started) / 60.0
    log(f"train exited rc={proc.returncode} after {mins:.1f} min")
    if proc.returncode != 0:
        log("CHAIN ABORTED")
        return 1
    log(f"=== done. checkpoint in {MODEL_DIR} — gate legs NOT run ===")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:                      # noqa: BLE001
        # sys.exit() inside the try would land here too and log the failure
        # marker on a clean run — call it after, never inside.
        log(f"CHAIN ABORTED: {type(exc).__name__}: {exc}")
        raise
    sys.exit(code)
