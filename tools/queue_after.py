"""Wait for a named run to finish, then run a command in its place.

The GPU fits one search job at a time, so queued work has to wait rather than
start alongside. This blocks on an existing `tools/runs.py` run by name, then
execs the next command in the same process — so the whole chain sits under one
`runs.py` entry, with one log and one pid that `status` reports honestly.

    py -3 tools/runs.py start --name next -- \
        py -3 tools/queue_after.py --after gen5_reanalysis -- py -3 tools/foo.py

If the awaited run is already finished, the command starts immediately.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from runs import _alive, _started_epoch  # noqa: E402

POLL_SECONDS = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", required=True,
                    help="name of the runs.py run to wait for")
    ap.add_argument("--poll", type=float, default=POLL_SECONDS)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("nothing to run: put the command after --")

    meta_path = os.path.join(ROOT, "logs", f"{args.after}.json")
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        raise SystemExit(f"no run named {args.after} (looked in {meta_path})")

    pid, started = meta.get("pid"), _started_epoch(meta.get("started"))
    waited = 0.0
    # The pid check is creation-time qualified, so a recycled pid cannot make
    # this wait forever on an unrelated process.
    while _alive(pid, started_epoch=started):
        if waited % 600 == 0:
            print(f"waiting for {args.after} (pid {pid}) — "
                  f"{waited / 60:.0f}m so far", flush=True)
        time.sleep(args.poll)
        waited += args.poll

    print(f"{args.after} finished after {waited / 60:.0f}m of waiting; "
          f"starting: {' '.join(command)}\n", flush=True)
    sys.exit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
