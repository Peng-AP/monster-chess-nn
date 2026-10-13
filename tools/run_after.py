"""Wait until a log contains a marker, then run a command (one GPU job at a time).

    py -3 tools/runs.py start --name gen56_rehearsal -- py -3 -u tools/run_after.py \\
        --log logs/colour_baselines.log --marker "COLOUR BASELINES COMPLETE" -- py -3 -u tools/gen56_campaign.py --rehearsal-only
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--marker", required=True)
    ap.add_argument("--poll", type=float, default=120.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("give the command after --")
    log = ROOT / args.log
    print(f"RUN AFTER: waiting for {args.marker!r} in {args.log}", flush=True)
    while not (log.exists() and args.marker in log.read_text(errors="ignore")):
        time.sleep(args.poll)
    print(f"RUN AFTER: starting {' '.join(command)}", flush=True)
    sys.exit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
