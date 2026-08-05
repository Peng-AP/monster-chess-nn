"""Launch and inspect long runs. One log per run, in the repo, readable.

Long runs were going to a session temp directory, which meant the only way to
answer "is it alive and how far along?" was reading per-process CPU counters.
That is also how a hung job hides. Everything long now lands in `logs/`:

    logs/<name>.log     the run's output, appended live
    logs/<name>.json    metadata: command, pid, start time, status

Usage:

    py -3 tools/runs.py start --name curve -- py -3 tools/foo.py --arg 1
    py -3 tools/runs.py status
    py -3 tools/runs.py tail --name curve

`status` is the one to look at: active and recently completed runs, how long
they have been going, and their most recent progress line. Older entries are
hidden automatically; use ``status --all`` when the history is useful.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
DEFAULT_RECENT_MINUTES = 60
DEFAULT_EXTERNAL_STALE_MINUTES = 30


def _alive(pid):
    if pid is None:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=15).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _last_progress(path, lookback=200):
    """The most recent line that looks like progress, else the last line.

    Carriage returns matter: tqdm-style bars overwrite in place, so a naive
    read of the tail returns one enormous line.
    """
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except Exception:
        return ""
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n") if ln.strip()]
    if not lines:
        return ""
    for line in reversed(lines[-lookback:]):
        if any(tok in line for tok in ("[", "%", "elapsed", "left", "===")):
            return line
    return lines[-1]


def _started_epoch(started):
    try:
        return time.mktime(time.strptime(started, "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return 0


def _last_activity(meta_path, log_path, started_epoch):
    """Best available approximation of when a run last did useful work."""
    candidates = [started_epoch]
    for path in (meta_path, log_path):
        try:
            candidates.append(os.path.getmtime(path))
        except OSError:
            pass
    return max(candidates)


def _run_state(meta, last_activity, now, external_stale_seconds):
    """Classify a record without letting pid-less adopted jobs live forever."""
    pid = meta.get("pid")
    if pid is not None:
        return "RUNNING" if _alive(pid) else "finished"
    if now - last_activity <= external_stale_seconds:
        return "external"
    return "stale"


def cmd_start(args):
    os.makedirs(LOGS, exist_ok=True)
    if not args.command:
        raise SystemExit("nothing to run: put the command after --")
    log_path = os.path.join(LOGS, f"{args.name}.log")
    meta_path = os.path.join(LOGS, f"{args.name}.json")
    log = open(log_path, "a", encoding="utf-8")
    log.write(f"\n=== {args.name} start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log.flush()
    env = dict(os.environ)
    for pair in args.env or []:
        key, _, value = pair.partition("=")
        env[key] = value
    proc = subprocess.Popen(args.command, cwd=ROOT, stdout=log, stderr=log, env=env)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"name": args.name, "pid": proc.pid,
                   "command": " ".join(args.command),
                   "env": args.env or [],
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=2)
    print(f"{args.name}: pid {proc.pid}, logging to logs/{args.name}.log")


def cmd_status(args):
    os.makedirs(LOGS, exist_ok=True)
    metas = sorted(f for f in os.listdir(LOGS) if f.endswith(".json"))
    if not metas:
        print("no runs recorded")
        return
    now = time.time()
    recent_seconds = max(0, args.recent_minutes) * 60
    external_stale_seconds = max(0, args.external_stale_minutes) * 60
    records = []
    for meta_name in metas:
        meta_path = os.path.join(LOGS, meta_name)
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        name = meta.get("name", meta_name[:-5])
        started_epoch = _started_epoch(meta.get("started"))
        log_path = os.path.join(LOGS, f"{name}.log")
        last_activity = _last_activity(meta_path, log_path, started_epoch)
        state = _run_state(meta, last_activity, now, external_stale_seconds)
        elapsed = max(0, now - started_epoch) if started_epoch else 0
        progress = _last_progress(log_path)[:70]
        records.append((name, state, elapsed, last_activity, progress))

    visible = [record for record in records if (
        args.all or record[1] in ("RUNNING", "external")
        or now - record[3] <= recent_seconds
    )]
    hidden = len(records) - len(visible)
    if not visible:
        suffix = (f" ({hidden} older entr{'y' if hidden == 1 else 'ies'} hidden; "
                  "use status --all for history)") if hidden else ""
        print(f"no active or recent runs{suffix}")
        return

    print(f"{'run':22s} {'state':9s} {'elapsed':>9s}  progress")
    print("-" * 100)
    for name, state, elapsed, _last_activity_epoch, progress in visible:
        print(f"{name:22s} {state:9s} {elapsed / 60:8.1f}m  {progress}")
    if hidden:
        print(f"\n{hidden} older entr{'y' if hidden == 1 else 'ies'} hidden; "
              "use status --all for history")


def cmd_tail(args):
    path = os.path.join(LOGS, f"{args.name}.log")
    if not os.path.exists(path):
        raise SystemExit(f"no log at {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read().replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    for line in lines[-args.lines:]:
        print(line)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start", help="launch a long run with its own log")
    start.add_argument("--name", required=True)
    start.add_argument("--env", action="append",
                       help="KEY=VALUE passed to the child (repeatable)")
    start.add_argument("command", nargs=argparse.REMAINDER)
    start.set_defaults(func=cmd_start)

    status = sub.add_parser(
        "status", help="active/recent runs: alive?, elapsed, progress")
    status.add_argument(
        "--all", action="store_true", help="include old finished/stale history")
    status.add_argument(
        "--recent-minutes", type=float, default=DEFAULT_RECENT_MINUTES,
        help=f"show completed runs this long (default: {DEFAULT_RECENT_MINUTES})")
    status.add_argument(
        "--external-stale-minutes", type=float,
        default=DEFAULT_EXTERNAL_STALE_MINUTES,
        help=("consider pid-less external runs stale after no log activity "
              f"(default: {DEFAULT_EXTERNAL_STALE_MINUTES})"))
    status.set_defaults(func=cmd_status)

    tail = sub.add_parser("tail", help="last lines of one run's log")
    tail.add_argument("--name", required=True)
    tail.add_argument("--lines", type=int, default=30)
    tail.set_defaults(func=cmd_tail)

    args = ap.parse_args()
    if getattr(args, "command", None) and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    args.func(args)


if __name__ == "__main__":
    main()
