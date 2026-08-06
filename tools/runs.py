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
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
DEFAULT_RECENT_MINUTES = 60
DEFAULT_EXTERNAL_STALE_MINUTES = 30
TAIL_BYTES = 256 * 1024
DEFAULT_PRUNE_DAYS = 7


def _windows_process_started_epoch(pid):
    """Return a Windows process creation timestamp without extra dependencies."""
    try:
        import ctypes
        from ctypes import wintypes

        query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            query_limited_information, False, int(pid))
        if not handle:
            return None
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        try:
            if not kernel32.GetProcessTimes(
                    handle, ctypes.byref(creation), ctypes.byref(exit_time),
                    ctypes.byref(kernel), ctypes.byref(user)):
                return None
        finally:
            kernel32.CloseHandle(handle)
        ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return ticks / 10_000_000 - 11_644_473_600
    except Exception:
        return None


def _live_pids():
    """Every running pid, from one `tasklist` call.

    Asking about pids one at a time costs a subprocess spawn each: with a few
    dozen recorded runs that was ~280ms per entry and made `status` take 14
    seconds. One call answers for all of them.

    Returns None if the snapshot cannot be taken, which sends callers back to
    the per-pid path rather than silently reporting everything dead.
    """
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    pids = set()
    for row in csv.reader(out.splitlines()):
        # Column 1 is the pid. Parsing the column (rather than scanning text)
        # keeps a dead pid from matching inside another process's memory field.
        if len(row) >= 2 and row[1].strip().isdigit():
            pids.add(int(row[1]))
    return pids


def _alive(pid, started_epoch=0, live_pids=None):
    if pid is None:
        return False
    try:
        if os.name == "nt":
            if live_pids is None:
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True, timeout=15).stdout
                present = any(
                    len(row) >= 2 and row[1].strip().isdigit()
                    and int(row[1]) == int(pid)
                    for row in csv.reader(out.splitlines()))
            else:
                present = int(pid) in live_pids
            if not present:
                return False
            if started_epoch:
                # Cheap in-process call, no spawn: qualifying by creation time
                # stops a recycled pid resurrecting a finished run.
                process_started = _windows_process_started_epoch(pid)
                if process_started is None:
                    # Unreadable creation time means the pid is not ours. A run
                    # we launched is always queryable; a pid Windows has since
                    # recycled into a service process is not. Observed for real:
                    # a finished run's pid became svchost.exe and read as alive
                    # forever, because this used to fall through to True.
                    return False
                # Metadata has one-second precision and is written immediately
                # after Popen returns.
                return abs(process_started - started_epoch) <= 5
            return True
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
        # Only the tail can be the latest progress line, and these logs append
        # for hours. Reading the last chunk keeps status flat in log size; a
        # split multi-byte character at the seek point is absorbed by
        # errors="replace" and can only affect the first line, never the last.
        with open(path, "rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - TAIL_BYTES))
            text = fh.read().decode("utf-8", errors="replace")
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


def _run_state(meta, last_activity, now, external_stale_seconds, live_pids=None):
    """Classify a record without letting pid-less adopted jobs live forever."""
    pid = meta.get("pid")
    if pid is not None:
        started_epoch = _started_epoch(meta.get("started"))
        return ("RUNNING" if _alive(pid, started_epoch=started_epoch,
                                    live_pids=live_pids)
                else "finished")
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
    live_pids = _live_pids()
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
        state = _run_state(meta, last_activity, now, external_stale_seconds,
                           live_pids=live_pids)
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


def cmd_stop(args):
    """Stop a run and everything it spawned.

    Killing the recorded pid alone is not enough and has caused real damage: a
    `queue_after` wrapper that has already launched its command dies while the
    command keeps running, orphaned, writing into the same output directory and
    the same log as its replacement. Windows does not reparent-and-kill, so the
    tree has to be taken explicitly.
    """
    meta_path = os.path.join(LOGS, f"{args.name}.json")
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        raise SystemExit(f"no run named {args.name}")
    pid = meta.get("pid")
    if pid is None:
        raise SystemExit(f"{args.name} has no recorded pid (external run)")
    if os.name == "nt":
        out = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                             capture_output=True, text=True).stdout.strip()
        print(out or f"{args.name}: nothing to stop")
    else:
        import signal
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        print(f"{args.name}: killed process group {pid}")


def cmd_prune(args):
    """Move old finished runs to logs/archive/. Never touches a live run."""
    archive = os.path.join(LOGS, "archive")
    os.makedirs(archive, exist_ok=True)
    now = time.time()
    cutoff = max(0, args.days) * 86400
    live_pids = _live_pids()
    moved = kept = 0
    for meta_name in sorted(f for f in os.listdir(LOGS) if f.endswith(".json")):
        meta_path = os.path.join(LOGS, meta_name)
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        name = meta.get("name", meta_name[:-5])
        log_path = os.path.join(LOGS, f"{name}.log")
        started_epoch = _started_epoch(meta.get("started"))
        last_activity = _last_activity(meta_path, log_path, started_epoch)
        state = _run_state(meta, last_activity, now,
                           max(0, args.external_stale_minutes) * 60,
                           live_pids=live_pids)
        if state in ("RUNNING", "external") or now - last_activity < cutoff:
            kept += 1
            continue
        for src in (meta_path, log_path):
            if os.path.exists(src):
                target = os.path.join(archive, os.path.basename(src))
                if os.path.exists(target):
                    os.remove(target)
                os.replace(src, target)
        moved += 1
        if args.verbose:
            print(f"  archived {name}")
    print(f"archived {moved} run(s) idle over {args.days}d, kept {kept}; "
          f"archive at logs/archive/")


def cmd_tail(args):
    path = os.path.join(LOGS, f"{args.name}.log")
    if not os.path.exists(path):
        archived = os.path.join(LOGS, "archive", f"{args.name}.log")
        if os.path.exists(archived):
            path = archived        # pruning must not make a log unreadable
        else:
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

    stop = sub.add_parser(
        "stop", help="kill a run AND its children (never just the pid)")
    stop.add_argument("--name", required=True)
    stop.set_defaults(func=cmd_stop)

    prune = sub.add_parser(
        "prune", help="archive old finished runs (never touches a live one)")
    prune.add_argument("--days", type=float, default=DEFAULT_PRUNE_DAYS,
                       help=f"archive runs idle this long (default: "
                            f"{DEFAULT_PRUNE_DAYS})")
    prune.add_argument("--external-stale-minutes", type=float,
                       default=DEFAULT_EXTERNAL_STALE_MINUTES)
    prune.add_argument("--verbose", action="store_true")
    prune.set_defaults(func=cmd_prune)

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
