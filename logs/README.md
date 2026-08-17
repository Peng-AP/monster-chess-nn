# logs/

One log per long run, appended live, so progress is readable without
touching a terminal or reading CPU counters.

    logs/<name>.log     the run's output
    logs/<name>.json    command, pid, start time

**To see what is happening:**

    py -3 tools/runs.py status          # every run: alive?, elapsed, latest progress
    py -3 tools/runs.py tail --name X   # last lines of one run

`status` reads the newest progress line from each log, stripping the carriage
returns that tqdm-style bars use to overwrite in place — otherwise the tail of
such a log is one enormous line.

Long runs are started through the same tool so the log and its metadata always
exist:

    py -3 tools/runs.py start --name curve -- py -3 tools/promotion_defense_probe.py ...

## Keeping the directory readable

The root holds only the current working set — recent runs, which are also the
ones `status` lists. Everything else is filed by the day it went idle:

    logs/archive/<YYYY-MM-DD>/<name>.log|json

    py -3 tools/runs.py prune              # archive runs idle over 7d
    py -3 tools/runs.py prune --days 1     # tighter working set

`tail` searches the archive recursively, so pruning never makes a log
unreadable — ask for a run by name and it is found wherever it was filed.

Two failure modes are pinned by `tests/test_runs_status.py`. The archive was
originally flat, which becomes unreadable long before it becomes large: 191
files in one directory is harder to search than the same files under a dozen
dated ones. And `prune` iterates *metadata* files, so a job launched outside
`runs.py` — writing a bare `.log` or `.err` with no `.json` beside it — was
invisible to it and accumulated forever; those orphans are now swept too, once
they are older than the cutoff and can no longer be actively written to.

Log *contents* are gitignored (they are run output, not source); this README
and the convention are tracked.
