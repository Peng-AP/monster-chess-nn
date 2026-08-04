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

Log *contents* are gitignored (they are run output, not source); this README
and the convention are tracked.
