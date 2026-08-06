import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import runs  # noqa: E402


class RunsStatusTests(unittest.TestCase):
    @mock.patch.object(runs.os, "name", "nt")
    @mock.patch.object(runs.subprocess, "run")
    def test_windows_alive_requires_exact_pid_column_match(self, run):
        run.return_value.stdout = (
            '"python.exe","9012","Console","1","28,104 K"\n')

        self.assertFalse(runs._alive(28104))
        self.assertTrue(runs._alive(9012))

    @mock.patch.object(runs.os, "name", "nt")
    @mock.patch.object(runs, "_windows_process_started_epoch")
    @mock.patch.object(runs.subprocess, "run")
    def test_windows_alive_rejects_reused_pid(self, run, process_started):
        run.return_value.stdout = (
            '"python.exe","9012","Console","1","28,104 K"\n')
        process_started.return_value = 2_000.0

        self.assertFalse(runs._alive(9012, started_epoch=1_000.0))
        self.assertTrue(runs._alive(9012, started_epoch=2_002.0))

    def _record(self, directory, name, pid, age_seconds):
        now = time.time()
        started = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(now - age_seconds))
        meta = directory / f"{name}.json"
        log = directory / f"{name}.log"
        meta.write_text(json.dumps({
            "name": name, "pid": pid, "started": started,
        }), encoding="utf-8")
        log.write_text("finished useful work\n", encoding="utf-8")
        activity = now - age_seconds
        os.utime(meta, (activity, activity))
        os.utime(log, (activity, activity))

    def _status(self, directory, *, show_all=False):
        args = SimpleNamespace(
            all=show_all,
            recent_minutes=60,
            external_stale_minutes=30,
        )
        output = io.StringIO()
        with mock.patch.object(runs, "LOGS", str(directory)), \
                mock.patch.object(runs, "_alive", return_value=False), \
                contextlib.redirect_stdout(output):
            runs.cmd_status(args)
        return output.getvalue()

    def test_old_finished_and_pidless_records_leave_default_view(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._record(directory, "finished_old", 123, 2 * 60 * 60)
            self._record(directory, "external_old", None, 2 * 60 * 60)

            output = self._status(directory)

            self.assertIn("no active or recent runs", output)
            self.assertIn("2 older entries hidden", output)
            self.assertNotIn("finished_old", output)
            self.assertNotIn("external_old", output)

    def test_all_retains_history_and_marks_old_external_record_stale(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._record(directory, "finished_old", 123, 2 * 60 * 60)
            self._record(directory, "external_old", None, 2 * 60 * 60)

            output = self._status(directory, show_all=True)

            self.assertIn("finished_old", output)
            self.assertIn("finished", output)
            self.assertIn("external_old", output)
            self.assertIn("stale", output)

    def test_recent_completed_run_remains_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._record(directory, "just_finished", 123, 5 * 60)

            output = self._status(directory)

            self.assertIn("just_finished", output)
            self.assertIn("finished", output)


class LivePidSnapshotTests(unittest.TestCase):
    """One tasklist call for all pids, instead of one call per pid."""

    @mock.patch.object(runs.os, "name", "nt")
    @mock.patch.object(runs.subprocess, "run")
    def test_snapshot_answers_without_spawning_per_pid(self, run):
        run.return_value.stdout = (
            '"python.exe","9012","Console","1","28,104 K"\n'
            '"python.exe","7777","Console","1","12,000 K"\n')
        live = runs._live_pids()
        self.assertEqual(live, {9012, 7777})

        run.reset_mock()
        self.assertTrue(runs._alive(9012, live_pids=live))
        self.assertFalse(runs._alive(1234, live_pids=live))
        run.assert_not_called()

    @mock.patch.object(runs.os, "name", "nt")
    @mock.patch.object(runs, "_windows_process_started_epoch")
    @mock.patch.object(runs.subprocess, "run")
    def test_snapshot_still_rejects_a_reused_pid(self, run, process_started):
        # The speedup must not cost the pid-reuse guard.
        process_started.return_value = 2_000.0
        self.assertFalse(
            runs._alive(9012, started_epoch=1_000.0, live_pids={9012}))
        self.assertTrue(
            runs._alive(9012, started_epoch=2_002.0, live_pids={9012}))

    @mock.patch.object(runs.os, "name", "nt")
    @mock.patch.object(runs.subprocess, "run", side_effect=OSError("boom"))
    def test_unavailable_snapshot_falls_back_rather_than_reporting_dead(self, _r):
        # None means "ask per pid", not "nothing is running" — otherwise a
        # failed tasklist would silently report every live run as finished.
        self.assertIsNone(runs._live_pids())

    def test_progress_reads_the_tail_of_a_large_log(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "big.log"
            filler = "noise line that is not progress\n" * 40_000
            path.write_text(filler + "[208/208] CONVERTED elapsed 3.0m\n",
                            encoding="utf-8")
            self.assertGreater(path.stat().st_size, runs.TAIL_BYTES)
            self.assertEqual(runs._last_progress(str(path)),
                             "[208/208] CONVERTED elapsed 3.0m")


class PruneTests(unittest.TestCase):
    def _prune(self, directory, live_pid, days=7):
        args = SimpleNamespace(
            days=days, external_stale_minutes=30, verbose=False)
        output = io.StringIO()
        with mock.patch.object(runs, "LOGS", str(directory)), \
                mock.patch.object(runs, "_live_pids", return_value=set()), \
                mock.patch.object(
                    runs, "_alive",
                    side_effect=lambda pid, **kw: pid == live_pid), \
                contextlib.redirect_stdout(output):
            runs.cmd_prune(args)
        return output.getvalue()

    def _record(self, directory, name, pid, age_seconds):
        RunsStatusTests._record(self, directory, name, pid, age_seconds)

    def test_prune_never_archives_a_live_run(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            # Live, but idle far longer than the cutoff: age must not win.
            self._record(directory, "still_running", 555, 30 * 86400)
            self._prune(directory, live_pid=555)
            self.assertTrue((directory / "still_running.json").exists())
            self.assertFalse(
                (directory / "archive" / "still_running.json").exists())

    def test_prune_archives_old_finished_runs_without_deleting_them(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._record(directory, "ancient", 123, 30 * 86400)
            self._prune(directory, live_pid=None)
            self.assertFalse((directory / "ancient.json").exists())
            self.assertTrue((directory / "archive" / "ancient.json").exists())
            self.assertTrue((directory / "archive" / "ancient.log").exists())

    def test_prune_keeps_recent_finished_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._record(directory, "yesterday", 123, 3600)
            self._prune(directory, live_pid=None)
            self.assertTrue((directory / "yesterday.json").exists())

    def test_archived_log_is_still_readable_by_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._record(directory, "ancient", 123, 30 * 86400)
            self._prune(directory, live_pid=None)
            output = io.StringIO()
            with mock.patch.object(runs, "LOGS", str(directory)), \
                    contextlib.redirect_stdout(output):
                runs.cmd_tail(SimpleNamespace(name="ancient", lines=10))
            self.assertIn("finished useful work", output.getvalue())


if __name__ == "__main__":
    unittest.main()
