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


if __name__ == "__main__":
    unittest.main()
