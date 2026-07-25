"""tools/set_human_duplication.py rebuilds a corpus at another human multiple.

The duplication multiple is baked in at merge time, so the only way to ask
whether 6x duplication cost Black is to rebuild the same games at 1x and
retrain.  This tool must touch human_games/ and nothing else, and must leave
copy boundaries machine-readable (data_processor computes distance-to-end
within each copy).
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE = [
    {"fen": "8/8/8/8/8/4p3/8/K6k w - - 0 1", "current_player": "white",
     "mcts_value": 0.0, "policy": {"a1a2": 1.0}, "game_result": 1.0},
    {"fen": "8/8/8/8/8/4p3/K7/7k b - - 1 1", "current_player": "black",
     "mcts_value": 0.0, "policy": {"e3e2": 1.0}, "game_result": 1.0},
]


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records),
                    encoding="utf-8")


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def run(raw, out, copies):
    return subprocess.run(
        [sys.executable, "tools/set_human_duplication.py",
         "--raw-dir", str(raw), "--out-dir", str(out), "--copies", str(copies)],
        cwd=ROOT, capture_output=True, text=True)


class SetHumanDuplicationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name) / "raw"
        self.out = Path(self._tmp.name) / "out"
        write(self.raw / "human_games" / "game_0000.jsonl", BASE * 6)
        write(self.raw / "heuristic" / "game_00000.jsonl", BASE)
        self.addCleanup(self._tmp.cleanup)

    def test_collapses_to_one_copy_and_stamps_segment(self):
        result = run(self.raw, self.out, 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        recs = read(self.out / "human_games" / "game_0000.jsonl")
        self.assertEqual(len(recs), len(BASE))
        self.assertEqual([r["segment"] for r in recs], [0, 0])
        self.assertEqual([{k: v for k, v in r.items() if k != "segment"}
                          for r in recs], BASE)

    def test_re_duplicates_and_numbers_each_copy(self):
        run(self.raw, self.out, 1)
        again = Path(self._tmp.name) / "out2"
        result = run(self.out, again, 3)
        self.assertEqual(result.returncode, 0, result.stderr)
        recs = read(again / "human_games" / "game_0000.jsonl")
        self.assertEqual(len(recs), 3 * len(BASE))
        self.assertEqual([r["segment"] for r in recs], [0, 0, 1, 1, 2, 2])

    def test_leaves_non_human_games_untouched(self):
        run(self.raw, self.out, 1)
        self.assertEqual(read(self.out / "heuristic" / "game_00000.jsonl"), BASE)

    def test_passes_through_files_whose_copies_differ(self):
        # Not a clean N-fold repeat: rewriting would silently drop real moves.
        mixed = BASE + [dict(BASE[0]), dict(BASE[1], policy={"e3e2": 0.5})]
        write(self.raw / "human_games" / "game_0001.jsonl", mixed)
        result = run(self.raw, self.out, 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read(self.out / "human_games" / "game_0001.jsonl"), mixed)
        self.assertIn("passed through", result.stdout)

    def test_refuses_to_write_over_its_input(self):
        result = run(self.raw, self.raw, 1)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
