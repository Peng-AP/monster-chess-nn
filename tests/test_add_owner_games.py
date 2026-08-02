"""Owner-game intake: membership by content, base corpus never touched.

Two ways this goes wrong silently, both pinned here:

1. **Double-adding.** Corpus copies are renamed and stored at N-fold in-file
   duplication, so a game already present looks nothing like its source file.
   Re-running the intake must skip it, not add a second copy at 6x weight.
2. **Mutating the base.** combined_v17 is what v17, ramp and every v18 arm
   trained on. The tool must always write a new directory.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "add_owner_games.py"


def write_game(path, fens, copies=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in range(copies):
            for fen in fens:
                rec = {"fen": fen, "policy": {"e2e4": 1.0}, "game_result": -1,
                       "mcts_value": 0.0, "current_player": "white",
                       "source": "human_game", "actor": "human"}
                if copies > 1:
                    rec["segment"] = c
                f.write(json.dumps(rec) + "\n")


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


class TestIntake(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.base = self.tmp / "base"
        self.src = self.tmp / "src_games"
        # One game already in the corpus, stored at 6x like the real thing.
        write_game(self.base / "human_games" / "game_0000.jsonl",
                   ["fen_a", "fen_b", "fen_c"], copies=6)
        (self.base / "corpus_manifest.json").write_text('{"recipe": "test"}')
        # The same game as its 1x source, plus a genuinely new one.
        write_game(self.src / "game_00000.jsonl", ["fen_a", "fen_b", "fen_c"])
        write_game(self.src / "game_00001.jsonl", ["fen_x", "fen_y"])

    def intake(self, out_name, copies=6):
        out = self.tmp / out_name
        r = run("--base-dir", str(self.base), "--out-dir", str(out),
                "--add", str(self.src), "--copies", str(copies))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return out, r.stdout

    def test_already_present_game_is_skipped_despite_duplication(self):
        out, log = self.intake("out1")
        self.assertIn("skip (already in corpus", log)
        names = sorted(p.name for p in (out / "human_games").glob("*.jsonl"))
        self.assertEqual(len(names), 2, names)   # the original + one new

    def test_added_game_gets_the_requested_duplication(self):
        out, _ = self.intake("out2", copies=6)
        added = [p for p in (out / "human_games").glob("*.jsonl")
                 if p.name != "game_0000.jsonl"][0]
        with open(added, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(recs), 2 * 6)
        self.assertEqual(sorted({r["segment"] for r in recs}), list(range(6)))

    def test_base_corpus_is_not_modified(self):
        before = {p.name: p.read_bytes()
                  for p in (self.base / "human_games").glob("*.jsonl")}
        self.intake("out3")
        after = {p.name: p.read_bytes()
                 for p in (self.base / "human_games").glob("*.jsonl")}
        self.assertEqual(before, after)

    def test_rerunning_the_intake_adds_nothing(self):
        out1, _ = self.intake("out4")
        out2 = self.tmp / "out5"
        r = run("--base-dir", str(out1), "--out-dir", str(out2),
                "--add", str(self.src), "--copies", "6")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("nothing to do", r.stdout)
        self.assertFalse(out2.exists(), "no corpus should be written")

    def test_refuses_to_overwrite_an_existing_corpus(self):
        out, _ = self.intake("out6")
        r = run("--base-dir", str(self.base), "--out-dir", str(out),
                "--add", str(self.src))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refusing to overwrite", r.stderr + r.stdout)

    def test_refuses_to_write_into_the_base(self):
        r = run("--base-dir", str(self.base), "--out-dir", str(self.base),
                "--add", str(self.src))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("never modified", r.stderr + r.stdout)

    def test_manifest_records_the_provenance(self):
        out, _ = self.intake("out7")
        m = json.loads((out / "corpus_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["recipe"], "test")          # base fields survive
        self.assertEqual(m["human_games_total"], 2)
        intake = m["owner_game_intake"][0]
        self.assertEqual(intake["games_added"], 1)
        self.assertEqual(intake["games_skipped_already_present"], 1)
        self.assertEqual(intake["copies"], 6)
        self.assertIn("game_00001.jsonl", intake["added"][0]["source"])


if __name__ == "__main__":
    unittest.main()
