"""Merging a source in as a policy-only teacher.

The D2 fork rests entirely on the stamp: Merge-K and Merge-B are the *same*
829 games, and the only difference is whether those records teach the value
head. If the stamp silently failed, K and B would be the same experiment and
the knowledge-vs-belief question would come back with a null that means
nothing.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "merge_source.py"


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.base = self.tmp / "base"
        write_jsonl(self.base / "selfplay" / "game_0000.jsonl",
                    [{"fen": "f1", "policy": {"e2e4": 1.0}, "game_result": 1}])
        (self.base / "corpus_manifest.json").write_text('{"recipe": "base"}')
        self.src = self.tmp / "ps"
        write_jsonl(self.src / "ps_a.jsonl",
                    [{"fen": "p1", "policy": {"a2a3": 1.0}, "game_result": 1,
                      "policy_weight": 1.0},
                     {"fen": "p2", "policy": {"b2b3": 1.0}, "game_result": 1,
                      "policy_weight": 0.0}])

    def merged(self, out_name, *extra):
        out = self.tmp / out_name
        r = run("--base-dir", str(self.base), "--source", str(self.src),
                "--as", "ps_monster", "--out-dir", str(out), *extra)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        lines = (out / "ps_monster" / "ps_a.jsonl").read_text(
            encoding="utf-8").splitlines()
        recs = [json.loads(line) for line in lines if line.strip()]
        return out, recs

    def test_value_weight_zero_is_stamped_on_every_record(self):
        _out, recs = self.merged("K", "--value-weight", "0")
        self.assertEqual([r["value_weight"] for r in recs], [0.0, 0.0])

    def test_without_the_flag_no_value_weight_is_added(self):
        _out, recs = self.merged("B")
        self.assertFalse(any("value_weight" in r for r in recs))

    def test_the_two_arms_differ_only_in_the_stamp(self):
        # This is the whole point of the A/B: same games, one difference.
        _k, k_recs = self.merged("K2", "--value-weight", "0")
        _b, b_recs = self.merged("B2")
        strip = lambda rs: [{k: v for k, v in r.items() if k != "value_weight"}  # noqa: E731
                            for r in rs]
        self.assertEqual(strip(k_recs), strip(b_recs))

    def test_existing_policy_weights_are_preserved(self):
        # ps_monster ships winner-only policy weighting; merging must not undo it.
        _out, recs = self.merged("K3", "--value-weight", "0")
        self.assertEqual([r["policy_weight"] for r in recs], [1.0, 0.0])

    def test_base_corpus_is_untouched(self):
        before = (self.base / "selfplay" / "game_0000.jsonl").read_bytes()
        self.merged("K4", "--value-weight", "0")
        self.assertEqual((self.base / "selfplay" / "game_0000.jsonl").read_bytes(),
                         before)
        self.assertFalse((self.base / "ps_monster").exists())

    def test_source_lands_in_its_own_subdirectory(self):
        out, _recs = self.merged("K5", "--value-weight", "0")
        self.assertTrue((out / "ps_monster" / "ps_a.jsonl").exists())
        self.assertTrue((out / "selfplay" / "game_0000.jsonl").exists())

    def test_manifest_records_the_stamp(self):
        out, _recs = self.merged("K6", "--value-weight", "0")
        m = json.loads((out / "corpus_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["recipe"], "base")
        entry = m["merged_sources"][0]
        self.assertEqual(entry["as"], "ps_monster")
        self.assertEqual(entry["value_weight"], 0.0)
        self.assertEqual(entry["records"], 2)

    def test_refuses_to_overwrite_an_existing_corpus(self):
        out, _recs = self.merged("K7", "--value-weight", "0")
        r = run("--base-dir", str(self.base), "--source", str(self.src),
                "--as", "ps_monster", "--out-dir", str(out))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refusing to overwrite", r.stdout + r.stderr)

    def test_optional_dedupe_skips_base_overlap_and_internal_duplicates(self):
        duplicate = [
            {"fen": "p1", "policy": {"a2a3": 1.0}, "game_result": 1,
             "policy_weight": 1.0, "value_weight": 0.0},
            {"fen": "p2", "policy": {"b2b3": 1.0}, "game_result": 1,
             "policy_weight": 0.0, "value_weight": 0.0},
        ]
        write_jsonl(self.base / "old" / "same_game.jsonl", duplicate)
        write_jsonl(self.src / "ps_dup.jsonl", [
            {k: v for k, v in rec.items() if k != "value_weight"}
            for rec in duplicate
        ])
        write_jsonl(self.src / "ps_new.jsonl", [
            {"fen": "new", "policy": {"c2c3": 1.0}, "game_result": -1}
        ])
        out = self.tmp / "deduped"
        result = run(
            "--base-dir", str(self.base), "--source", str(self.src),
            "--as", "expanded", "--out-dir", str(out),
            "--value-weight", "0", "--dedupe-against-base",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([path.name for path in (out / "expanded").glob("*.jsonl")],
                         ["ps_new.jsonl"])
        manifest = json.loads((out / "corpus_manifest.json").read_text())
        entry = manifest["merged_sources"][0]
        self.assertEqual(entry["games"], 1)
        self.assertEqual(entry["duplicate_games_skipped"], 2)
        self.assertTrue(entry["dedupe_against_base"])


if __name__ == "__main__":
    unittest.main()
