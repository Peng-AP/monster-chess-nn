import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from stage_artifacts import generation_complete, reanalysis_complete


class TestGenerationArtifacts(unittest.TestCase):
    def test_requires_summary_counts_and_every_game(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = {
                "num_games_requested": 2, "saved_games": 2,
                "skipped_empty": 0, "failed_games": 0,
                "timed_out_games": 0,
            }
            with open(os.path.join(directory, "generation_summary.json"),
                      "w", encoding="utf-8") as handle:
                json.dump(summary, handle)
            open(os.path.join(directory, "game_00000.jsonl"), "w").close()
            self.assertFalse(generation_complete(directory, 2))
            for index in range(2):
                with open(os.path.join(directory, f"game_{index:05d}.jsonl"),
                          "w", encoding="utf-8") as handle:
                    handle.write("{}\n")
            self.assertTrue(generation_complete(directory, 2))
            self.assertFalse(generation_complete(directory, 3))

    def test_rejects_reported_generation_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = {
                "num_games_requested": 1, "saved_games": 1,
                "skipped_empty": 0, "failed_games": 1,
                "timed_out_games": 0,
            }
            with open(os.path.join(directory, "generation_summary.json"),
                      "w", encoding="utf-8") as handle:
                json.dump(summary, handle)
            with open(os.path.join(directory, "game_00000.jsonl"), "w") as h:
                h.write("{}\n")
            self.assertFalse(generation_complete(directory, 1))


class TestReanalysisArtifacts(unittest.TestCase):
    def test_requires_bound_summary_and_teacher_set(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = {"census": {"sampled": 2}, "kept": 1,
                       "simulations": 3200}
            with open(os.path.join(directory, "reanalysis_summary.json"),
                      "w", encoding="utf-8") as handle:
                json.dump(summary, handle)
            with open(os.path.join(directory, "teacher_00000.jsonl"), "w") as h:
                h.write("{}\n")
            self.assertTrue(reanalysis_complete(directory, 2, 1, 3200))
            self.assertFalse(reanalysis_complete(directory, 2, 1, 1600))
            self.assertFalse(reanalysis_complete(directory, 3, 1, 3200))


if __name__ == "__main__":
    unittest.main()
