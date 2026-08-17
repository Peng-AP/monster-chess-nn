import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_generation_data as audit  # noqa: E402


class GenerationDataAuditContracts(unittest.TestCase):
    def _fixture(self, root: Path, include_all=True):
        raw = root / "raw"
        reanalysis = raw / "reanalysis"
        processed = root / "processed"
        reanalysis.mkdir(parents=True)
        processed.mkdir()
        split = {"train": [], "val": [], "test": []}
        for index, side in enumerate(("white", "black")):
            source = f"selfplay/game_{index:05d}.jsonl"
            teacher = f"reanalysis/teacher_{index:05d}.jsonl"
            record = {
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "current_player": side,
                "policy": {"a1a2": 1.0},
                "source": "deep_search_reanalysis",
                "source_record": {"path": source, "line": 1},
            }
            (reanalysis / f"teacher_{index:05d}.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8")
            split["train"].append(source)
            if include_all or index == 0:
                split["train"].append(teacher)
        (reanalysis / "reanalysis_summary.json").write_text(json.dumps({
            "kept": 2, "simulations": 3200,
        }), encoding="utf-8")
        split.update({
            "retention": {"min_nonhuman_plies": 0},
            "augment": True, "value_floor": 0.5, "value_horizon": 60,
        })
        (processed / "split_game_ids.json").write_text(
            json.dumps(split), encoding="utf-8")
        np.save(processed / "positions.npy",
                np.zeros((4, 8, 8, 15), np.float32))
        np.save(processed / "policy_weights.npy", np.ones(4, np.float32))
        np.save(processed / "value_weights.npy", np.zeros(4, np.float32))
        return raw, reanalysis, processed

    def test_complete_teacher_set_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            result = audit.audit(*paths, 2, 0.5, 0.5, 60)
            self.assertEqual(result["verdict"], "PASS")
            self.assertEqual(result["teacher_rows_after_augmentation"], 4)
            self.assertEqual(result["teacher_split_linkage_checked"], 2)

    def test_missing_processed_teacher_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory), include_all=False)
            with self.assertRaisesRegex(RuntimeError, "absent from processed"):
                audit.audit(*paths, 2, 0.5, 0.5, 60)


if __name__ == "__main__":
    unittest.main()
