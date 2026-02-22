import sys
import json
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_generation as dg


class _FixedRng:
    def __init__(self, val):
        self._val = float(val)

    def random(self):
        return self._val


class DataGenerationContracts(unittest.TestCase):
    def setUp(self):
        self._orig_train_side = dg._train_side
        self._orig_skip_check_positions = dg._skip_check_positions
        self._orig_curriculum = dg._curriculum
        self._orig_record_all_plies = dg._record_all_plies

    def tearDown(self):
        dg._train_side = self._orig_train_side
        dg._skip_check_positions = self._orig_skip_check_positions
        dg._curriculum = self._orig_curriculum
        dg._record_all_plies = self._orig_record_all_plies

    def test_check_positions_kept_for_training_side(self):
        dg._train_side = "black"
        dg._skip_check_positions = True
        dg._curriculum = False
        dg._record_all_plies = False
        # training-side position (black to move), no random skip
        skip = dg._should_skip_record(
            move_number=9,
            is_white=False,
            board_is_check=True,
            rng=_FixedRng(0.99),
        )
        self.assertFalse(skip)

    def test_check_positions_skipped_for_non_training_side(self):
        dg._train_side = "black"
        dg._skip_check_positions = True
        dg._curriculum = False
        dg._record_all_plies = False
        # non-training-side position (white to move), no random skip
        skip = dg._should_skip_record(
            move_number=9,
            is_white=True,
            board_is_check=True,
            rng=_FixedRng(0.99),
        )
        self.assertTrue(skip)

    def test_record_all_plies_overrides_skip_policy(self):
        dg._train_side = "black"
        dg._skip_check_positions = True
        dg._curriculum = False
        dg._record_all_plies = True
        skip = dg._should_skip_record(
            move_number=0,
            is_white=True,
            board_is_check=True,
            rng=_FixedRng(0.0),
        )
        self.assertFalse(skip)

    def test_to_white_perspective_value_respects_side(self):
        self.assertAlmostEqual(dg._to_white_perspective_value(0.4, "white"), 0.4)
        self.assertAlmostEqual(dg._to_white_perspective_value(0.4, "black"), -0.4)
        self.assertIsNone(dg._to_white_perspective_value("bad", "white"))

    def test_load_start_fens_filters_by_white_value_max(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "starts.jsonl"
            records = [
                {
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1",
                    "current_player": "white",
                    "mcts_value": 0.5,
                },
                {
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1",
                    "current_player": "white",
                    "mcts_value": -0.5,
                },
            ]
            with open(p, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            fens, stats = dg._load_start_fens(
                file_path=str(p),
                side_filter="any",
                white_value_max=0.0,
            )
            self.assertEqual(len(fens), 1)
            self.assertEqual(stats.get("filtered_by_white_value"), 1)

    def test_load_start_fens_filters_before_black_conversion(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "starts.jsonl"
            records = [
                {
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1",
                    "current_player": "white",
                    "mcts_value": 0.4,
                },
                {
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1",
                    "current_player": "white",
                    "mcts_value": -0.4,
                },
            ]
            with open(p, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            fens, stats = dg._load_start_fens(
                file_path=str(p),
                side_filter="black",
                convert_white_to_black=True,
                white_value_max=0.0,
                seed=123,
            )
            self.assertEqual(len(fens), 1)
            self.assertEqual(stats.get("filtered_by_white_value"), 1)
            self.assertEqual(stats.get("converted_added"), 1)


if __name__ == "__main__":
    unittest.main()
