import sys
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


if __name__ == "__main__":
    unittest.main()
