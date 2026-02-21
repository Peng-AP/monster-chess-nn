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

    def tearDown(self):
        dg._train_side = self._orig_train_side
        dg._skip_check_positions = self._orig_skip_check_positions
        dg._curriculum = self._orig_curriculum

    def test_check_positions_kept_for_training_side(self):
        dg._train_side = "black"
        dg._skip_check_positions = True
        dg._curriculum = False
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
        # non-training-side position (white to move), no random skip
        skip = dg._should_skip_record(
            move_number=9,
            is_white=True,
            board_is_check=True,
            rng=_FixedRng(0.99),
        )
        self.assertTrue(skip)


if __name__ == "__main__":
    unittest.main()
