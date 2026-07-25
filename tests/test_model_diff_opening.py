import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from model_diff import (  # noqa: E402
    BLACK_KING_PLANE, BLACK_PAWN_PLANE, WHITE_PAWN_PLANE, opening_mask,
)


def _position(white_pawns, black_men, channels=15):
    """Minimal position tensor with the requested piece counts."""
    t = np.zeros((8, 8, channels), dtype=np.float32)
    for i in range(white_pawns):
        t[1, i, WHITE_PAWN_PLANE] = 1.0
    # spread Black men across the Black piece planes (6..11)
    placed = 0
    for plane in range(BLACK_PAWN_PLANE, BLACK_KING_PLANE + 1):
        for file in range(8):
            if placed >= black_men:
                break
            t[6, file, plane] = 1.0
            placed += 1
    return t


class OpeningMaskContracts(unittest.TestCase):
    def test_selects_only_full_material_openings(self):
        positions = np.stack([
            _position(4, 16),   # untouched opening -> keep
            _position(4, 14),   # two Black men gone -> keep (boundary)
            _position(3, 16),   # a White pawn gone -> drop
            _position(4, 13),   # three Black men gone -> drop (boundary)
            _position(0, 16),   # post-cliff -> drop
        ])
        mask = opening_mask(positions, np.arange(len(positions)))
        self.assertEqual(list(mask), [True, True, False, False, False])

    def test_mask_is_indexed_by_position_in_idx_not_by_row(self):
        """The mask must align with idx order, not the underlying array order."""
        positions = np.stack([_position(0, 16), _position(4, 16)])
        mask = opening_mask(positions, np.array([1, 0]))
        self.assertEqual(list(mask), [True, False])

    def test_chunking_does_not_change_the_result(self):
        positions = np.stack(
            [_position(4, 16) if i % 3 == 0 else _position(1, 16) for i in range(50)])
        idx = np.arange(len(positions))
        self.assertEqual(
            list(opening_mask(positions, idx, chunk=7)),
            list(opening_mask(positions, idx, chunk=8192)),
        )

    def test_works_on_the_17_plane_layout_unchanged(self):
        """Planes 0-11 are shared, so the filter needs no channel conversion."""
        p15 = np.stack([_position(4, 16, channels=15), _position(1, 16, channels=15)])
        p17 = np.stack([_position(4, 16, channels=17), _position(1, 16, channels=17)])
        idx = np.arange(2)
        self.assertEqual(list(opening_mask(p15, idx)), list(opening_mask(p17, idx)))


if __name__ == "__main__":
    unittest.main()
