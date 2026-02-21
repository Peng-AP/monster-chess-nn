import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train


class TrainWdlContracts(unittest.TestCase):
    def test_build_wdl_targets_respects_draw_band(self):
        vals = np.array([-1.0, -0.2, -0.04, 0.0, 0.03, 0.2, 1.0], dtype=np.float32)
        labels = train.build_wdl_targets(vals, draw_epsilon=0.05)
        # loss, loss, draw, draw, draw, win, win
        self.assertEqual(labels.tolist(), [0, 0, 1, 1, 1, 2, 2])

    def test_dualhead_forward_with_wdl_shapes(self):
        model = train.build_model(
            use_wdl_head=True,
            value_head_mode="wdl",
            use_se_blocks=False,
            use_side_specialized_heads=False,
        )
        x = torch.zeros((3, train.IN_CHANNELS, 8, 8), dtype=torch.float32)
        x[:, train.TURN_LAYER, :, :] = 1.0
        value, policy = model(x)
        self.assertEqual(tuple(value.shape), (3, 1))
        self.assertEqual(tuple(policy.shape), (3, train.POLICY_SIZE))

        value2, policy2, wdl = model.forward_with_wdl(x)
        self.assertEqual(tuple(value2.shape), (3, 1))
        self.assertEqual(tuple(policy2.shape), (3, train.POLICY_SIZE))
        self.assertIsNotNone(wdl)
        self.assertEqual(tuple(wdl.shape), (3, 3))

    def test_dualhead_forward_with_wdl_none_when_disabled(self):
        model = train.build_model(
            use_wdl_head=False,
            value_head_mode="scalar",
            use_se_blocks=False,
            use_side_specialized_heads=False,
        )
        x = torch.zeros((2, train.IN_CHANNELS, 8, 8), dtype=torch.float32)
        value, policy, wdl = model.forward_with_wdl(x)
        self.assertEqual(tuple(value.shape), (2, 1))
        self.assertEqual(tuple(policy.shape), (2, train.POLICY_SIZE))
        self.assertIsNone(wdl)


if __name__ == "__main__":
    unittest.main()
