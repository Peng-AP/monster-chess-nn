import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train


class TrainWdlContracts(unittest.TestCase):
    def test_load_data_defaults_legacy_policy_weights_to_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            np.save(data_dir / "positions.npy", np.zeros((2, 8, 8, train.IN_CHANNELS),
                                                          dtype=np.float32))
            np.save(data_dir / "mcts_values.npy", np.zeros((2,), dtype=np.float32))
            np.save(data_dir / "game_results.npy", np.zeros((2,), dtype=np.float32))
            np.save(data_dir / "policies.npy",
                    np.zeros((2, train.POLICY_SIZE), dtype=np.float32))
            np.savez(data_dir / "splits.npz", train=np.array([0]),
                     val=np.array([], dtype=np.int64), test=np.array([1]))

            (_x, _mv, _gr, _pol, weights, value_weights,
             _splits) = train.load_data(data_dir)

            self.assertEqual(weights.tolist(), [1.0, 1.0])
            self.assertEqual(value_weights.tolist(), [1.0, 1.0])

    def test_policy_loss_ignores_zero_weight_targets(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        weights = torch.tensor([1.0, 0.0])

        loss = train.weighted_policy_cross_entropy(logits, targets, weights)
        expected = torch.nn.functional.cross_entropy(logits[:1], targets[:1])
        self.assertAlmostEqual(loss.item(), expected.item())

        loss.backward()
        self.assertTrue(torch.all(logits.grad[1] == 0))

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
        )
        x = torch.zeros((2, train.IN_CHANNELS, 8, 8), dtype=torch.float32)
        value, policy, wdl = model.forward_with_wdl(x)
        self.assertEqual(tuple(value.shape), (2, 1))
        self.assertEqual(tuple(policy.shape), (2, train.POLICY_SIZE))
        self.assertIsNone(wdl)

    def test_checkpoint_stem_infers_legacy_and_current_encodings(self):
        legacy = train.build_model(
            input_channels=15,
            use_wdl_head=True,
            value_head_mode="wdl",
            use_se_blocks=False,
        )
        current = train.build_model(
            input_channels=17,
            use_wdl_head=True,
            value_head_mode="wdl",
            use_se_blocks=False,
        )

        self.assertEqual(train.infer_input_channels(legacy.state_dict()), 15)
        self.assertEqual(train.infer_input_channels(current.state_dict()), 17)

    def test_unpack_loader_batch_accepts_3_and_4_tuple_batches(self):
        x = torch.zeros((1, train.IN_CHANNELS, 8, 8), dtype=torch.float32)
        yv = torch.zeros((1, 1), dtype=torch.float32)
        yp = torch.zeros((1, train.POLICY_SIZE), dtype=torch.float32)
        yw = torch.zeros((1,), dtype=torch.int64)

        X3, yv3, yp3, ypw3, yvw3, yw3 = train._unpack_loader_batch((x, yv, yp))
        self.assertIsNone(yw3)
        self.assertEqual(ypw3.tolist(), [1.0])
        self.assertEqual(yvw3.tolist(), [1.0])
        self.assertEqual(tuple(X3.shape), tuple(x.shape))
        self.assertEqual(tuple(yv3.shape), tuple(yv.shape))
        self.assertEqual(tuple(yp3.shape), tuple(yp.shape))

        X4, yv4, yp4, ypw4, yvw4, yw4 = train._unpack_loader_batch((x, yv, yp, yw))
        self.assertIsNotNone(yw4)
        self.assertEqual(ypw4.tolist(), [1.0])
        self.assertEqual(yvw4.tolist(), [1.0])
        self.assertEqual(tuple(X4.shape), tuple(x.shape))
        self.assertEqual(tuple(yv4.shape), tuple(yv.shape))
        self.assertEqual(tuple(yp4.shape), tuple(yp.shape))
        self.assertEqual(tuple(yw4.shape), tuple(yw.shape))

        policy_weight = torch.zeros((1,), dtype=torch.float32)
        X5, yv5, yp5, ypw5, yvw5, yw5 = train._unpack_loader_batch(
            (x, yv, yp, policy_weight, yw))
        self.assertEqual(ypw5.tolist(), [0.0])
        self.assertEqual(yvw5.tolist(), [1.0])
        self.assertEqual(yw5.tolist(), [0])

if __name__ == "__main__":
    unittest.main()
