import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from train import build_model, infer_se_config, load_model_for_inference  # noqa: E402


def small(**kwargs):
    return build_model(input_channels=15, stem_channels=16,
                       residual_block_channels=(16, 16), **kwargs)


class SqueezeExciteContracts(unittest.TestCase):
    def test_se_remains_off_by_default(self):
        self.assertFalse(small().use_se_blocks)

    def test_se_forward_preserves_public_shapes(self):
        model = small(use_se_blocks=True, se_reduction=4)
        value, policy = model(torch.zeros(2, 15, 8, 8))
        self.assertEqual(tuple(value.shape), (2, 1))
        self.assertEqual(tuple(policy.shape), (2, 4096))

    def test_checkpoint_infers_reduction_and_round_trips(self):
        model = small(use_se_blocks=True, se_reduction=4).eval()
        self.assertEqual(infer_se_config(model.state_dict()), (True, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            torch.save(model.state_dict(), path)
            loaded, _ = load_model_for_inference(path, torch.device("cpu"))
        self.assertTrue(loaded.use_se_blocks)
        self.assertEqual(loaded.se_reduction, 4)

    def test_se_parameter_cost_is_small(self):
        base = small()
        se = small(use_se_blocks=True, se_reduction=4)
        added = (sum(p.numel() for p in se.parameters())
                 - sum(p.numel() for p in base.parameters()))
        # Two 16-channel blocks: each has 16->4->16 plus biases.
        self.assertLess(added, 400)


if __name__ == "__main__":
    unittest.main()
