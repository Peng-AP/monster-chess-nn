import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from train import (build_model, infer_policy_head_config,  # noqa: E402
                   infer_side_policy_adapters,
                   load_model_for_inference)


def small(**kwargs):
    return build_model(input_channels=15, stem_channels=8,
                       residual_block_channels=(8, 8), **kwargs)


class AttentionPolicyHeadContracts(unittest.TestCase):
    def test_dense_remains_default(self):
        self.assertEqual(small().policy_head_type, "dense")

    def test_attention_preserves_policy_abi(self):
        model = small(policy_head_type="attention",
                      policy_attention_channels=6)
        value, policy = model(torch.zeros(3, 15, 8, 8))
        self.assertEqual(tuple(value.shape), (3, 1))
        self.assertEqual(tuple(policy.shape), (3, 4096))

    def test_flat_index_is_source_times_64_plus_destination(self):
        model = small(policy_head_type="attention",
                      policy_attention_channels=4).eval()
        with torch.no_grad():
            model.policy_attention.weight.zero_()
            model.policy_attention.bias.zero_()
            model.policy_relative_bias.zero_()
            model.policy_relative_bias[2, 5] = 3.0
            _value, policy = model(torch.zeros(1, 15, 8, 8))
        self.assertEqual(policy.argmax(dim=1).item(), 2 * 64 + 5)
        self.assertEqual(policy[0, 2 * 64 + 5].item(), 3.0)

    def test_attention_head_is_materially_smaller(self):
        dense = small()
        attention = small(policy_head_type="attention",
                          policy_attention_channels=32)
        dense_params = sum(p.numel() for p in dense.parameters())
        attention_params = sum(p.numel() for p in attention.parameters())
        self.assertLess(attention_params, dense_params / 10)

    def test_side_adapter_requires_attention_and_starts_as_zero_residual(self):
        with self.assertRaisesRegex(ValueError, "requires the attention"):
            small(side_policy_adapters=True)
        model = small(policy_head_type="attention",
                      policy_attention_channels=6,
                      side_policy_adapters=True)
        self.assertTrue(model.side_policy_adapters)
        self.assertTrue(torch.count_nonzero(
            model.policy_side_adapter.weight).item() == 0)
        self.assertTrue(torch.count_nonzero(
            model.policy_side_adapter.bias).item() == 0)

    def test_side_adapter_separates_white_and_black_policy_logits(self):
        model = small(policy_head_type="attention",
                      policy_attention_channels=2,
                      side_policy_adapters=True).eval()
        backbone = torch.zeros(2, model.backbone_out_channels, 8, 8)
        backbone[:, 0, 0, 2] = 1.0
        with torch.no_grad():
            model.policy_attention.weight.zero_()
            model.policy_attention.bias.zero_()
            model.policy_relative_bias.zero_()
            model.policy_side_adapter.weight.zero_()
            model.policy_side_adapter.bias.zero_()
            model.policy_side_adapter.weight[4, 0, 0, 0] = 1.0
            logits = model._policy_logits(
                backbone, torch.tensor([1.0, -1.0]))
        move_from_2 = 2 * 64
        self.assertGreater(logits[0, move_from_2].item(),
                           logits[1, move_from_2].item())

    def test_checkpoint_geometry_is_inferred_and_round_trips(self):
        model = small(policy_head_type="attention",
                      policy_attention_channels=12).eval()
        self.assertEqual(
            infer_policy_head_config(model.state_dict()),
            ("attention", 32, 12),
        )
        x = torch.randn(2, 15, 8, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            torch.save(model.state_dict(), path)
            loaded, _ = load_model_for_inference(path, torch.device("cpu"))
        self.assertEqual(loaded.policy_head_type, "attention")
        self.assertEqual(loaded.policy_attention_channels, 12)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x)[1], loaded(x)[1]))

    def test_side_adapter_checkpoint_is_inferred_and_round_trips(self):
        model = small(policy_head_type="attention",
                      policy_attention_channels=8,
                      side_policy_adapters=True).eval()
        self.assertTrue(infer_side_policy_adapters(model.state_dict()))
        x = torch.randn(2, 15, 8, 8)
        x[0, 12] = 1.0
        x[1, 12] = -1.0
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            torch.save(model.state_dict(), path)
            loaded, _ = load_model_for_inference(path, torch.device("cpu"))
        self.assertTrue(loaded.side_policy_adapters)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x)[1], loaded(x)[1]))


if __name__ == "__main__":
    unittest.main()
