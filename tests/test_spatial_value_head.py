import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from train import (  # noqa: E402
    build_model, infer_spatial_value_head_config, load_model_for_inference,
)


def _small(**kw):
    """Tiny net so these stay fast; head geometry is what is under test."""
    return build_model(input_channels=15, stem_channels=8,
                       residual_block_channels=(8, 8), **kw)


class SpatialValueHeadContracts(unittest.TestCase):
    def test_gap_head_is_still_the_default(self):
        """The spatial head is an A/B candidate, never silently promoted."""
        self.assertFalse(_small().spatial_value_head)

    def test_gap_head_pools_away_the_board(self):
        head = _small().value_head
        self.assertIsInstance(head[0], nn.AdaptiveAvgPool2d)

    def test_spatial_head_preserves_the_8x8_layout(self):
        head = _small(spatial_value_head=True).value_head
        self.assertIsInstance(head[0], nn.Conv2d)
        self.assertFalse(
            any(isinstance(m, nn.AdaptiveAvgPool2d) for m in head),
            "spatial value head must not average the board away",
        )
        # the FC must see conv_channels * 64 squares, not conv_channels
        linear = next(m for m in head if isinstance(m, nn.Linear))
        self.assertEqual(linear.in_features, 32 * 64)

    def test_both_variants_produce_the_same_output_shapes(self):
        x = torch.zeros(3, 15, 8, 8)
        for spatial in (False, True):
            v, p = _small(spatial_value_head=spatial)(x)
            self.assertEqual(tuple(v.shape), (3, 1))
            self.assertEqual(tuple(p.shape), (3, 4096))

    def test_variant_is_inferable_from_the_state_dict(self):
        gap = _small().state_dict()
        spatial = _small(spatial_value_head=True).state_dict()
        self.assertEqual(infer_spatial_value_head_config(gap), (False, 32))
        self.assertEqual(infer_spatial_value_head_config(spatial), (True, 32))
        # the marker is the presence of a 4-D conv weight at index 0
        self.assertNotIn("value_head.0.weight", gap)
        self.assertIn("value_head.0.weight", spatial)

    def test_spatial_checkpoint_round_trips_through_inference_loader(self):
        model = _small(spatial_value_head=True).eval()
        x = torch.randn(2, 15, 8, 8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            torch.save(model.state_dict(), path)
            loaded, _ = load_model_for_inference(path, torch.device("cpu"))
        self.assertTrue(loaded.spatial_value_head)
        loaded.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x)[0], loaded(x)[0]))

    def test_custom_conv_width_survives_a_round_trip(self):
        model = _small(spatial_value_head=True, value_head_conv_channels=8)
        self.assertEqual(
            infer_spatial_value_head_config(model.state_dict()), (True, 8))

    def test_rejects_a_nonpositive_conv_width(self):
        with self.assertRaises(ValueError):
            _small(spatial_value_head=True, value_head_conv_channels=0)


if __name__ == "__main__":
    unittest.main()
