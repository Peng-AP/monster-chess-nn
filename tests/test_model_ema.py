import os
import sys
import unittest

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from train import ModelEMA  # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self.bn = nn.BatchNorm1d(1)


class ModelEMAContracts(unittest.TestCase):
    def test_update_averages_floats_and_copies_integer_buffers(self):
        model = Tiny()
        with torch.no_grad():
            model.linear.weight.fill_(2.0)
        ema = ModelEMA(model, decay=0.75)
        with torch.no_grad():
            model.linear.weight.fill_(6.0)
            model.bn.num_batches_tracked.fill_(9)
        ema.update(model)
        self.assertTrue(torch.allclose(
            ema.module.linear.weight, torch.full_like(model.linear.weight, 3.0)))
        self.assertEqual(ema.module.bn.num_batches_tracked.item(), 9)

    def test_shadow_parameters_do_not_require_grad(self):
        ema = ModelEMA(Tiny(), decay=0.9)
        self.assertFalse(any(p.requires_grad for p in ema.module.parameters()))

    def test_invalid_decay_is_rejected(self):
        for decay in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                ModelEMA(Tiny(), decay)


if __name__ == "__main__":
    unittest.main()
