import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import data_processor as dp  # noqa: E402
from config import POLICY_SIZE  # noqa: E402
from train import (mask_policy_logits, unpack_legal_policy_mask,  # noqa: E402
                   weighted_policy_cross_entropy)


def packed(indices):
    mask = np.zeros((1, POLICY_SIZE), dtype=np.uint8)
    mask[0, indices] = 1
    return torch.from_numpy(np.packbits(mask, axis=1))


class LegalPolicyMaskContracts(unittest.TestCase):
    @staticmethod
    def _corrupt_record():
        return {
            "fen": "4k3/4p3/8/8/8/8/8/4K3 b - - 0 1",
            "current_player": "white",
            "half": 1,
            "mcts_value": 0.0,
            "game_result": 1.0,
            "policy": {"e7e6": 1.0},
        }

    def test_packbits_round_trip_uses_numpy_bit_order(self):
        indices = [0, 1, 7, 8, 4095]
        out = unpack_legal_policy_mask(packed(indices))
        self.assertEqual(torch.where(out[0])[0].tolist(), indices)

    def test_illegal_logits_do_not_affect_loss_or_receive_gradient(self):
        logits = torch.zeros((1, POLICY_SIZE), requires_grad=True)
        with torch.no_grad():
            logits[0, 2] = 100.0  # illegal and otherwise dominant
        targets = torch.zeros_like(logits)
        targets[0, 0] = 1.0
        mask = packed([0, 1])
        loss = weighted_policy_cross_entropy(
            logits, targets, torch.ones(1), legal_masks_packed=mask)
        self.assertAlmostEqual(loss.item(), np.log(2), places=5)
        loss.backward()
        self.assertEqual(logits.grad[0, 2].item(), 0.0)

    def test_illegal_target_mass_is_rejected(self):
        logits = torch.zeros((1, POLICY_SIZE))
        targets = torch.zeros_like(logits)
        targets[0, 2] = 1.0
        with self.assertRaises(ValueError):
            weighted_policy_cross_entropy(
                logits, targets, torch.ones(1),
                legal_masks_packed=packed([0, 1]))

    def test_data_mask_contains_policy_target_before_and_after_mirror(self):
        rec = {
            "fen": "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
            "current_player": "black",
            "half": 0,
            "mcts_value": 0.0,
            "game_result": -1.0,
            "policy": {"e8d8": 1.0},
        }
        out = dp._convert_games_to_arrays(
            [{"records": [rec]}], augment=True,
            include_legal_masks=True)
        policies = out[3]
        masks = np.unpackbits(out[-1], axis=1, count=POLICY_SIZE).astype(bool)
        for target, legal in zip(policies, masks):
            self.assertTrue(np.all(legal[target > 0]))
            self.assertGreater(int(legal.sum()), 0)

    def test_masked_argmax_ignores_illegal_move(self):
        logits = torch.zeros((1, POLICY_SIZE))
        logits[0, 2] = 100
        logits[0, 1] = 2
        masked, _ = mask_policy_logits(logits, packed([0, 1]))
        self.assertEqual(masked.argmax(dim=1).item(), 1)

    def test_processing_masks_a_corrupt_enabled_policy_row_and_counts_it(self):
        stats = {}
        out = dp._convert_games_to_arrays(
            [{"records": [self._corrupt_record()]}], augment=False,
            include_legal_masks=True, conversion_stats=stats)
        self.assertEqual(out[4].tolist(), [0.0])
        self.assertEqual(stats["illegal_policy_targets_masked"], 1)

    def test_strict_processing_mode_rejects_a_corrupt_row(self):
        with self.assertRaises(ValueError):
            dp._convert_games_to_arrays(
                [{"records": [self._corrupt_record()]}], augment=False,
                include_legal_masks=True,
                mask_illegal_policy_targets=False)


if __name__ == "__main__":
    unittest.main()
