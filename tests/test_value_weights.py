"""Value weighting: the pipeline that makes "teach policy, not value" expressible.

HANDOFF SS7.1 posed the ps_monster question as a fork -- is the pawn-phase cliff
a knowledge problem or a belief problem? -- and noted it could not be split,
because the pipeline had policy weights and no value equivalent. This is that
equivalent (DIRECTIVE D1).

The two guarantees, both load-bearing:

1. **Default is exactly the old behaviour.** Every corpus and every checkpoint
   predates this. If an all-ones weight vector changed the loss by so much as a
   float, every v17/v18 number on record would stop being comparable.
2. **Weight 0 means zero value gradient.** Not "small", not "downweighted" --
   a record excluded from value teaching must not move the value head at all,
   through the scalar head or the WDL head.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as train_mod  # noqa: E402
from data_processor import value_weight_for_record  # noqa: E402


class TestRecordWeighting(unittest.TestCase):
    def test_default_is_one_for_every_ordinary_record(self):
        for rec in (
            {"source": "human_game", "actor": "human", "game_result": -1},
            {"source": "playstrategy", "actor": "human", "game_result": 1},
            {"source": "selfplay", "actor": "ai", "game_result": 0},
            {},
        ):
            self.assertEqual(value_weight_for_record(rec), 1.0, rec)

    def test_explicit_weight_wins(self):
        self.assertEqual(value_weight_for_record({"value_weight": 0.0}), 0.0)
        self.assertEqual(value_weight_for_record({"value_weight": 0.5}), 0.5)
        # Explicit weight overrides any source-based intuition.
        self.assertEqual(
            value_weight_for_record({"source": "human_game", "value_weight": 0}), 0.0)

    def test_value_weight_is_independent_of_policy_weight(self):
        from data_processor import policy_weight_for_record
        # A ps_monster-shaped record: a policy teacher, not a value teacher.
        rec = {"source": "playstrategy", "actor": "human", "game_result": 1,
               "current_player": "white", "value_weight": 0.0}
        self.assertEqual(policy_weight_for_record(rec), 1.0)
        self.assertEqual(value_weight_for_record(rec), 0.0)


class TestPowerLossDefaultUnchanged(unittest.TestCase):
    def test_all_ones_reproduces_the_unweighted_loss_bitwise(self):
        torch.manual_seed(0)
        pred = torch.randn(64, 1)
        target = torch.randn(64, 1).clamp(-1, 1)
        plain = train_mod._power_loss(pred, target)
        weighted = train_mod._power_loss(
            pred, target, weights=torch.ones(64))
        self.assertEqual(plain.item(), weighted.item())

    def test_none_weights_is_the_plain_mean(self):
        pred = torch.tensor([[0.5], [-0.5]])
        target = torch.tensor([[0.0], [0.0]])
        expected = torch.pow(torch.abs(pred - target),
                             train_mod.VALUE_LOSS_EXPONENT).mean()
        self.assertAlmostEqual(
            train_mod._power_loss(pred, target).item(), expected.item(), places=6)


class TestZeroWeightMeansZeroGradient(unittest.TestCase):
    def test_masked_record_contributes_no_value_gradient(self):
        # Two records; the second is excluded. Its prediction is wildly wrong,
        # so if it leaked into the loss at all the gradient would move.
        pred = torch.tensor([[0.10], [0.90]], requires_grad=True)
        target = torch.tensor([[0.00], [-1.00]])
        weights = torch.tensor([1.0, 0.0])
        train_mod._power_loss(pred, target, weights=weights).backward()
        self.assertGreater(abs(pred.grad[0, 0].item()), 0.0)
        self.assertEqual(pred.grad[1, 0].item(), 0.0)

    def test_masked_record_cannot_change_the_loss_value(self):
        pred = torch.tensor([[0.10], [0.90]])
        target_a = torch.tensor([[0.00], [-1.00]])
        target_b = torch.tensor([[0.00], [+1.00]])  # different masked label
        weights = torch.tensor([1.0, 0.0])
        a = train_mod._power_loss(pred, target_a, weights=weights)
        b = train_mod._power_loss(pred, target_b, weights=weights)
        self.assertEqual(a.item(), b.item())

    def test_weighting_matches_training_on_the_kept_subset(self):
        # Masking record 2 must equal training on record 1 alone.
        pred = torch.tensor([[0.10], [0.90]])
        target = torch.tensor([[0.00], [-1.00]])
        masked = train_mod._power_loss(
            pred, target, weights=torch.tensor([1.0, 0.0]))
        alone = train_mod._power_loss(pred[:1], target[:1])
        self.assertAlmostEqual(masked.item(), alone.item(), places=6)

    def test_all_zero_batch_is_finite_and_backpropagates(self):
        pred = torch.tensor([[0.3], [0.4]], requires_grad=True)
        target = torch.tensor([[1.0], [-1.0]])
        loss = train_mod._power_loss(pred, target, weights=torch.zeros(2))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()  # must not raise: the graph stays connected
        self.assertTrue(torch.all(pred.grad == 0))

    def test_wdl_head_honours_value_weights_too(self):
        # Otherwise "policy only" leaks beliefs back in via --value-head wdl.
        logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
                              requires_grad=True)
        labels = torch.tensor([0, 0])  # second record predicted confidently wrong
        loss = train_mod._weighted_wdl_ce(logits, labels,
                                          torch.tensor([1.0, 0.0]))
        loss.backward()
        self.assertGreater(logits.grad[0].abs().sum().item(), 0.0)
        self.assertEqual(logits.grad[1].abs().sum().item(), 0.0)


class TestBatchUnpacking(unittest.TestCase):
    """Batches have grown twice; every historical shape must still unpack."""

    def test_legacy_shapes_default_both_weights_to_one(self):
        X = torch.zeros(3, 15, 8, 8)
        v = torch.zeros(3, 1)
        p = torch.zeros(3, 4096)
        for batch in ((X, v, p),
                      (X, v, p, torch.ones(3)),                    # + policy w
                      (X, v, p, torch.zeros(3, dtype=torch.long)),  # + wdl
                      ):
            _X, _v, _p, pw, vw, _wdl = train_mod._unpack_loader_batch(batch)
            self.assertTrue(torch.all(vw == 1.0), batch)
            self.assertEqual(pw.shape, (3,))

    def test_length_five_is_disambiguated_by_dtype(self):
        X, v, p, pw = (torch.zeros(2, 15, 8, 8), torch.zeros(2, 1),
                       torch.zeros(2, 4096), torch.ones(2))
        # float fifth -> value weights, no WDL
        _1, _2, _3, _4, vw, wdl = train_mod._unpack_loader_batch(
            (X, v, p, pw, torch.full((2,), 0.5)))
        self.assertIsNone(wdl)
        self.assertTrue(torch.all(vw == 0.5))
        # long fifth -> legacy WDL labels, value weights default to one
        _1, _2, _3, _4, vw, wdl = train_mod._unpack_loader_batch(
            (X, v, p, pw, torch.zeros(2, dtype=torch.long)))
        self.assertIsNotNone(wdl)
        self.assertTrue(torch.all(vw == 1.0))

    def test_full_six_tuple_passes_through(self):
        parts = (torch.zeros(2, 15, 8, 8), torch.zeros(2, 1), torch.zeros(2, 4096),
                 torch.ones(2), torch.full((2,), 0.25), torch.zeros(2, dtype=torch.long))
        out = train_mod._unpack_loader_batch(parts)
        self.assertEqual(len(out), 6)
        self.assertTrue(torch.all(out[4] == 0.25))


class TestLoaderAndLoading(unittest.TestCase):
    def test_loader_emits_value_weights(self):
        X = np.zeros((4, 8, 8, 15), dtype=np.float32)
        yv = np.zeros((4,), dtype=np.float32)
        yp = np.zeros((4, 4096), dtype=np.float32)
        loader = train_mod._make_loader(
            X, yv, yp, batch_size=4, shuffle=False,
            y_value_weight=np.array([1, 0, 1, 0], dtype=np.float32))
        batch = next(iter(loader))
        _X, _v, _p, _pw, vw, _wdl = train_mod._unpack_loader_batch(batch)
        self.assertEqual(vw.tolist(), [1.0, 0.0, 1.0, 0.0])

    def test_corpus_without_value_weights_file_loads_as_all_ones(self):
        # Every processed corpus on disk predates this file.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            n = 5
            np.save(Path(d) / "positions.npy", np.zeros((n, 8, 8, 15), np.float32))
            np.save(Path(d) / "mcts_values.npy", np.zeros((n,), np.float32))
            np.save(Path(d) / "game_results.npy", np.zeros((n,), np.float32))
            np.save(Path(d) / "policies.npy", np.zeros((n, 4096), np.float32))
            np.savez(Path(d) / "splits.npz", train=np.arange(n))
            out = train_mod.load_data(d)
            value_weights = out[5]
            self.assertEqual(len(value_weights), n)
            self.assertTrue(np.all(value_weights == 1.0))


if __name__ == "__main__":
    unittest.main()
