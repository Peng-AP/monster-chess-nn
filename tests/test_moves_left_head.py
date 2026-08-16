import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import data_processor as dp  # noqa: E402
from train import (build_model, infer_moves_left_head_config,  # noqa: E402
                   load_model_for_inference, weighted_moves_left_huber,
                   configure_moves_left_head_only)
from evaluation import NNEvaluator  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from native_mcts import NativeMCTS, make_bridge  # noqa: E402


FEN = "8/8/8/8/8/8/8/K6k b - - 0 1"


def record(plies_to_end, result=-1.0, value_weight=1.0, segment=0):
    return {
        "fen": FEN,
        "current_player": "black",
        "mcts_value": 0.0,
        "game_result": result,
        "policy": {"h1h2": 1.0},
        "value_weight": value_weight,
        "plies_to_end": plies_to_end,
        "segment": segment,
    }


def small(**kwargs):
    return build_model(input_channels=15, stem_channels=8,
                       residual_block_channels=(8, 8), **kwargs)


class MovesLeftDataContracts(unittest.TestCase):
    def test_explicit_distance_includes_the_current_decision(self):
        records = [record(2), record(1), record(0)]
        self.assertEqual(dp._moves_left_targets(records).tolist(), [3, 2, 1])

    def test_segment_fallback_resets_distance(self):
        records = [
            dict(record(0, segment=0), plies_to_end=1),
            dict(record(0, segment=0), plies_to_end=0),
            dict(record(0, segment=1), plies_to_end=1),
            dict(record(0, segment=1), plies_to_end=0),
        ]
        self.assertEqual(dp._moves_left_targets(records).tolist(), [2, 1, 2, 1])

    def test_draws_and_policy_only_sources_are_masked(self):
        self.assertEqual(dp.moves_left_weight_for_record(record(0, result=0)), 0.0)
        self.assertEqual(dp.moves_left_weight_for_record(record(0, result=-0.5)), 0.0)
        self.assertEqual(dp.moves_left_weight_for_record(
            record(0, result=-1, value_weight=0)), 0.0)
        explicit = dict(record(0, result=0), moves_left_weight=0.25)
        self.assertEqual(dp.moves_left_weight_for_record(explicit), 0.25)

    def test_conversion_keeps_targets_aligned_through_mirroring(self):
        game = {"records": [record(1), record(0)]}
        out = dp._convert_games_to_arrays(
            [game], augment=True, include_moves_left=True)
        moves_left, weights = out[-2:]
        self.assertEqual(moves_left.tolist(), [2, 2, 1, 1])
        self.assertEqual(weights.tolist(), [1, 1, 1, 1])


class MovesLeftModelContracts(unittest.TestCase):
    def test_head_is_optional_and_normal_forward_stays_two_headed(self):
        base = small()
        aux = small(use_moves_left_head=True, moves_left_head_channels=12)
        self.assertFalse(base.use_moves_left_head)
        value, policy = aux(torch.zeros(2, 15, 8, 8))
        self.assertEqual(tuple(value.shape), (2, 1))
        self.assertEqual(tuple(policy.shape), (2, 4096))
        _v, _p, _wdl, moves_left = aux.forward_with_aux(
            torch.zeros(2, 15, 8, 8))
        self.assertEqual(tuple(moves_left.shape), (2, 1))
        self.assertTrue(torch.all(moves_left >= 0))

    def test_checkpoint_inference_detects_and_round_trips_head(self):
        model = small(use_moves_left_head=True, moves_left_head_channels=12).eval()
        self.assertEqual(
            infer_moves_left_head_config(model.state_dict()), (True, 12))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            torch.save(model.state_dict(), path)
            loaded, _ = load_model_for_inference(path, torch.device("cpu"))
        self.assertTrue(loaded.use_moves_left_head)
        self.assertEqual(loaded.moves_left_head_channels, 12)

    def test_weighted_huber_masks_untrusted_rows(self):
        predictions = torch.tensor([[1.0], [100.0]], requires_grad=True)
        targets = torch.tensor([[2.0], [0.0]])
        weights = torch.tensor([1.0, 0.0])
        loss = weighted_moves_left_huber(predictions, targets, weights)
        self.assertAlmostEqual(loss.item(), 0.5)
        loss.backward()
        self.assertEqual(predictions.grad[1].item(), 0.0)

    def test_head_only_mode_freezes_every_existing_parameter_and_batchnorm(self):
        model = small(use_moves_left_head=True)
        names = configure_moves_left_head_only(model, True)
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("moves_left_head.") for name in names))
        self.assertTrue(getattr(model, "_freeze_backbone_batchnorm", False))
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad,
                             name.startswith("moves_left_head."))

    def test_search_inference_is_opt_in_and_legacy_compatible(self):
        state = MonsterChessGame()
        with tempfile.TemporaryDirectory() as directory:
            aux_path = os.path.join(directory, "aux.pt")
            base_path = os.path.join(directory, "base.pt")
            torch.save(small(use_moves_left_head=True).eval().state_dict(), aux_path)
            torch.save(small().eval().state_dict(), base_path)

            aux_eval = NNEvaluator(aux_path)
            value, policy = aux_eval.evaluate_with_policy(state)
            value3, policy3, moves_left = (
                aux_eval.evaluate_with_policy_and_moves_left(state))
            self.assertAlmostEqual(value, value3, places=5)
            np.testing.assert_allclose(policy, policy3, rtol=1e-5, atol=1e-5)
            self.assertIsNotNone(moves_left)
            self.assertGreaterEqual(moves_left, 0.0)

            base_eval = NNEvaluator(base_path)
            _value, _policy, base_moves_left = (
                base_eval.evaluate_with_policy_and_moves_left(state))
            self.assertIsNone(base_moves_left)

    def test_native_bridge_adds_a_third_buffer_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "aux.pt")
            torch.save(small(use_moves_left_head=True).eval().state_dict(), path)
            evaluator = NNEvaluator(path)
            tensor = evaluator.fen_to_tensor(
                MonsterChessGame().fen(), is_white_turn=True,
                half_pending=False, input_channels=evaluator.input_channels)
            buf = tensor.transpose(2, 0, 1)[None].astype(np.float32).tobytes()
            legacy_bridge, channels = make_bridge(
                evaluator, include_moves_left=False)
            mlh_bridge, _ = make_bridge(evaluator, include_moves_left=True)
            self.assertEqual(len(legacy_bridge(buf, 1, channels)), 2)
            result = mlh_bridge(buf, 1, channels)
            self.assertEqual(len(result), 3)
            self.assertEqual(len(result[2]), 4)

    def test_search_flag_rejects_a_checkpoint_without_the_head(self):
        with tempfile.TemporaryDirectory() as directory:
            aux_path = os.path.join(directory, "aux.pt")
            base_path = os.path.join(directory, "base.pt")
            torch.save(small(use_moves_left_head=True).eval().state_dict(), aux_path)
            torch.save(small().eval().state_dict(), base_path)
            enabled = NativeMCTS(
                eval_fn=NNEvaluator(aux_path), moves_left_utility=True)
            self.assertTrue(enabled.moves_left_utility)
            with self.assertRaises(ValueError):
                NativeMCTS(eval_fn=NNEvaluator(base_path),
                           moves_left_utility=True)


if __name__ == "__main__":
    unittest.main()
