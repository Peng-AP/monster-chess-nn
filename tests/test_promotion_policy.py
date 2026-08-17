import sys
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import POLICY_SIZE, PROMOTION_AWARE_POLICY_SIZE  # noqa: E402
from encoding import (  # noqa: E402
    mirror_move_index,
    mirror_policy,
    move_to_index,
    move_to_policy_index,
    policy_dict_to_target,
    promotion_move_to_index,
)
from train import (  # noqa: E402
    build_model,
    configure_promotion_head_only,
    infer_promotion_policy,
    load_state_dict_flexible,
    _set_training_mode,
    weighted_policy_cross_entropy,
)


def test_promotion_indices_are_distinct_and_legacy_remains_stable():
    moves = [chess.Move.from_uci(f"a7a8{piece}") for piece in "qrbn"]
    assert len({promotion_move_to_index(move) for move in moves}) == 4
    assert all(POLICY_SIZE <= promotion_move_to_index(move)
               < PROMOTION_AWARE_POLICY_SIZE for move in moves)
    assert len({move_to_index(move) for move in moves}) == 1
    assert move_to_policy_index(moves[0], False) == move_to_index(moves[0])


def test_promotion_target_keeps_search_distribution_instead_of_merging_it():
    policy = {"c7c8q": 0.10, "c7c8r": 0.20,
              "c7c8b": 0.30, "c7c8n": 0.40}
    legacy = policy_dict_to_target(policy, is_white=False)
    aware = policy_dict_to_target(policy, is_white=False, promotion_aware=True)
    assert legacy.shape == (POLICY_SIZE,)
    assert aware.shape == (PROMOTION_AWARE_POLICY_SIZE,)
    assert np.isclose(legacy[move_to_index(chess.Move.from_uci("c7c8q"))], 1.0)
    for piece, expected in zip("qrbn", (0.10, 0.20, 0.30, 0.40)):
        move = chess.Move.from_uci(f"c7c8{piece}")
        assert np.isclose(aware[promotion_move_to_index(move)], expected)


def test_promotion_mirror_is_a_bijection_and_preserves_piece_choice():
    for uci, mirrored in (("a7b8n", "h7g8n"), ("f2f1q", "c2c1q")):
        idx = promotion_move_to_index(chess.Move.from_uci(uci))
        expected = promotion_move_to_index(chess.Move.from_uci(mirrored))
        assert mirror_move_index(idx) == expected
        assert mirror_move_index(expected) == idx
    policy = np.arange(PROMOTION_AWARE_POLICY_SIZE, dtype=np.float32)
    np.testing.assert_array_equal(mirror_policy(mirror_policy(policy)), policy)


def test_zero_initialized_extension_exactly_lifts_legacy_logits():
    torch.manual_seed(7)
    legacy = build_model(
        input_channels=15, policy_head_type="attention",
        policy_attention_channels=8)
    lifted = build_model(
        input_channels=15, policy_head_type="attention",
        policy_attention_channels=8, promotion_policy=True)
    loaded, skipped = load_state_dict_flexible(lifted, legacy.state_dict())
    assert loaded == len(legacy.state_dict())
    assert skipped == []
    legacy.eval()
    lifted.eval()
    x = torch.randn(2, 15, 8, 8)
    with torch.no_grad():
        _, old_logits = legacy(x)
        _, new_logits = lifted(x)
    assert tuple(new_logits.shape) == (2, PROMOTION_AWARE_POLICY_SIZE)
    torch.testing.assert_close(new_logits[:, :POLICY_SIZE], old_logits)
    for uci in ("a7a8q", "c7d8n", "h2g1r"):
        move = chess.Move.from_uci(uci)
        torch.testing.assert_close(
            new_logits[:, promotion_move_to_index(move)],
            old_logits[:, move_to_index(move)],
        )
    assert infer_promotion_policy(lifted.state_dict())
    assert not infer_promotion_policy(legacy.state_dict())


def test_ordinary_position_loss_is_identical_after_zero_lift():
    torch.manual_seed(11)
    base = torch.randn(3, POLICY_SIZE)
    lifted = torch.cat((base, torch.randn(3, 192)), dim=1)
    targets = torch.zeros(3, PROMOTION_AWARE_POLICY_SIZE)
    targets[0, 10] = 1
    targets[1, 20] = 1
    targets[2, 30] = 1
    weights = torch.ones(3)
    old_loss = weighted_policy_cross_entropy(
        base, targets[:, :POLICY_SIZE], weights)
    new_loss = weighted_policy_cross_entropy(lifted, targets, weights)
    torch.testing.assert_close(new_loss, old_loss)


def test_promotion_head_only_freezes_every_legacy_parameter():
    model = build_model(
        input_channels=15, policy_head_type="attention",
        policy_attention_channels=8, promotion_policy=True)
    names = configure_promotion_head_only(model, True)
    assert names == ["policy_promotion_delta.weight",
                     "policy_promotion_delta.bias"]
    assert all(parameter.requires_grad == name.startswith("policy_promotion_delta.")
               for name, parameter in model.named_parameters())
    _set_training_mode(model)
    assert model.training
    assert all(not module.training for module in model.modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    assert model.policy_promotion_delta.training
