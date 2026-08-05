import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import TURN_LAYER  # noqa: E402
from train import (  # noqa: E402
    apply_black_policy_weight,
    build_model,
    configure_policy_head_only,
)


def positions_with_turns(*white_turns):
    positions = np.zeros((len(white_turns), 8, 8, TURN_LAYER + 1), dtype=np.float32)
    for index, white_turn in enumerate(white_turns):
        positions[index, 0, 0, TURN_LAYER] = 1.0 if white_turn else 0.0
    return positions


def test_black_policy_weight_scales_only_black_and_preserves_input():
    original = np.array([1.0, 0.5, 0.0, 2.0], dtype=np.float32)
    adjusted = apply_black_policy_weight(
        original, positions_with_turns(True, False, False, True), 1.25)

    np.testing.assert_array_equal(original, [1.0, 0.5, 0.0, 2.0])
    np.testing.assert_allclose(adjusted, [1.0, 0.625, 0.0, 2.0])


def test_black_policy_weight_one_is_exact_copy():
    original = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    adjusted = apply_black_policy_weight(
        original, positions_with_turns(False, True, False), 1.0)

    np.testing.assert_array_equal(adjusted, original)
    assert adjusted is not original


@pytest.mark.parametrize("weight", [0.0, -0.1])
def test_black_policy_weight_must_be_positive(weight):
    with pytest.raises(ValueError, match="black_weight must be > 0"):
        apply_black_policy_weight(
            np.ones(1, dtype=np.float32), positions_with_turns(False), weight)


def test_policy_head_only_freezes_backbone_and_value_head():
    model = build_model(policy_head_type="attention")
    trainable = configure_policy_head_only(model, enabled=True)

    assert set(trainable) == {
        "policy_relative_bias",
        "policy_attention.weight",
        "policy_attention.bias",
    }
    assert {name for name, parameter in model.named_parameters()
            if parameter.requires_grad} == set(trainable)
