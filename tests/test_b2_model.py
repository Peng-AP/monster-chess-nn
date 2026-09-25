import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train import build_model, load_model_for_inference


@pytest.mark.parametrize('channels,blocks', [(15, 0), (24, 0), (24, 2)])
def test_roundtrip_and_backward(tmp_path, channels, blocks):
    torch.set_num_threads(1)
    model = build_model(input_channels=channels, attention_blocks=blocks,
                        stem_channels=16, residual_block_channels=(16, 16),
                        policy_head_type='attention', policy_attention_channels=8)
    x = torch.randn(2, channels, 8, 8)
    value, policy = model(x)
    assert value.shape == (2, 1) and policy.shape == (2, 4096)
    (value.square().mean() + policy.square().mean()).backward()
    assert all(p.grad is not None for p in model.parameters())
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    restored, _ = load_model_for_inference(path, 'cpu')
    model.eval()
    for a, b in zip(model(x), restored(x)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_missing_schema_rejected(tmp_path):
    model = build_model(input_channels=24, stem_channels=16,
                        residual_block_channels=(16,), policy_head_type='attention')
    state = model.state_dict()
    del state['_b2_schema']
    path = tmp_path / 'bad.pt'
    torch.save(state, path)
    with pytest.raises(ValueError, match='schema'):
        load_model_for_inference(path, 'cpu')
