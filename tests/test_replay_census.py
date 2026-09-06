import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from replay_census import weight_summary


def test_census_distinguishes_repeated_rows_and_policy_only_weight():
    out = weight_summary(np.array([2, 0, 1, 3]), np.array([1., 1., 4., 4.]),
                         np.array([1., 1., 0., 0.]))
    assert out["scheduled_rows"] == 6
    assert out["distinct_array_rows"] == 3
    assert out["policy_only_scheduled_rows"] == 4
    assert out["policy_weight_mass"] == 18
    assert out["policy_only_weight_mass"] == 16
    assert out["policy_only_weight_fraction"] == 16 / 18
    assert out["value_weight_mass"] == 2


def test_census_attributes_data_to_generator_not_source_generation_number(tmp_path):
    import json
    import pytest
    from replay_census import census
    data = tmp_path / "replay"
    data.mkdir()
    source = tmp_path / "gen44"
    np.save(data / "policy_weights.npy", np.array([1., 4., 1.]))
    np.save(data / "value_weights.npy", np.array([1., 0., 1.]))
    np.savez(data / "splits.npz", train=np.array([0, 1, 1]), val=np.array([2]), test=np.array([], dtype=int))
    (data / "replay_manifest.json").write_text(json.dumps({"rows": 3,
        "split_rows": {"train": 3, "val": 1, "test": 0},
        "sources": [{"name": "gen_0044", "path": str(source), "rows": 3}]}))
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"entries": [{"generation": 44, "path": str(source),
        "rows": 3, "incumbent": "gen42.pt", "incumbent_sha256": "recorded_hash"}]}))
    out = census(str(data), str(registry))
    assert out["sources"][0]["generator"] == "gen42.pt"
    assert out["splits"]["train"]["policy_only_weight_mass"] == 8
    assert out["splits"]["train"]["distinct_array_rows"] == 2
    registry.write_text('{"entries": []}')
    with pytest.raises(ValueError, match="not in accepted"):
        census(str(data), str(registry))
