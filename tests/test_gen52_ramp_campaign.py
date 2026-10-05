"""Arm R contracts (CPU only): only the deep-value labels change; new directories; non-colliding seeds."""
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gen52_campaign as g52
import gen52_large_campaign as L
import gen52_poolcap_campaign as c
import gen52_ramp_campaign as R
import gen53_campaign as g53


@pytest.mark.local_artifacts
def test_compose_swaps_only_the_deep_value_sources_and_output():
    for smoke in (False, True):
        original = R.read(R.gen52_root(smoke) / "receipts/compose_b.json")["command"]
        ours = R.compose_command(smoke)
        assert len(original) == len(ours)
        diff = [(a, b) for a, b in zip(original, ours) if a != b]
        deep = R.deep_sources(smoke)
        assert len(diff) == len(deep) + 1          # every deep source, plus --output-dir
        for a, b in diff:
            if "=" in a:
                name, path = a.split("=", 1)
                assert name.endswith("_deepvalue") and b == f"{name}={R.ramped(path)}"
    assert [n for n, _ in R.deep_sources(False)] == ["gen_0051_deepvalue", "gen_0052_deepvalue"]


@pytest.mark.local_artifacts
def test_train_command_changes_only_directories():
    for smoke in (False, True):
        original = R.read(R.gen52_root(smoke) / "receipts/train_b.json")["command"]
        ours = R.train_command(smoke)
        diff = [i for i, (a, b) in enumerate(zip(original, ours)) if a != b]
        assert sorted(original[i - 1] for i in diff) == ["--data-dir", "--model-dir"]
        assert ours[ours.index("--res-channels") + 1] == L.BASE_TOWER


def test_new_directories_only():
    _, extra, arm = R.arm_r_paths(False)
    assert arm["model_dir"].name.endswith("_ramp") and arm["replay"].name.endswith("_armR")
    assert arm["model_dir"] not in (extra["a"]["model_dir"], extra["b"]["model_dir"])
    assert arm["replay"] not in (extra["a"]["replay"], extra["b"]["replay"])
    for _, p in R.deep_sources(False):
        assert R.ramped(p).name.endswith("_ramped") and R.ramped(p) != p


def test_head_to_head_seeds_do_not_collide():
    for smoke in (False, True):
        seed = R.layout(smoke)["h2h_seed"]
        for other in (g52.layout(smoke), g53.layout(smoke)):
            assert not other["par_seed"] <= seed < other["arms_seed"] + 2_000_000
        for other in (c.layout(smoke)["h2h_seed"], L.layout(smoke)["h2h_seed"]):
            assert abs(seed - other) >= 2_000_000
        assert seed + 2_000_000 < 2 ** 32
