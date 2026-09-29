"""Arm C contracts (CPU only): the one declared change, matched seeds, new namespaces."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gen52_poolcap_campaign as c
import gen52_campaign as g52
import gen53_campaign as g53


def test_arm_c_drops_only_the_rolled_forward_gen51_source():
    names = [n for n, _ in c.sources(False)]
    assert names == ["gen_0052_deepvalue", "gen_0052_pool"] and c.DROPPED not in names
    decision = g52.decision()
    rolled = [s["name"] for s in decision["prior_extra_sources"]]
    assert rolled == [c.DROPPED]


def test_arm_c_writes_new_directories_only():
    _, extra, arm = c.arm_c_paths(False)
    assert arm["model_dir"].name.endswith("_poolcap") and arm["replay"].name.endswith("_armC")
    assert arm["model_dir"] not in (extra["a"]["model_dir"], extra["b"]["model_dir"])
    assert arm["replay"] not in (extra["a"]["replay"], extra["b"]["replay"])


def test_head_to_head_seeds_do_not_collide_with_other_campaigns():
    for smoke in (False, True):
        seed = c.layout(smoke)["h2h_seed"]
        for other in (g52.layout(smoke), g53.layout(smoke)):
            assert not other["par_seed"] <= seed < other["arms_seed"] + 2_000_000
    assert c.layout(False)["h2h_seed"] + 2_000_000 <= c.layout(True)["h2h_seed"]
