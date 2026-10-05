"""Arm L contracts (CPU only): the one declared change, matched seeds, fixed verdict rule."""
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gen52_campaign as g52
import gen52_large_campaign as L
import gen52_poolcap_campaign as c
import gen53_campaign as g53


@pytest.mark.local_artifacts
def test_train_command_changes_only_model_dir_and_tower():
    for smoke in (False, True):
        receipt = L.read(L.gen52_root(smoke) / "receipts/train_b.json")["command"]
        ours = L.train_command(smoke)
        diff = [i for i, (a, b) in enumerate(zip(receipt, ours)) if a != b]
        assert len(receipt) == len(ours)
        assert sorted(receipt[i - 1] for i in diff) == ["--model-dir", "--res-channels"]
        assert ours[ours.index("--res-channels") + 1] == L.WIDE_TOWER
        assert ours[ours.index("--seed") + 1] == receipt[receipt.index("--seed") + 1]
    assert L.train_command(False)[L.train_command(False).index("--seed") + 1] == "3173"


def test_arm_l_writes_a_new_model_directory_and_reuses_arm_b_replay():
    _, extra, arm = L.arm_l_paths(False)
    assert arm["model_dir"].name.endswith("_large") and not arm["model_dir"].exists() or \
        (L.RUN / "production/receipts/train_l.json").exists()
    assert arm["model_dir"] not in (extra["a"]["model_dir"], extra["b"]["model_dir"])
    assert arm["replay"] == extra["b"]["replay"]


def test_head_to_head_seeds_do_not_collide():
    for smoke in (False, True):
        seed = L.layout(smoke)["h2h_seed"]
        for other in (g52.layout(smoke), g53.layout(smoke)):
            assert not other["par_seed"] <= seed < other["arms_seed"] + 2_000_000
        assert abs(seed - c.layout(smoke)["h2h_seed"]) >= 2_000_000
        assert seed + 2_000_000 < 2 ** 32
    assert L.layout(False)["h2h_seed"] + 2_000_000 <= L.layout(True)["h2h_seed"]


def audit(score, se, unique):
    return {"diagnostics": {"sampled": {"n": 400, "score": score, "se": se}, "unique": {"n": 80, "score": unique}}}


def test_verdict_rule_matches_the_plan():
    assert L.verdict(audit(0.56, 0.02, 0.53), 0.91)["call"] == "capacity_helps"
    assert L.verdict(audit(0.56, 0.02, 0.49), 0.91)["call"] == "null"      # repeated openings only
    assert L.verdict(audit(0.56, 0.02, 0.53), 0.90)["call"] == "null"      # held-out regression
    assert L.verdict(audit(0.52, 0.02, 0.55), 0.95)["call"] == "null"      # interval includes 50%
    assert L.verdict(audit(0.44, 0.02, 0.45), 0.95)["call"] == "capacity_harmful"
