"""Overnight variant contracts (CPU only): one declared change each, new dirs, non-colliding seeds."""
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gen52_campaign as g52
import gen52_large_campaign as L
import gen52_poolcap_campaign as c
import gen52_ramp_campaign as R
import gen52_variant_campaign as V
import gen53_campaign as g53
import top_round_robin as top


def changed_flags(original, ours):
    assert len(original) == len(ours)
    return sorted(original[i - 1] for i, (a, b) in enumerate(zip(original, ours)) if a != b)


@pytest.mark.local_artifacts
def test_lr_changes_only_data_and_model_dirs_from_arm_l():
    for smoke in (False, True):
        parent = V.read(V.root(smoke, L.RUN) / "receipts/train_l.json")["command"]
        ours = V.train_command("lr", smoke)
        assert changed_flags(parent, ours) == ["--data-dir", "--model-dir"]
        assert ours[ours.index("--data-dir") + 1] == str(R.arm_r_paths(smoke)[2]["replay"])


@pytest.mark.local_artifacts
def test_l2_changes_only_seed_and_model_dir_from_arm_l():
    for smoke in (False, True):
        parent = V.read(V.root(smoke, L.RUN) / "receipts/train_l.json")["command"]
        ours = V.train_command("l2", smoke)
        assert changed_flags(parent, ours) == ["--model-dir", "--seed"]
        assert int(ours[ours.index("--seed") + 1]) == int(parent[parent.index("--seed") + 1]) + 1
    assert V.train_command("l2", False)[V.train_command("l2", False).index("--seed") + 1] == "3174"


def test_variants_write_new_model_directories():
    taken = {L.arm_l_paths(False)[2]["model_dir"], R.arm_r_paths(False)[2]["model_dir"]}
    _, extra = g52.paths_for(False)
    taken |= {extra["a"]["model_dir"], extra["b"]["model_dir"]}
    dirs = {V.model_dir(v, False) for v in V.VARIANTS}
    assert len(dirs) == 2 and not dirs & taken


def test_seed_blocks_do_not_collide():
    blocks = [V.layout(v, s)["h2h_seed"] for v in V.VARIANTS for s in (False, True)]
    others = [c.layout(s)["h2h_seed"] for s in (False, True)] + [L.layout(s)["h2h_seed"] for s in (False, True)] \
        + [R.layout(s)["h2h_seed"] for s in (False, True)]
    for b in blocks:
        assert b + 3_000_000 < 2 ** 32                      # primary, deep, secondary
        for o in others + [x for x in blocks if x != b]:
            assert abs(b - o) >= 3_000_000
        for other in (g52.layout(False), g53.layout(False)):
            assert not other["par_seed"] <= b < other["arms_seed"] + 2_000_000
        assert not top.SEED_BASE <= b < top.SEED_BASE + 28 * top.SEED_STRIDE


def test_verdict_rule():
    def audit(score, se, unique):
        return {"diagnostics": {"sampled": {"n": 400, "score": score, "se": se}, "unique": {"n": 80, "score": unique}}}
    assert V.verdict(audit(0.56, 0.02, 0.53), 0.92, 0.925)["call"] == "helps"
    assert V.verdict(audit(0.56, 0.02, 0.53), 0.91, 0.925)["call"] == "null"
    assert V.verdict(audit(0.44, 0.02, 0.45), 0.95, 0.925)["call"] == "hurts"


def test_top_round_robin_schedule_and_rating_reference():
    names = [n for n, _ in top.PLAYERS]
    pairs = top.schedule(names)
    assert len(pairs) == 28 == len({frozenset(p) for p in pairs})
    assert top.SEED_BASE + 28 * top.SEED_STRIDE < 2 ** 32
    rows = [dict(a="v29", b="v28", games=100, wins=60, draws=10, losses=30)]
    r = top.rate(["v29", "v28"], rows, reps=20)
    assert r["v29"]["elo"] == 0.0 and r["v28"]["elo"] < 0
