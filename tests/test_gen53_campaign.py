"""Gen53 campaign contracts (CPU only): declared changes from gen52, pool switch, seeds."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gen53_campaign as g
import gen52_campaign as g52
import gen51_campaign as g51
import search_constants_campaign as stage1


def recipe(name):
    return json.loads((ROOT / "tools/recipes" / name).read_text())


def test_pool_membership_and_held_out_models_are_unchanged():
    assert g.POOL == g52.POOL and g.HELD_OUT == g52.HELD_OUT
    for name in ("gen53_pool.json", "gen53_pool_rehearsal.json"):
        assert recipe(name)["opponents"] == g.POOL and not set(recipe(name)["opponents"]) & set(g.HELD_OUT)


def test_recipes_differ_from_gen52_only_in_seed():
    for a, b in (("gen52.json", "gen53.json"), ("gen52_pool.json", "gen53_pool.json")):
        old, new = recipe(a), recipe(b)
        changed = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
        assert changed <= {"model", "seed"} and "seed" in changed


def test_seed_blocks_are_disjoint_across_all_campaigns():
    blocks = []
    for smoke in (False, True):
        s1, s51, s52, s53 = stage1.layout(smoke), g51.layout(smoke), g52.layout(smoke), g.layout(smoke)
        blocks += [(s1["par_seed"], s1["confirm_seed"] + 500_000),
                   (s51["continuation_seed"], s51["arms_seed"] + 2_000_000),
                   (s52["par_seed"], s52["arms_seed"] + 2_000_000),
                   (s53["par_seed"], s53["arms_seed"] + 2_000_000)]
    blocks.sort()
    assert all(a[1] <= b[0] for a, b in zip(blocks, blocks[1:])), blocks
    assert max(b for _, b in blocks) < 2 ** 32
    seeds = sorted(recipe(n)["seed"] for n in ("gen52.json", "gen53.json", "gen52_pool.json", "gen53_pool.json"))
    assert all(b - a >= 1_000_000 for a, b in zip(seeds, seeds[1:]))


def test_generation_number_and_namespaces_advance():
    assert g.layout(False)["generation"] == 53
    _, extra = g.paths_for(False)
    assert all("0053" in p.name for p in (extra["a"]["model_dir"], extra["b"]["model_dir"],
                                          extra["a"]["replay"], extra["b"]["replay"], extra["pool"], extra["deep"]))


def test_revised_decision_uses_arm_r_with_ramped_deep_labels():
    d = g.decision()
    assert d["teacher"].endswith("bootstrap_main_gen_0052_ramp/arena_selected.pt")
    assert d["deep_value"] and d["deep_value_labels"] == "ramped" and d["pool"] is False
    assert [s["name"] for s in d["prior_extra_sources"]] == ["gen_0052_deepvalue"]
    assert all(s["path"].endswith("_ramped") and (ROOT / s["path"] / "derivation.json").exists()
               for s in d["prior_extra_sources"])
    for name in ("gen53.json", "gen53_rehearsal.json", "gen53_pool.json", "gen53_pool_rehearsal.json"):
        assert recipe(name)["model"] == d["teacher"]
    assert g.ramped_path(g.paths_for(False)[1]["deep"]).name.endswith("_disagreement_ramped")


def test_the_pool_arm_is_conditional_on_the_decision():
    source = (ROOT / "tools/gen53_campaign.py").read_text()
    work = source[source.index("def work("):source.index("def main(")]
    assert 'if d["pool"]:' in work and 'arms = ["a"]' in work
    assert 'd["pool"] and all(' in work
