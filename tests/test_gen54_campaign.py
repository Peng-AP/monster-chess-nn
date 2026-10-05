"""Gen54 contracts (CPU only): teacher mix, hole-scan pool, held-out isolation, seeds, one arm."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gen51_campaign as g51
import gen52_campaign as g52
import gen53_campaign as g53
import gen54_campaign as g
import search_constants_campaign as stage1


def recipe(kind, smoke=False):
    return json.loads(g.recipe_path(kind, smoke).read_text())


def test_decision_is_the_owner_mix_with_held_out_isolation():
    d = g.decision()
    assert d["teacher"].endswith("bootstrap_main_gen_0053/arena_selected.pt")
    assert [t["name"] for t in d["extra_teachers"]] == ["r", "lr"] and all(t["games"] == 700 for t in d["extra_teachers"])
    assert d["pool"] and d["single_arm"] and d["deep_value_labels"] == "ramped"
    assert len(d["pool_opponents"]) == 7
    held = {str(Path(p)) for p in g.HELD_OUT}
    everyone = [d["teacher"], *(t["model"] for t in d["extra_teachers"]), *d["pool_opponents"]]
    assert not {str(Path(p)) for p in everyone} & held
    assert [s["name"] for s in d["prior_extra_sources"]] == ["gen_0053_deepvalue"]
    assert all(s["path"].endswith("_ramped") and (ROOT / s["path"] / "derivation.json").exists()
               for s in d["prior_extra_sources"])


def test_recipes_match_the_decision_and_split_evenly():
    d = g.decision()
    for smoke in (False, True):
        assert recipe("", smoke)["model"] == d["teacher"] and recipe("_pool", smoke)["model"] == d["teacher"]
        assert recipe("_pool", smoke)["opponents"] == d["pool_opponents"]
        assert recipe("_pool", smoke)["league_games"] % (2 * len(d["pool_opponents"])) == 0
        for t in d["extra_teachers"]:
            r = recipe(f"_teacher_{t['name']}", smoke)
            assert r["model"] == t["model"] and r["opponents"] == [] and r["league_games"] == 0
            assert r["temperature_plies"] == 30
    assert recipe("_teacher_r")["free_games"] == 700 and recipe("_teacher_r")["sims"] == 1600
    canonical, old = recipe(""), json.loads((ROOT / "tools/recipes/gen53.json").read_text())
    assert {k for k in set(old) | set(canonical) if old.get(k) != canonical.get(k)} == {"model", "seed"}


def test_identity_validates():
    i = g.identity()
    assert [e["generation"] for e in i["replay"]][-1] == 53


def test_seed_blocks_are_disjoint_and_below_2_32():
    blocks = []
    for smoke in (False, True):
        s51 = g51.layout(smoke)
        blocks.append((s51["continuation_seed"], s51["arms_seed"] + 2_000_000))
        for spec in (g52.layout(smoke), g53.layout(smoke), g.layout(smoke)):
            blocks.append((spec["par_seed"], spec["arms_seed"] + 2_000_000))
    blocks.sort()
    assert all(a[1] <= b[0] for a, b in zip(blocks, blocks[1:])), blocks
    assert max(b for _, b in blocks) < 2 ** 32
    seeds = [recipe(k, s)["seed"] for k in ("", "_pool", "_teacher_r", "_teacher_lr") for s in (False, True)]
    seeds += [json.loads((ROOT / "tools/recipes" / n).read_text())["seed"]
              for n in ("gen53.json", "gen53_pool.json", "gen53_rehearsal.json", "gen53_pool_rehearsal.json")]
    seeds.sort()
    assert all(b - a >= 1_000_000 for a, b in zip(seeds, seeds[1:])), seeds


def test_each_generated_source_has_its_own_directory():
    source = (ROOT / "tools/gen54_campaign.py").read_text()
    body = source[source.index("def generated_source("):source.index("def extra_sources(")]
    assert '"generation" / label / "summary.json"' in body


def test_single_arm_composes_all_sources():
    source = (ROOT / "tools/gen54_campaign.py").read_text()
    work = source[source.index("def work("):source.index("def main(")]
    assert "extra_sources(campaign, smoke)" in work and "deep_source(campaign, smoke)" in work
    assert '"b"' not in work
    _, extra = g.paths_for(False)
    assert "0054" in extra["a"]["replay"].name and extra["teacher"]("r")[1].name.endswith("_teacher_r")
