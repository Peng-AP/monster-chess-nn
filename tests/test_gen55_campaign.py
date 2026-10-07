"""Gen55 contracts (CPU only): teacher mix, pool, held-out isolation, seeds, the White-check selection rule."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gen51_campaign as g51
import gen52_campaign as g52
import gen53_campaign as g53
import gen54_campaign as g54
import gen55_campaign as g


def recipe(kind, smoke=False):
    return json.loads(g.recipe_path(kind, smoke).read_text())


@pytest.mark.local_artifacts
def test_decision_is_the_owner_mix_with_held_out_isolation():
    d = g.decision()
    assert d["teacher"].endswith("bootstrap_main_gen_0054/arena_selected.pt")
    assert [t["name"] for t in d["extra_teachers"]] == ["gen53", "r", "lr"]
    assert all(t["games"] == 700 for t in d["extra_teachers"])
    assert d["release"] == "models/bootstrap_v29/best_value_net.pt"
    assert d["white_check"]["max_white_deficit_vs_release_par"] == 0.05
    gen54_pool = json.loads((ROOT / "docs/plans/gen54_teacher_decision.json").read_text())["pool_opponents"]
    assert d["pool_opponents"] == gen54_pool + ["models/candidates/bootstrap_main_gen_0053/arena_selected.pt"]
    held = {str(Path(p)) for p in g.HELD_OUT}
    everyone = [d["teacher"], *(t["model"] for t in d["extra_teachers"]), *d["pool_opponents"]]
    assert not {str(Path(p)) for p in everyone} & held
    assert [s["name"] for s in d["prior_extra_sources"]] == ["gen_0054_deepvalue"]
    assert all((ROOT / s["path"] / "derivation.json").exists() for s in d["prior_extra_sources"])


def test_recipes_match_the_decision_and_split_evenly():
    d = json.loads((ROOT / "docs/plans/gen55_teacher_decision.json").read_text())
    for smoke in (False, True):
        assert recipe("", smoke)["model"] == d["teacher"] and recipe("_pool", smoke)["model"] == d["teacher"]
        assert recipe("_pool", smoke)["opponents"] == d["pool_opponents"]
        assert recipe("_pool", smoke)["league_games"] % (2 * len(d["pool_opponents"])) == 0
        for t in d["extra_teachers"]:
            r = recipe(f"_teacher_{t['name']}", smoke)
            assert r["model"] == t["model"] and r["opponents"] == [] and r["league_games"] == 0
            assert r["temperature_plies"] == 30
    assert recipe("_pool")["league_games"] == 86 * 2 * 8
    assert all(recipe(f"_teacher_{n}")["free_games"] == 700 for n in ("gen53", "r", "lr"))
    canonical, old = recipe(""), json.loads((ROOT / "tools/recipes/gen54.json").read_text())
    assert {k for k in set(old) | set(canonical) if old.get(k) != canonical.get(k)} == {"model", "seed"}


@pytest.mark.local_artifacts
def test_identity_validates():
    i = g.identity()
    assert [e["generation"] for e in i["replay"]][-1] == 54


def test_seed_blocks_are_disjoint_and_below_2_32():
    blocks = []
    for smoke in (False, True):
        s51 = g51.layout(smoke)
        blocks.append((s51["continuation_seed"], s51["arms_seed"] + 2_000_000))
        for spec in (g52.layout(smoke), g53.layout(smoke), g54.layout(smoke), g.layout(smoke)):
            blocks.append((spec["par_seed"], spec["arms_seed"] + 2_000_000))
    blocks.append((3_400_000_000, 3_520_000_000))  # tools/elo_tournament.py
    blocks.append((3_600_000_000, 3_610_000_000))  # tools/elo_ladder.py
    blocks.sort()
    assert all(a[1] <= b[0] for a, b in zip(blocks, blocks[1:])), blocks
    assert max(b for _, b in blocks) < 2 ** 32
    spec = g.layout(False)
    offsets = sorted(v for k, v in spec.items() if k.endswith("_seed"))
    assert all(b - a >= 1_000_000 for a, b in zip(offsets, offsets[1:]))
    seeds = [recipe(k, s)["seed"] for k in g.RECIPE_KINDS for s in (False, True)]
    seeds += [json.loads((ROOT / "tools/recipes" / f"gen54{k}{s}.json").read_text())["seed"]
              for k in ("", "_pool", "_teacher_r", "_teacher_lr") for s in ("", "_rehearsal")]
    seeds.sort()
    assert all(b - a >= 1_000_000 for a, b in zip(seeds, seeds[1:])), seeds


def candidate(name, deep_score=0.7, deep_white=0.0, deep_black=0.0, white_delta=0.0):
    return dict(name=name, deep=dict(score=deep_score, par_comparisons=dict(
        white=dict(delta_from_par=deep_white), black=dict(delta_from_par=deep_black))),
        white_check=dict(white_delta_from_release_par=white_delta))


def test_white_check_rule_takes_best_screen_rank_passing_both():
    cands = [candidate("e13", white_delta=-0.20), candidate("e22", white_delta=-0.03), candidate("e4", white_delta=0.0)]
    name, eligible, rule = g.white_check_nominee(cands, 0.05)
    assert (name, eligible) == ("e22", ["e22", "e4"]) and "white_check" in rule


def test_white_check_rule_falls_back_to_best_white_when_none_pass():
    cands = [candidate("e13", white_delta=-0.20), candidate("e22", deep_score=0.3, white_delta=-0.01),
             candidate("e4", white_delta=-0.09)]
    name, eligible, rule = g.white_check_nominee(cands, 0.05)
    assert (name, eligible, rule) == ("e22", [], "none_passed_both; best_white_check")


def test_single_arm_composes_all_sources_and_gates_against_teacher_and_release():
    source = (ROOT / "tools/gen55_campaign.py").read_text()
    work = source[source.index("def work("):source.index("def main(")]
    assert "extra_sources(campaign, smoke)" in work and "deep_source(campaign, smoke)" in work
    assert '"release_par"' in work and '"b"' not in work
    evaluate = source[source.index("def evaluate("):source.index("def work(")]
    assert 'gate(campaign, smoke, "a"' in evaluate and 'gate(campaign, smoke, "release"' in evaluate
    _, extra = g.paths_for(False)
    assert "0055" in extra["a"]["replay"].name and extra["teacher"]("gen53")[1].name.endswith("_teacher_gen53")
