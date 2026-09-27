"""Gen52 campaign contracts (CPU only): pool design, held-out opponents, seeds, recipes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gen52_campaign as g
import gen51_campaign as g51
import search_constants_campaign as stage1
import stateful_generation as sg
from data_processor import policy_weight_for_record


def recipe(name):
    return json.loads((ROOT / "tools/recipes" / name).read_text())


def test_pool_is_the_owner_approved_four_and_excludes_held_out_models():
    assert g.POOL == [g.V28, g.GEN49, g.GEN48, g.V26]
    assert set(g.HELD_OUT) == {"models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt",
                               "models/bootstrap_v27/best_value_net.pt"}
    for name in ("gen52_pool.json", "gen52_pool_rehearsal.json"):
        r = recipe(name)
        assert r["opponents"] == g.POOL and not set(r["opponents"]) & set(g.HELD_OUT)
        assert r["free_games"] == r["fresh_games"] == r["fork_games"] == 0


def test_1200_pool_games_split_evenly_across_models_and_colours():
    r = dict(recipe("gen52_pool.json"), model="teacher.pt")
    batches = sg.ordinary_batches(r)
    tasks = [t for batch in batches for t in batch]
    assert len(tasks) == 1200
    counts = {}
    for t in tasks:
        counts[(t["other"], t["train_side"])] = counts.get((t["other"], t["train_side"]), 0) + 1
        assert t["kind"] == "league" and t["temperature_plies"] == 30
    assert set(counts.values()) == {150} and len(counts) == 8
    # One model pair per batch: the worker pool loads only the first task's models.
    assert all(len({t["other"] for t in batch}) == 1 for batch in batches)


def test_only_the_teachers_moves_carry_policy_weight():
    assert policy_weight_for_record({"policy_weight": 0.0, "source": "stateful_league"}) == 0.0
    assert policy_weight_for_record({"policy_weight": 1.0, "source": "stateful_league"}) == 1.0


def test_generation_recipe_is_gen51_with_only_teacher_and_seed_changed():
    gen51, gen52 = recipe("gen51.json"), recipe("gen52.json")
    assert {k for k in set(gen51) | set(gen52) if gen51.get(k) != gen52.get(k)} == {"model", "seed"}


def test_seed_blocks_are_disjoint_from_gen51_and_stage1():
    blocks = []
    for smoke in (False, True):
        s1, s51, s52 = stage1.layout(smoke), g51.layout(smoke), g.layout(smoke)
        blocks += [(s1["par_seed"], s1["confirm_seed"] + 500_000),
                   (s51["continuation_seed"], s51["arms_seed"] + 2_000_000),
                   (s52["par_seed"], s52["arms_seed"] + 2_000_000)]
    blocks.sort()
    assert all(a[1] <= b[0] for a, b in zip(blocks, blocks[1:])), blocks
    assert max(b for _, b in blocks) < 2 ** 32
    seeds = sorted(recipe(n)["seed"] for n in ("gen51.json", "gen52.json", "gen52_pool.json"))
    assert seeds[1] - seeds[0] >= 1_000_000 and seeds[2] - seeds[1] >= 1_000_000


def test_arm_directories_are_distinct_and_new():
    _, extra = g.paths_for(False)
    names = {extra["a"]["model_dir"].name, extra["b"]["model_dir"].name, extra["a"]["replay"].name,
             extra["b"]["replay"].name, extra["pool"].name, extra["deep"].name}
    assert len(names) == 6 and all("0052" in n for n in names)
