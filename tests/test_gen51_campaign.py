"""Gen51 campaign contracts (CPU only): commands, seeds, selection rules, linked splits."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gen51_campaign as g
import disagreement_continuations as dc
import process_linked_extra as ple
import search_constants_campaign as stage1


def test_control_iteration_is_the_gen50_recipe_with_only_declared_changes():
    command = g.iteration_command(False)
    value = lambda flag: command[command.index(flag) + 1]
    assert value("--recipe") == "tools/recipes/gen51.json"
    assert value("--incumbent") == g.V28
    assert value("--expected-generation") == "51"
    assert value("--through-phase") == "train"
    for flag, expected in [("--games", "3200"), ("--sims", "1600"), ("--reanalysis-sims", "12800"),
                           ("--reanalysis-sample", "24000"), ("--reanalysis-keep", "12000"),
                           ("--replay-generations", "8"), ("--seed", "3173"), ("--epochs", "30"),
                           ("--value-floor", ".5"), ("--teacher-policy-multiplier", "4")]:
        assert value(flag) == expected, flag


def test_recipes_differ_from_gen50_only_in_teacher_seed_and_exploration():
    import json
    gen50 = json.loads((ROOT / "campaigns/gen50/gen50_recipe.json").read_text())
    gen51 = json.loads((ROOT / "tools/recipes/gen51.json").read_text())
    changed = {k for k in set(gen50) | set(gen51) if gen50.get(k) != gen51.get(k)}
    assert changed == {"model", "seed", "temperature_plies"}
    assert gen51["temperature_plies"] == 30 and gen51["model"] == g.V28


def test_seed_blocks_are_disjoint_from_stage1_and_between_production_and_rehearsal():
    blocks = []
    for smoke in (False, True):
        s1 = stage1.layout(smoke)
        blocks.append((s1["par_seed"], s1["confirm_seed"] + 500_000))
        spec = g.layout(smoke)
        blocks.append((spec["continuation_seed"], spec["arms_seed"] + 2_000_000))
    blocks.sort()
    assert all(a[1] <= b[0] for a, b in zip(blocks, blocks[1:])), blocks
    assert max(b for _, b in blocks) < 2 ** 32


def result(name, aggregate, white, black, checkpoint=None):
    return {"name": name, "checkpoint": checkpoint or f"models/x/{name}.pt",
            "deltas": {"aggregate": aggregate, "white": white, "black": black},
            "minimum_color_delta": min(white, black)}


def test_screen_candidates_take_the_screen_ranking_then_fill_from_probes():
    report = {"results": [result("e14", .10, .05, .15), result("e15", .12, -.20, .40)],
              "probe": {"results": [result("e14", .2, .1, .1), result("e9", .05, .02, .08),
                                    result("e3", -.1, -.1, -.1)]}}
    names = [c["name"] for c in g.screen_candidates(report)]
    # e15 collapsed White (<= -0.10), so it ranks below e14 despite the higher aggregate.
    assert names == ["e14", "e15", "e9"]


def deep(name, score, white, black):
    return {"name": name, "deep": {"score": score, "par_comparisons": {
        "white": {"delta_from_par": white}, "black": {"delta_from_par": black}}}}


def test_deep_nominee_prefers_the_best_screen_rank_that_passes_the_guard():
    candidates = [deep("a", .60, -.15, .30), deep("b", .50, -.05, .02), deep("c", .55, .0, .0)]
    assert g.deep_nominee(candidates)[:2] == ("b", ["b", "c"])


def test_deep_nominee_falls_back_to_the_best_deep_score_when_none_pass():
    candidates = [deep("a", .40, -.2, .1), deep("b", .46, -.3, .2), deep("c", .46, -.4, .3)]
    name, eligible, rule = g.deep_nominee(candidates)
    assert (name, eligible) == ("b", []) and rule.startswith("none_passed")


def test_linked_splitter_follows_parents_and_rejects_orphans():
    split = ple.parent_splitter({"train": ["selfplay/game_00001.jsonl"], "val": ["selfplay/game_00002.jsonl"],
                                 "test": []})
    games = [{"game_id": "disagree/root_0000_0.jsonl", "split_parent": "selfplay/game_00002.jsonl"},
             {"game_id": "disagree/root_0000_1.jsonl", "split_parent": "selfplay/game_00002.jsonl"},
             {"game_id": "disagree/root_0001_0.jsonl", "split_parent": "selfplay/game_00001.jsonl"}]
    out = split(games, 0)
    assert [x["game_id"] for x in out["val"]] == ["disagree/root_0000_0.jsonl", "disagree/root_0000_1.jsonl"]
    assert len(out["train"]) == 1 and out["test"] == []
    with pytest.raises(ValueError):
        split([{"game_id": "x", "split_parent": "selfplay/game_09999.jsonl"}], 0)


def record(player, half, plies, policy_size=3):
    return {"current_player": player, "half": half, "state": {"moves": ["e2e4"] * plies},
            "policy": {f"m{i}": 1 / policy_size for i in range(policy_size)}}


def test_disagreement_root_phases_and_eligibility():
    assert dc.PHASE_CYCLE == ("black", "black", "white_first", "white_second")
    assert dc.phase(record("black", 0, 10)) == "black"
    assert dc.phase(record("white", 1, 10)) == "white_second"
    assert dc.eligible(record("white", 0, 4), 4, 120) and dc.eligible(record("white", 0, 120), 4, 120)
    assert not dc.eligible(record("white", 0, 3), 4, 120)
    assert not dc.eligible(record("white", 0, 121), 4, 120)
    assert not dc.eligible(record("black", 0, 30, policy_size=1), 4, 120)


def test_continuation_tasks_link_parents_and_keep_historical_exploration():
    config = dict(player=g.V28, sims=6400, seed=2_910_000_000, continuations=2)
    roots = [dict(index=0, parent="selfplay/game_00007.jsonl", state={"moves": []}, selection={"line": 12}),
             dict(index=1, parent="selfplay/game_00003.jsonl", state={"moves": []}, selection={"line": 5})]
    tasks = dc.tasks_for(config, roots)
    assert [t["id"] for t in tasks] == ["disagree/root_0000_0", "disagree/root_0000_1",
                                        "disagree/root_0001_0", "disagree/root_0001_1"]
    assert len({t["seed"] for t in tasks}) == 4
    assert all("temperature_plies" not in t for t in tasks)
    assert tasks[2]["source_record"] == {"path": "selfplay/game_00003.jsonl", "line": 5}


def test_campaign_pins_existing_files_and_the_approved_weight():
    assert g.DEEP_VALUE_WEIGHT == 4.0 and g.DEEP_PROBE_GAMES == 80 and g.TOP_EPOCHS == 3
    assert all((ROOT / p).is_file() for p in g.PINNED if p != "tests/test_gen51_campaign.py")
