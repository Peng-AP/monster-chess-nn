"""Gen56 contracts (CPU only): census book starts, teachers, pool, seeds, the colour selection rule."""
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
import gen55_campaign as g55
import gen56_campaign as g

DECISION = json.loads((ROOT / "docs/plans/gen56_teacher_decision.json").read_text())


def recipe(kind, smoke=False):
    return json.loads(g.recipe_path(kind, smoke).read_text())


def book_lines():
    return [dict(moves=b["moves"], weight=b["weight"]) for b in DECISION["book_starts"]]


def test_decision_shape_and_held_out_isolation():
    d = DECISION
    assert d["teacher"].endswith("bootstrap_main_gen_0055/arena_selected.pt")
    assert "extra_teachers" not in d and d["paired_book_games"] == 1400
    assert [(p["white"], p["black"]) for p in d["book_pairings"]] == [
        ("gen55", "gen53"), ("gen55", "v29"), ("gen54", "gen53"), ("gen54", "v29")]
    assert d["pairing_models"]["gen55"]["model"] == d["teacher"] and d["pairing_models"]["v29"]["model"] == d["release"]
    assert len(d["pool_opponents"]) == 8 and d["pool_opponents"][0] == d["release"]
    held = {str(Path(p)) for p in g.HELD_OUT}
    everyone = [d["teacher"], *(m["model"] for m in d["pairing_models"].values()), *d["pool_opponents"]]
    assert not {str(Path(p)) for p in everyone} & held
    assert d["colour_check"]["black_tolerance"] == 0.02 and d["proposed_promotion"]["per_colour_tolerance"] == 0.03


def test_book_starts_are_the_census_lines():
    lines = DECISION["book_starts"]
    assert len(lines) == 70
    assert sum(1 for b in lines if len(b["moves"]) == 2) == 58          # every White first turn
    assert sum(1 for b in lines if len(b["moves"]) == 3) == 12          # two best Black replies to six turns
    key = json.loads((ROOT / "benchmarks/opening_census_20261012/phase2/report.json").read_text())["labels"] \
        if (ROOT / "benchmarks/opening_census_20261012/phase2/report.json").exists() else None
    heavy = {" ".join(sorted(b["moves"])) for b in lines if len(b["moves"]) == 2 and b["weight"] == 2.0}
    assert len(heavy) == 14 and (key is None or heavy == set(key))
    import stateful_generation as sg
    tasks = sg.book_tasks(dict(model="m.pt", sims=8, seed=5, book_games=140, book_starts=book_lines()))
    assert len(tasks) == 140 and len({t["state"]["fen"] for t in tasks}) <= 70


def test_recipes_match_the_decision():
    for smoke in (False, True):
        canonical = recipe("", smoke)
        assert canonical["model"] == DECISION["teacher"] and canonical["book_starts"] == book_lines()
        assert recipe("_pool", smoke)["opponents"] == DECISION["pool_opponents"]
        assert recipe("_pool", smoke)["league_games"] % (2 * len(DECISION["pool_opponents"])) == 0
        r = recipe("_paired", smoke)
        assert r["model"] == DECISION["teacher"] and r["free_games"] == 0 and r["book_starts"] == book_lines()
        assert r["book_pairings"] == g.pairings(DECISION) and r["temperature_plies"] == 30
        assert r["book_games"] % len(r["book_pairings"]) == 0
    assert recipe("")["free_games"] == 1400 and recipe("")["book_games"] == 1400
    assert recipe("_paired")["book_games"] == DECISION["paired_book_games"] == 1400
    assert recipe("_paired")["sims"] == 1600 and recipe("_pool")["league_games"] == 86 * 2 * 8


def test_paired_book_tasks_give_every_pairing_an_equal_share():
    import stateful_generation as sg
    r = recipe("_paired")
    tasks = sg.book_tasks(r)
    assert len(tasks) == 1400
    shares = {}
    for t in tasks:
        key = (t["white_model"], t["black_model"])
        shares[key] = shares.get(key, 0) + 1
    assert sorted(shares) == sorted((p["white"], p["black"]) for p in r["book_pairings"])
    assert max(shares.values()) - min(shares.values()) <= 1
    for line in {tuple(t["state"]["moves"]) for t in tasks}:
        mine = [(t["white_model"], t["black_model"]) for t in tasks if tuple(t["state"]["moves"]) == line]
        counts = [mine.count(p) for p in set(shares)]
        assert max(counts) - min(counts) <= 1


@pytest.mark.local_artifacts
def test_identity_validates():
    i = g.identity()
    assert [e["generation"] for e in i["replay"]][-1] == 55


def test_seed_blocks_are_disjoint_and_below_2_32():
    blocks = []
    for smoke in (False, True):
        s51 = g51.layout(smoke)
        blocks.append((s51["continuation_seed"], s51["arms_seed"] + 2_000_000))
        for spec in (g52.layout(smoke), g53.layout(smoke), g54.layout(smoke), g55.layout(smoke), g.layout(smoke)):
            blocks.append((spec["par_seed"], spec["arms_seed"] + 2_000_000))
    blocks += [(3_400_000_000, 3_520_000_000), (3_600_000_000, 3_610_000_000),   # Elo round robin, ladder
               (3_640_000_000, 3_696_000_000),                                     # opening census
               (4_100_000_000, 4_101_000_000), (4_110_000_000, 4_111_000_000),     # release-gate chains
               (4_200_000_000, 4_201_000_000)]                                     # position probe
    blocks.sort()
    assert all(a[1] <= b[0] for a, b in zip(blocks, blocks[1:])), blocks
    assert max(b for _, b in blocks) < 2 ** 32
    spec = g.layout(False)
    used = sorted([spec["par_seed"], spec["continuation_seed"], spec["screen_seed"],
                   *(spec["deep_probe_seed"] + i * 1_000_000 for i in range(3)), spec["release_par_seed"],
                   spec["colour_seed"], spec["colour_seed"] + 1_000_000,
                   *(spec["candidate_colour_seed"] + i * 1_000_000 for i in range(6)),
                   spec["gate_seed"], spec["release_gate_seed"], spec["instrument_seed"], spec["instrument_seed"] + 1_000_000,
                   *(spec["diag_seed"] + i * 1_000_000 for i in range(6))])
    assert all(b - a >= 1_000_000 for a, b in zip(used, used[1:])), used
    seeds = sorted(recipe(k, s)["seed"] for k in g.RECIPE_KINDS for s in (False, True))
    assert all(b - a >= 1_000_000 for a, b in zip(seeds, seeds[1:])), seeds


def candidate(name, deep_score=0.7, white=0.2, black=0.8):
    return dict(name=name, deep=dict(score=deep_score, par_comparisons=dict(
        white=dict(delta_from_par=0.0), black=dict(delta_from_par=0.0))), colour=dict(white=white, black=black))


def test_colour_rule_takes_best_screen_rank_passing_both():
    teacher = dict(white=0.18, black=0.85)
    cands = [candidate("e15", white=0.17), candidate("e9", white=0.19, black=0.84), candidate("e4", white=0.25, black=0.80)]
    name, eligible, rule = g.colour_nominee(cands, teacher, 0.02)
    assert (name, eligible) == ("e9", ["e9"]) and "colour" in rule


def test_colour_rule_falls_back_to_best_white():
    teacher = dict(white=0.30, black=0.85)
    cands = [candidate("e15", white=0.20), candidate("e9", white=0.25, deep_score=0.2), candidate("e4", white=0.10)]
    assert g.colour_nominee(cands, teacher, 0.02) == ("e9", [], "none_passed_both; best_colour_white")


def test_release_baseline_reads_the_census():
    if not g.CENSUS_REPORT.exists():
        pytest.skip("census report not present")
    b = g.release_colour_baseline()["vs"]
    assert b["v29"]["white"] == pytest.approx(0.0973, abs=1e-4) and b["gen55"]["white"] == pytest.approx(0.1143, abs=1e-3)
    assert 0 < b["gen55"]["black"] < 1 and b["v29"]["black"] == pytest.approx(0.9027, abs=1e-4)
