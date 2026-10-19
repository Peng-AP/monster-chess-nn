"""Opening atlas contracts (CPU only): the tree, transpositions, and the pre-declared selection rules."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import opening_atlas as atlas


def test_tree_is_58_first_turns_and_1156_replies():
    w1, b1 = atlas.tree()
    assert len(w1) == 58 and len(b1) == 1156
    assert all(label.split(" / ")[0] in w1 for label in b1)
    assert all(not v["entry"]["half"] for v in b1.values())          # White to move, a fresh turn


def test_white_turns_merge_transpositions_and_flag_terminals():
    from repetition import position_key
    _, b1 = atlas.tree()
    label = "d2d4 e2e4 / d7d5"
    kids = atlas.white_turns(label, b1[label])
    keys = [position_key(atlas.replay(v["moves"])) for v in kids.values()]
    assert len(keys) == len(set(keys)) and 50 <= len(kids) <= 120
    assert all(k.startswith(label + " / ") for k in kids)
    assert any(v["terminal"] is False for v in kids.values())
    assert all(v["result"] in (None, 1, -1, 0) for v in kids.values())


def test_g2_takes_three_lowest_plus_near_minimum_capped_at_six():
    by_w1 = {"A": [f"A / {i}" for i in range(10)]}
    values = {f"A / {i}": v for i, v in enumerate([0.0, 0.05, 0.08, 0.09, 0.10, 0.11, 0.5, 0.6, 0.02, 0.03])}
    chosen = atlas.g2_selection(by_w1, values)
    assert chosen == ["A / 0", "A / 8", "A / 9", "A / 1", "A / 2", "A / 3"]


def test_minimax_is_black_choosing_the_lowest_refined_value():
    by_w1 = {"A": ["A / x", "A / y"], "B": ["B / z"]}
    g1 = {"A / x": 0.10, "A / y": 0.30, "B / z": 0.25}
    pooled = {"A / x": dict(value=0.40), "A / y": dict(value=0.30), "B / z": dict(value=0.0)}
    m = atlas.refined_minimax(by_w1, g1, pooled, {"A / x"})
    assert m["A"] == (0.30, "A / y") and m["B"] == (0.25, "B / z")


def test_w2_candidates_are_top_twelve_by_search_plus_four_seeded_controls():
    kids = {f"n / {i}": dict(terminal=False) for i in range(30)}
    kids["n / T"] = dict(terminal=True)
    s2 = {f"n / {i}": {m: dict(value_white=i / 100) for m in atlas.S2_MODELS} for i in range(30)}
    top, controls = atlas.w2_candidates(3, kids, s2)
    assert top == [f"n / {i}" for i in range(29, 17, -1)]
    assert len(controls) == 4 and not set(controls) & set(top) and "n / T" not in controls
    assert (top, controls) == atlas.w2_candidates(3, kids, s2)


def test_seed_block_is_clear_of_the_held_out_remeasure():
    assert atlas.SEED >= 3_055_000_000 and atlas.SEED + 20_000_000 + 2_000_000 < 3_100_000_000
