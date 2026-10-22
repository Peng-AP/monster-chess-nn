"""Deep-dive scoring rules (CPU only): strict minimax, near-best candidates only, never averages."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import opening_deepdive as dd


def test_black_candidates_drop_substantially_worse_replies():
    # Best reply holds White to 5%; 12% and 14% are within 0.10; 30% and 60% are substantially worse.
    values = {"r1": 0.05, "r2": 0.12, "r3": 0.30, "r4": 0.14, "r5": 0.60}
    assert dd.candidates(values, best_is_low=True) == ["r1", "r2", "r4"]


def test_one_clear_best_reply_is_judged_alone():
    values = {"best": 0.10, "worse": 0.45, "much_worse": 0.80}
    assert dd.candidates(values, best_is_low=True) == ["best"]


def test_white_candidates_are_the_highest():
    values = {"a": 0.50, "b": 0.42, "c": 0.20, "d": None}
    assert dd.candidates(values, best_is_low=False) == ["a", "b"]


def test_candidates_capped_and_deterministic():
    values = {f"m{i}": 0.10 for i in range(9)}
    assert dd.candidates(values, best_is_low=True) == ["m0", "m1", "m2", "m3"]


def test_minimax_takes_the_minimum_not_the_mean():
    # A line's value is Black's best answer: min over its candidates, whatever the others score.
    cr = {"x": 0.05, "y": 0.12}
    assert min(cr.values()) == 0.05 and min(cr.values()) != sum(cr.values()) / 2


def test_seed_blocks_clear_of_atlas_and_campaigns():
    assert dd.SEED >= 3_080_000_000 and dd.SEED + 16_000_000 <= 3_100_000_000
    assert dd.REHEARSAL_SEED + 16_000_000 <= 3_050_000_000


def test_dive_first_turns_are_atlas_positions():
    import opening_atlas as A
    w1, _ = A.tree()
    assert all(w in w1 for w in dd.DIVE_W1) and len(dd.DIVE_W1) == 6
