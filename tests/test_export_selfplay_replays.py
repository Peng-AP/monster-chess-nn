"""Small contracts for deterministic self-play replay selection."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from export_selfplay_replays import (  # noqa: E402
    classify_black_result, game_seeds, select_games,
)


def test_match_scoring_categories_only_count_captures_as_wins():
    assert classify_black_result(1.0) == "black_loss"
    assert classify_black_result(-1.0) == "black_win"
    assert classify_black_result(0.5) == "draw"
    assert classify_black_result(-0.5) == "draw"


def test_match_seed_layout_matches_match_tool():
    assert game_seeds({"games": 6, "seed": 100}) == [100, 101, 102, 1100, 1101, 1102]


def test_selection_is_three_games_per_category_in_stable_order():
    games = []
    for category in ("black_loss", "black_win", "draw"):
        for seed in range(4):
            games.append({"category": category, "category_label": category,
                          "seed": seed})
    selected = select_games(games, 3)
    assert [game["category"] for game in selected] == (
        ["black_loss"] * 3 + ["black_win"] * 3 + ["draw"] * 3)
    assert [game["seed"] for game in selected] == [0, 1, 2] * 3
