import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from analyze_free_campaign import conversion_audit


def row(white, result, placement):
    return {"a_is_white": white, "result_for_a": result, "plies": 20,
            "game": {"trajectory": [{"fen": placement + " w - - 0 1", "plies_reached": 7}]}}


def test_bare_king_is_actual_material_not_an_assumed_win():
    rows = [row(True, -1, "8/8/8/8/8/8/8/K6k"),
            row(False, -1, "8/8/8/8/8/8/8/K6k"),
            row(False, .5, "8/8/8/8/8/8/8/K6k"),
            row(True, 1, "8/8/8/8/8/8/8/KQ5k"),  # promoted material is not bare
            row(True, -1, "8/8/8/8/8/8/8/7k")]  # captured White king is not bare
    out = conversion_audit(rows)
    assert out["games_reaching_bare_white_king"] == 3
    assert out["subsequent_outcomes"] == {"black_king_capture": 1, "white_king_capture": 1, "draw": 1}
    assert out["mean_remaining_search_plies"] == 13
