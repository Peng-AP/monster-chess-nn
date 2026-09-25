"""Stage 1 campaign contracts: arm design, seeds and the fixed nomination rule."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import search_constants_campaign as c
from config import C_PUCT, FPU_REDUCTION


def test_arms_are_one_factor_changes_from_the_engine_defaults():
    assert c.DEFAULT == {"c_puct": C_PUCT, "fpu_reduction": FPU_REDUCTION}
    for name, search in c.ARMS.items():
        changed = [k for k in search if search[k] != c.DEFAULT[k]]
        assert len(changed) == 1, name
    assert len(c.ARMS) == 4


def test_seed_blocks_do_not_overlap_between_stages_or_with_rehearsal():
    for smoke in (False, True):
        spec = c.layout(smoke)
        ranges = [(spec["par_seed"], spec["par_seed"] + 500_000)]
        ranges += [(c.screen_leg(i, spec)["seed"], c.screen_leg(i, spec)["seed"] + 1000 + spec["screen_games"])
                   for i in range(len(c.ARMS))]
        ranges.append((spec["confirm_seed"], spec["confirm_seed"] + 500_000))
        ranges.sort()
        assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), ranges
    production, rehearsal = c.layout(False), c.layout(True)
    assert rehearsal["par_seed"] >= production["confirm_seed"] + 500_000


def result(score, white, black):
    return {"score": score, "par_comparisons": {"white": {"delta_from_par": white},
                                                "black": {"delta_from_par": black}}}


def test_nomination_requires_the_score_floor_and_both_colours():
    assert c.nominate({"A": result(0.524, 0, 0)}) == (None, [])
    assert c.nominate({"A": result(0.60, -0.051, 0.2)}) == (None, [])
    assert c.nominate({"A": result(0.525, -0.05, -0.05)}) == ("A", ["A"])


def test_nomination_picks_the_best_score_then_the_better_worst_colour():
    arms = list(c.ARMS)
    results = {arms[0]: result(0.55, 0.00, 0.02), arms[1]: result(0.56, -0.04, 0.10),
               arms[2]: result(0.56, 0.01, 0.00), arms[3]: result(0.50, 0.2, 0.2)}
    chosen, eligible = c.nominate(results)
    assert chosen == arms[2] and set(eligible) == set(arms[:3])


def test_identity_pins_specific_files_not_whole_directories():
    assert all(not p.endswith("/") and "*" not in p for p in c.PINNED)
    assert "docs/plans/GEN51_STRENGTH_PLAN.md" in c.PINNED
