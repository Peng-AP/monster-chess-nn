"""A/B a candidate heuristic against the frozen baseline, same-side paired.

Naive A/B (candidate plays one color, baseline the other) is confounded: the
game's side imbalance is huge at reachable sim counts — baseline-vs-itself
scores ~0.9 as White and ~0.0 as Black — so it swamps any eval-quality
difference. Instead we test ONE side at a time:

  --test-side white:  candidate-White vs ref-Black, and baseline-White vs
                      ref-Black. Both play White against the SAME reference
                      Black, so the side imbalance cancels; the score gap is
                      pure White-eval quality.

To keep scores off the 0/1 rails (where nothing is distinguishable), handicap
sims: give the reference opponent more sims than the tested side so the tested
side scores in a discriminating ~0.3-0.7 band. Run with candidate == baseline
(default scales) first to read that band (the "baseline" block) before trusting
a delta.

    py -3 tools/heuristic_ab.py --test-side white --geom-scale 0.4 --attack-scale 1.8 \
        --cand-sims 150 --ref-sims 600 --games 16
"""
import argparse
import functools
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mcts import MCTS
from evaluation import evaluate
from benchmark import play_one, summarize_side, _build_engine
from config import CURRICULUM_FENS, CURRICULUM_TIER_BOUNDARIES


def fens_for_tiers(tiers):
    """Curriculum start FENs for the given 1-indexed tier numbers."""
    starts = [0] + list(CURRICULUM_TIER_BOUNDARIES)
    ends = list(CURRICULUM_TIER_BOUNDARIES) + [len(CURRICULUM_FENS)]
    out = []
    for t in tiers:
        out.extend(CURRICULUM_FENS[starts[t - 1]:ends[t - 1]])
    return out


def _engine(eval_fn, sims):
    return MCTS(num_simulations=sims, eval_fn=eval_fn, root_noise=False,
                allow_early_stop=True)


def _ref_engine(ref_model, sims):
    """Reference opponent: an NN model if given, else the frozen heuristic.

    A strong NN reference (e.g. v8) is what de-saturates White-eval tests —
    heuristic-Black is too weak to stress White, so White pins at ~1.0 and no
    White change is visible.
    """
    if ref_model:
        engine, _ = _build_engine(ref_model, sims)
        return engine
    return _engine(evaluate, sims)


def _score_one_eval(tested_eval, test_side, cand_sims, ref_sims, games, seed,
                    ref_model=None):
    """Score one eval on a fixed side against the reference opponent."""
    tested = _engine(tested_eval, cand_sims)
    ref = _ref_engine(ref_model, ref_sims)
    rows = []
    for i in range(games):
        random.seed(seed + i)
        if test_side == "white":
            result, plies, _ = play_one(tested, ref)        # tested is White
            rows.append((result, plies))
        else:
            result, plies, _ = play_one(ref, tested)        # tested is Black
            rows.append((-result, plies))
    return summarize_side(rows)


def _score_eval_on_fens(tested_eval, test_side, fens, sims, seed):
    """Score one eval on a fixed side vs the baseline opponent, one game per FEN.

    Contested mid-game start positions (not the White-decided opening), so the
    tested side's play quality actually swings the result. temp 0 -> one
    deterministic game per FEN; the score is the fraction won across the suite.
    """
    ref_eval = evaluate
    tested = _engine(tested_eval, sims)
    ref = _engine(ref_eval, sims)
    rows = []
    for i, fen in enumerate(fens):
        random.seed(seed + i)
        board_side = fen.split()[1]  # 'w' or 'b' to move in this FEN
        if test_side == "white":
            result, plies, _ = play_one(tested, ref, start_fen=fen)
            rows.append((result, plies))
        else:
            result, plies, _ = play_one(ref, tested, start_fen=fen)
            rows.append((-result, plies))
    return summarize_side(rows)


def run_fen_suite(test_side, geom, attack, tiers, sims, seed):
    """Same-side paired A/B over a curriculum-tier FEN suite."""
    all_fens = fens_for_tiers(tiers)
    want = "w" if test_side == "white" else "b"
    fens = [f for f in all_fens if f.split()[1] == want]  # tested side to move
    candidate_eval = functools.partial(
        evaluate, king_geom_scale=geom, king_attack_scale=attack)
    baseline = _score_eval_on_fens(evaluate, test_side, fens, sims, seed)
    candidate = _score_eval_on_fens(candidate_eval, test_side, fens, sims, seed)
    return {
        "mode": "fen_suite",
        "test_side": test_side,
        "tiers": tiers,
        "n_fens": len(fens),
        "king_geom_scale": geom,
        "king_attack_scale": attack,
        "sims": sims,
        "seed": seed,
        "baseline": baseline,
        "candidate": candidate,
        "delta_score": round((candidate["score"] or 0) - (baseline["score"] or 0), 4),
    }


def run_side_test(test_side, geom, attack, cand_sims, ref_sims, games, seed,
                  ref_model=None):
    candidate_eval = functools.partial(
        evaluate, king_geom_scale=geom, king_attack_scale=attack)
    baseline = _score_one_eval(evaluate, test_side, cand_sims, ref_sims, games,
                               seed, ref_model=ref_model)
    candidate = _score_one_eval(candidate_eval, test_side, cand_sims, ref_sims,
                                games, seed, ref_model=ref_model)
    return {
        "test_side": test_side,
        "ref_model": ref_model or "heuristic",
        "king_geom_scale": geom,
        "king_attack_scale": attack,
        "cand_sims": cand_sims,
        "ref_sims": ref_sims,
        "games": games,
        "seed": seed,
        "baseline": baseline,     # baseline eval on the tested side vs ref
        "candidate": candidate,   # candidate eval on the tested side vs ref
        "delta_score": round((candidate["score"] or 0) - (baseline["score"] or 0), 4),
    }


def main():
    p = argparse.ArgumentParser(description="Same-side A/B of a heuristic change vs baseline")
    p.add_argument("--test-side", choices=["white", "black"], required=True)
    p.add_argument("--geom-scale", type=float, default=1.0)
    p.add_argument("--attack-scale", type=float, default=1.0)
    p.add_argument("--mode", choices=["match", "fens"], default="fens",
                   help="fens: contested curriculum positions (default); "
                        "match: full games from the opening (saturated, diagnostic only)")
    p.add_argument("--tiers", type=str, default="6,7,9,10",
                   help="curriculum tiers for --mode fens (contested mid-games)")
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--cand-sims", type=int, default=150, help="[match mode] tested side sims")
    p.add_argument("--ref-sims", type=int, default=600, help="[match mode] reference sims")
    p.add_argument("--games", type=int, default=16, help="[match mode] games")
    p.add_argument("--ref-model", type=str, default=None,
                   help="[match mode] NN reference opponent path (default: heuristic). "
                        "Use a strong model (e.g. v8) to de-saturate White tests.")
    p.add_argument("--seed", type=int, default=20260707)
    args = p.parse_args()

    if args.mode == "fens":
        tiers = [int(t) for t in args.tiers.split(",") if t.strip()]
        print(f"A/B fens test-side={args.test_side}: candidate(geom={args.geom_scale}, "
              f"attack={args.attack_scale}) vs baseline, tiers {tiers} @ {args.sims} sims")
        r = run_fen_suite(args.test_side, args.geom_scale, args.attack_scale,
                          tiers, args.sims, args.seed)
    else:
        print(f"A/B match test-side={args.test_side}: candidate(geom={args.geom_scale}, "
              f"attack={args.attack_scale}) @ {args.cand_sims} vs "
              f"{args.ref_model or 'heuristic'} @ {args.ref_sims} sims, {args.games} games")
        r = run_side_test(args.test_side, args.geom_scale, args.attack_scale,
                          args.cand_sims, args.ref_sims, args.games, args.seed,
                          ref_model=args.ref_model)
    print(json.dumps(r, indent=2))
    base, cand = r["baseline"], r["candidate"]
    saturated = base["score"] in (0.0, 1.0)
    n = r.get("n_fens", r.get("games"))
    print(f"\n  baseline {args.test_side}: {base['score']} "
          f"({base['wins']}-{base['losses']}-{base['draws']} over {n})")
    print(f"  candidate {args.test_side}: {cand['score']} "
          f"({cand['wins']}-{cand['losses']}-{cand['draws']} over {n})")
    print(f"  delta: {r['delta_score']:+}"
          + ("   [WARNING: baseline saturated at rail — not discriminating]" if saturated else ""))


if __name__ == "__main__":
    main()
