"""Colour skew as a function of search depth, from self-match artifacts.

Monster Chess is asymmetric by construction -- White moves twice per turn,
Black has a full army -- so a model playing itself does not score 0.500 per
colour. The size of that skew, and whether it shrinks as search deepens, is the
question that decides whether "close the colour gap" is a reachable target or a
property of the game.

Reads every self-match in benchmarks/ (model_a == model_b), groups by
simulation count, and compares the gaps with **both** errors propagated. Doing
this by hand is how a noisy baseline turns into a three-sigma result that
isn't there.

    py -3 tools/skew_by_depth.py
    py -3 tools/skew_by_depth.py --model fresh_start_v21
"""
import argparse
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_self_matches(model_filter):
    """Self-matches only: a candidate-vs-other match measures strength, not skew."""
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "benchmarks", "match_*.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if d.get("name_a") != d.get("name_b"):
            continue
        if d.get("sims") != d.get("sims_b"):
            continue          # asymmetric search is a different experiment
        if model_filter and model_filter not in str(d.get("name_a")):
            continue
        w, b = d.get("a_as_white"), d.get("a_as_black")
        if not w or not b:
            continue
        out.append({
            "sims": int(d["sims"]), "games": int(d["games"]),
            "model": d["name_a"],
            "white": w["score"], "white_games": w["games"],
            "black": b["score"], "black_games": b["games"],
            "draws": w["draws"] + b["draws"],
            "path": os.path.basename(path),
        })
    return out


def gap_se(white_games, black_games):
    """SE of (White - Black); the two are measured on disjoint game sets."""
    return math.sqrt(0.25 / white_games + 0.25 / black_games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fresh_start_v21")
    args = ap.parse_args()

    rows = load_self_matches(args.model)
    if not rows:
        raise SystemExit(f"no self-match artifacts found for {args.model!r}")
    rows.sort(key=lambda r: (r["sims"], -r["games"]))

    print(f"{'sims':>6} {'games':>6} {'White':>8} {'Black':>8} {'gap':>8} "
          f"{'gap SE':>8} {'draws':>7}  artifact")
    print("-" * 92)
    for r in rows:
        gap = r["white"] - r["black"]
        print(f"{r['sims']:6d} {r['games']:6d} {r['white']:8.4f} {r['black']:8.4f} "
              f"{gap:8.4f} {gap_se(r['white_games'], r['black_games']):8.4f} "
              f"{r['draws'] / r['games']:6.1%}  {r['path'][:38]}")

    if len(rows) < 2:
        print("\nonly one depth measured -- nothing to compare yet")
        return

    print("\npairwise gap comparisons (both errors propagated):")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a["sims"] == b["sims"]:
                continue
            ga = a["white"] - a["black"]
            gb = b["white"] - b["black"]
            se = math.sqrt(gap_se(a["white_games"], a["black_games"]) ** 2
                           + gap_se(b["white_games"], b["black_games"]) ** 2)
            diff = gb - ga
            z = diff / se if se else 0.0
            verdict = ("no detectable change" if abs(z) < 1.0
                       else "suggestive" if abs(z) < 2.0
                       else "detectable")
            direction = "narrows" if diff < 0 else "widens"
            print(f"  {a['sims']:>5} -> {b['sims']:<5}  gap {ga:.3f} -> {gb:.3f}  "
                  f"{direction} by {abs(diff):.3f} +- {se:.3f}  z={z:+.2f}  {verdict}")

    print("\nNote: gap SE is 1/sqrt(games) for an even colour split, so resolving")
    print("a 0.05 shift needs ~1600 games per depth. These sample sizes detect")
    print("large changes only -- read 'no detectable change' as 'not large'.")


if __name__ == "__main__":
    main()
