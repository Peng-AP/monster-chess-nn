"""Pool gate legs across replicate seeds and say whether an effect is real.

Four arms scored against a 0.50 line will produce a pass or two by chance even
when the true effect is zero. Eyeballing the best number is how a noise result
gets promoted, so this pools them and reports the pooled score with its
standard error.

**The bar leg is the honest statistic.** Every arm plays it regardless of
outcome, so pooling bar legs across seeds is unbiased. The confirmation leg is
only played when the first legs pass, so pooling confirmations conditions on
success and is guaranteed to look better than the truth -- it is reported
separately and labelled, never mixed into the headline.

    py -3 tools/replication_summary.py --prefix capture_wdl_w003_seed
    py -3 tools/replication_summary.py --prefix capture_wdl_w0 --leg vs_v21
"""
import argparse
import glob
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR = "vs_v21"
CONFIRM = "vs_v21_confirm"


def points(block):
    """Match points from a summarize_side block: a draw is half a point."""
    if not block:
        return 0.0, 0
    return block["wins"] + 0.5 * block["draws"], block["games"]


def leg_points(leg):
    if not leg:
        return 0.0, 0
    pw, gw = points(leg.get("a_as_white"))
    pb, gb = points(leg.get("a_as_black"))
    return pw + pb, gw + gb


def collect(prefix):
    """Newest gate artifact per arm whose name starts with prefix."""
    by_arm = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "benchmarks", "gate_*.json"))):
        name = re.sub(r"^gate_(.+)_\d{8}_\d{6}\.json$", r"\1",
                      os.path.basename(path))
        if not name.startswith(prefix):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                by_arm[name] = json.load(fh)      # sorted, so newest wins
        except Exception:
            pass
    return by_arm


def band(pooled, se):
    if se == 0:
        return "no games"
    z = (pooled - 0.50) / se
    if abs(z) < 1.0:
        return f"z={z:+.2f}  indistinguishable from parity"
    if abs(z) < 2.0:
        return f"z={z:+.2f}  suggestive, not established"
    return f"z={z:+.2f}  distinguishable from parity"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--leg", default=BAR)
    args = ap.parse_args()

    arms = collect(args.prefix)
    if not arms:
        raise SystemExit(f"no gate artifacts matching {args.prefix!r}")

    print(f"{'arm':38} {'verdict':8} {'bar':>8} {'confirm':>9} {'conf B':>8}")
    print("-" * 76)
    bar_pts, bar_games, con_pts, con_games = 0.0, 0, 0.0, 0
    n_confirm = 0
    for name in sorted(arms):
        d = arms[name]
        legs = d.get("legs") or {}
        b, c = legs.get(args.leg), legs.get(CONFIRM)
        p, g = leg_points(b)
        bar_pts += p
        bar_games += g
        if c:
            pc, gc = leg_points(c)
            con_pts += pc
            con_games += gc
            n_confirm += 1
        print(f"{name:38} {str(d.get('verdict')):8} "
              f"{(b or {}).get('a_score', float('nan')):8.4f} "
              f"{(c or {}).get('a_score', float('nan')):9.4f} "
              f"{(c or {}).get('a_as_black', {}).get('score', float('nan')):8.3f}")

    print("-" * 76)
    if bar_games:
        pooled = bar_pts / bar_games
        se = math.sqrt(0.25 / bar_games)
        print(f"POOLED {args.leg} over {len(arms)} arms, {bar_games} games "
              f"(unbiased -- every arm plays this leg)")
        print(f"   score {pooled:.4f}  SE {se:.4f}   {band(pooled, se)}")
    if con_games:
        pooled_c = con_pts / con_games
        se_c = math.sqrt(0.25 / con_games)
        print(f"\nPOOLED {CONFIRM} over {n_confirm}/{len(arms)} arms, "
              f"{con_games} games")
        print(f"   score {pooled_c:.4f}  SE {se_c:.4f}   {band(pooled_c, se_c)}")
        if n_confirm < len(arms):
            print("   NOTE: conditioned on passing the first legs -- this is "
                  "biased upward and is not the headline number.")
        else:
            # Every arm played both legs, so combining them is unbiased and is
            # the best point estimate available. Worth printing because the
            # gate is a DECISION procedure (two reads above the line), not a
            # measurement: an arm can pass it while the pooled effect remains
            # indistinguishable from parity at these sample sizes.
            allp, allg = bar_pts + con_pts, bar_games + con_games
            pooled_all = allp / allg
            se_all = math.sqrt(0.25 / allg)
            print(f"\nPOOLED BOTH LEGS, all {len(arms)} arms, {allg} games "
                  f"(unbiased: every arm played both)")
            print(f"   score {pooled_all:.4f}  SE {se_all:.4f}   "
                  f"{band(pooled_all, se_all)}")


if __name__ == "__main__":
    main()
