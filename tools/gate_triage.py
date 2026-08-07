"""Which rejected arms were rejected by noise rather than by weakness?

The per-side floor is checked on half a leg -- 20 games, SE 0.112 -- so a
candidate genuinely better than the incumbent on Black clears both bar legs
only about half the time (REPORT.md 21.1). Every FAIL in benchmarks/ was
produced by that procedure, so the archive is not a clean record of what
works: some fraction of it is candidates that lost a coin toss.

This ranks past failures by how *narrowly* they failed, so a re-test at proper
sample size can be spent on the ones most likely to have been misjudged. It
reads artifacts only -- no games are played.

Ranking rule: a candidate is a re-test priority when it failed on a per-side
floor by less than one standard error (0.112), or failed an aggregate by less
than one standard error of a 40-game leg (0.079), while showing no sign of
genuine weakness elsewhere.

    py -3 tools/gate_triage.py
    py -3 tools/gate_triage.py --max-shortfall 0.05
"""
import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDE_SE = 0.112        # 20 games per colour
LEG_SE = 0.079         # 40-game leg

FLOOR_RE = re.compile(r"(\S+) (white|black) leg ([\d.]+) < ([\d.]+)")
AGG_RE = re.compile(r"(\S+) aggregate ([\d.]+) <= ([\d.]+)")


def shortfalls(failures):
    """How far under the line each failure was, with its noise scale."""
    out = []
    for text in failures or []:
        m = FLOOR_RE.search(text)
        if m:
            leg, side, got, need = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
            out.append({"kind": "floor", "leg": leg, "side": side,
                        "short": need - got, "se": SIDE_SE, "text": text})
            continue
        m = AGG_RE.search(text)
        if m:
            leg, got, need = m.group(1), float(m.group(2)), float(m.group(3))
            out.append({"kind": "aggregate", "leg": leg, "side": None,
                        "short": need - got, "se": LEG_SE, "text": text})
            continue
        out.append({"kind": "other", "leg": None, "side": None,
                    "short": None, "se": None, "text": text})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shortfall-se", type=float, default=1.0,
                    help="report failures within this many SE of the line")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "benchmarks", "gate_*.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if d.get("verdict") != "FAIL" or not d.get("binding", True):
            continue
        name = re.sub(r"^gate_(.+)_\d{8}_\d{6}\.json$", r"\1", os.path.basename(path))
        items = shortfalls(d.get("failures"))
        if not items or any(i["short"] is None for i in items):
            continue
        worst = max(i["short"] / i["se"] for i in items)
        rows.append({"name": name, "worst_se": worst, "items": items,
                     "n_failures": len(items),
                     "legs": d.get("legs") or {}, "path": os.path.basename(path)})

    rows.sort(key=lambda r: r["worst_se"])
    near = [r for r in rows if r["worst_se"] <= args.max_shortfall_se]

    print(f"{len(rows)} binding FAILs on record; {len(near)} failed by "
          f"<= {args.max_shortfall_se:.1f} SE\n")
    print(f"{'candidate':40} {'worst miss':>11} {'fails':>6}  reason")
    print("-" * 100)
    for r in near:
        first = min(r["items"], key=lambda i: -i["short"] / i["se"])
        print(f"{r['name'][:40]:40} {r['worst_se']:8.2f} SE {r['n_failures']:6d}  "
              f"{first['text'][:44]}")

    if near:
        print("\nRe-test priority: these lost by less than the measurement's own noise.")
        print("A single 40-game leg cannot separate them from the incumbent; an")
        print("800-game direct match can, at roughly one hour each.")
    print(f"\n{len(rows) - len(near)} failed by more than "
          f"{args.max_shortfall_se:.1f} SE and are better explained by weakness.")


if __name__ == "__main__":
    main()
