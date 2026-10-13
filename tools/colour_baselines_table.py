"""Per-colour baselines table (docs/experiments/game_value/COLOUR_BASELINES.md).

gen55 and v29 come from the census phase 1b cells. gen54, gen53, v28 and gen49
come from tools/colour_instrument.py, run on the same book with the same sims
and sampling. Paired differences are position-weighted over the 14 key
positions. Their SE is within-position binomial, because the book is fixed.
Half-results (the +-0.5 move-limit relabel) score as draws, as in match.py.

    py -3 tools/colour_baselines_table.py
"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "benchmarks/opening_census_20261012/phase1b/report.json"
INSTRUMENT = ROOT / "benchmarks/colour_instrument"
MEASURED = ("gen54", "gen53", "v28", "gen49")
MODELS = ("gen55", "v29", *MEASURED)
CELLS = (("white", "gen55"), ("white", "v29"), ("black", "gen55"), ("black", "v29"))


def load():
    rep = json.loads(CENSUS.read_text())
    labels, copies = rep["labels"], rep["copies"]
    cells = defaultdict(dict)                       # label -> (model, colour, ref) -> [points, n]
    for label in labels:
        c = rep["positions"][label]
        for ref in ("gen55", "v29"):
            for m in ("gen55", "v29"):
                x, y = c[f"{m}>{ref}"], c[f"{ref}>{m}"]
                cells[label][(m, "white", ref)] = [x["w"] + 0.5 * x["d"], x["n"]]
                cells[label][(m, "black", ref)] = [y["l"] + 0.5 * y["d"], y["n"]]
    for m in MEASURED:
        for ref in ("gen55", "v29"):
            for line in open(INSTRUMENT / m / f"vs_{ref}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                if "game" not in r:
                    continue
                key = (m, "white" if r["a_is_white"] else "black", ref)
                cell = cells[labels[r["entry"] // copies]].setdefault(key, [0.0, 0])
                cell[0] += 1 if r["result_for_a"] == 1 else 0 if r["result_for_a"] == -1 else 0.5
                cell[1] += 1
    return labels, cells


def main():
    labels, cells = load()

    def total(m, colour, ref):
        p = sum(cells[l][(m, colour, ref)][0] for l in labels)
        n = sum(cells[l][(m, colour, ref)][1] for l in labels)
        return p / n

    def paired(m, colour, ref, base):
        d = v = 0.0
        for l in labels:
            (pa, na), (pb, nb) = cells[l][(m, colour, ref)], cells[l][(base, colour, ref)]
            sa, sb = pa / na, pb / nb
            d += sa - sb
            v += sa * (1 - sa) / na + sb * (1 - sb) / nb
        return d / len(labels), math.sqrt(v) / len(labels)

    print("model  | W vs gen55 | W vs v29 | B vs gen55 | B vs v29 | W mean | B mean")
    for m in MODELS:
        s = [total(m, c, r) for c, r in CELLS]
        print(f"{m:6s} | " + " | ".join(f"{x * 100:5.1f}" for x in s) +
              f" | {(s[0] + s[1]) * 50:5.1f} | {(s[2] + s[3]) * 50:5.1f}")
    for base in ("v29", "gen55"):
        print(f"\npaired difference vs {base}, pp (SE)")
        for m in MODELS:
            if m != base:
                print(f"{m:6s} | " + " | ".join(f"{d * 100:+5.1f} ({s * 100:.1f})"
                                                 for d, s in (paired(m, c, r, base) for c, r in CELLS)))
    for colour, ref in (("white", "v29"), ("black", "gen55")):
        print(f"\nper position: {colour} vs {ref}'s {'Black' if colour == 'white' else 'White'} (%)")
        for l in labels:
            print(f"{l:12s} " + " ".join(f"{m}={cells[l][(m, colour, ref)][0] / cells[l][(m, colour, ref)][1] * 100:5.1f}"
                                         for m in MODELS))


if __name__ == "__main__":
    main()
