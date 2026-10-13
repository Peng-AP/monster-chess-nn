"""Per-colour instrument: a model's White and Black strength, measured separately (owner, 2026-10-12).

docs/experiments/game_value/OPENING_CENSUS.md, recommendation 1. Strong models'
results are decided by a handful of opening lines and by one or two precise
Black replies, so one aggregate score from model-chosen openings mostly measures
repertoire luck. This plays a candidate against each reference over the census
key positions (the 14 first turns of phase 1b, Black to move), 20 copies with 8
sampled half-moves, every entry played twice with colours reversed:

    White score = candidate as White vs the reference's Black
    Black score = candidate as Black vs the reference's White

Baselines are the references against themselves on the same book and sampling
(phase 1b: gen55 White 21.7%; v29 White 9.7%). Report both colours and both
references; never one aggregate, and never against 50%.

    py -3 tools/colour_instrument.py --name gen53 --model models/.../arena_selected.pt
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "benchmarks/opening_census_20261012/phase1b/book.json"
REFERENCES = {"gen55": "models/candidates/bootstrap_main_gen_0055/arena_selected.pt",
              "v29": "models/bootstrap_v29/best_value_net.pt"}
BASELINE = ROOT / "benchmarks/opening_census_20261012/phase1b/report.json"
OUT = ROOT / "benchmarks/colour_instrument"
SEED = 3_696_000_000   # one 1M block per (candidate slot, reference); see seed_for
SLOTS = {"gen53": 0, "gen54": 1, "v28": 2, "gen49": 3}


def seed_for(name, ref_index):
    slot = SLOTS.get(name)
    if slot is None:
        raise SystemExit(f"add {name} to SLOTS so its seeds are fixed and disjoint")
    return SEED + (slot * 2 + ref_index) * 1_000_000


def score(rows, white):
    rs = [r for r in rows if r["a_is_white"] == white]
    w = sum(1 for r in rs if r["result_for_a"] == 1)
    l = sum(1 for r in rs if r["result_for_a"] == -1)
    d = len(rs) - w - l
    return dict(n=len(rs), w=w, d=d, l=l, score=round((w + 0.5 * d) / len(rs), 4) if rs else None,
                distinct=len({tuple(x for t in r["game"]["trajectory"] for x in t.get("action", [])) for r in rs}))


def baselines():
    """Each reference against itself on the same book: its White score and Black score."""
    pos = json.loads(BASELINE.read_text())["positions"]
    out = {}
    for ref in REFERENCES:
        w = d = l = 0
        for cells in pos.values():
            c = cells[f"{ref}>{ref}"]; w += c["w"]; d += c["d"]; l += c["l"]
        n = w + d + l
        out[ref] = dict(white=round((w + 0.5 * d) / n, 4), black=round((l + 0.5 * d) / n, 4), n=n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    os.chdir(ROOT)
    entries = len(json.loads(BOOK.read_text())["entries"])
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    result = dict(candidate=args.name, model=args.model, book=str(BOOK.relative_to(ROOT)), baselines=baselines(), vs={})
    for i, (ref, path) in enumerate(REFERENCES.items()):
        report, log = out / f"vs_{ref}.json", out / f"vs_{ref}.jsonl"
        if not report.exists():
            subprocess.run([sys.executable, "-u", "tools/match.py", "--model-a", args.model, "--model-b", path,
                            "--games", str(2 * entries), "--sims", "3200", "--sims-b", "3200", "--engine", "native",
                            "--workers", "8", "--seed", str(seed_for(args.name, i)), "--book", str(BOOK),
                            "--book-temp-plies", "8", "--stall-timeout", "1800", "--game-log", str(log),
                            "--report-path", str(report), "--resume"], check=True)
        rows = [json.loads(l) for l in open(log) if '"game"' in l]
        result["vs"][ref] = dict(white=score(rows, True), black=score(rows, False))
        b = result["baselines"][ref]
        print(f"COLOUR {args.name} vs {ref}: White {result['vs'][ref]['white']['score']*100:.1f}% "
              f"(baseline {b['white']*100:.1f}%), Black {result['vs'][ref]['black']['score']*100:.1f}% "
              f"(baseline {b['black']*100:.1f}%)", flush=True)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"COLOUR INSTRUMENT {args.name} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
