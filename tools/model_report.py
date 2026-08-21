"""The full measuring standard for a model that claims a version number.

WHY THIS EXISTS. The gate is a cheap shortlist and should stay one: 400 sims,
one book, one opponent. It cannot carry a promotion. Measured across
2026-08-18/20:

  * A gate pass on the gate's own book overstates by 10-20 Elo every time.
    gen25 read +32/+16 there and -3.5 on a neutral book; gen26 +40/+36 and
    +25.5; gen27 +20/+15 and +9.2; gen30 +34/+32 and +20.9.
  * Book and free play disagree about COLOUR by more than any two models
    differ from each other. gen30 self-match White is 0.2800 under a book and
    0.8500 free, both at 3200 sims.
  * Per-colour scores move ~0.07 between 400 and 1600 sims, in 16 of 16 models,
    so a per-colour number without its sim count and its bar's own par is
    uninterpretable.
  * Free play cannot judge a candidate against the bar it TRAINED against:
    five for five, always large, always the same direction.

So one number cannot describe a model. This runs the matrix the owner
specified (2026-08-20): self and anchor, book and free, plus the same four
formats against each named predecessor, at 1600 and/or 3200 sims.

THE ANCHOR IS A FIXED MODEL, not the moving bar and not the heuristic. v22 is
the default because it is the only colour-BALANCED model on record (self-match
White 0.5300 at 1600 and 0.5350 at 3200, stable across depth where every chain
model drifts toward Black) and because it already anchors the 16-model ladder,
so new numbers stay comparable to the historical ones. An unbalanced anchor
would contaminate exactly the colour readings this exists to capture.

READING THE OUTPUT. A book self-match is paired -- identical engines replay the
same game, so White + Black = 1.0 exactly and the White score IS the colour
split with no pairing error. A FREE self-match is not paired: the two colours
come from different game sets, need not sum to 1.0, and its aggregate carries
no information at all. Only the two colour scores mean anything there.
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

DEFAULT_ANCHOR = "models/fresh_start_v22/best_value_net.pt"
DEFAULT_BOOK = "books/gate_v27_mixed_p8_20260817.json"
DEFAULT_OFFSET = 1620


def elo(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def main():
    ap = argparse.ArgumentParser(
        description="Full promotion-standard match matrix for one model.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True,
                    help="short label used in the report and filenames")
    ap.add_argument("--anchor", default=DEFAULT_ANCHOR,
                    help="FIXED reference model (default v22)")
    ap.add_argument("--anchor-name", default="v22")
    ap.add_argument("--opponent", action="append", default=[],
                    metavar="NAME=PATH",
                    help="important predecessor, repeatable")
    ap.add_argument("--sims", action="append", type=int, default=[],
                    help="repeatable; default 1600")
    ap.add_argument("--book", default=DEFAULT_BOOK)
    ap.add_argument("--book-offset", type=int, default=DEFAULT_OFFSET)
    ap.add_argument("--book-games", type=int, default=600)
    ap.add_argument("--free-games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--out-dir", default="benchmarks/model_report")
    ap.add_argument("--report-path", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sims_list = args.sims or [1600]
    opponents = [("self", None), (args.anchor_name, args.anchor)]
    for spec in args.opponent:
        if "=" not in spec:
            ap.error(f"--opponent must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        opponents.append((name, path))

    os.chdir(ROOT)
    out_dir = args.out_dir
    os.makedirs(os.path.join(ROOT, out_dir), exist_ok=True)

    from match import run_match

    cells, seed = [], 5_000_000
    plan = [(opp_name, opp_path, instrument, sims)
            for opp_name, opp_path in opponents
            for instrument in ("book", "free")
            for sims in sims_list]

    print(f"model_report: {args.name} -- {len(plan)} matches "
          f"({len(opponents)} opponents x 2 instruments x {len(sims_list)} "
          f"sim levels)", flush=True)
    for opp_name, opp_path, instrument, sims in plan:
        games = args.book_games if instrument == "book" else args.free_games
        print(f"  {args.name} vs {opp_name:8s} {instrument:4s} @{sims:5d} "
              f"{games} games", flush=True)
    if args.dry_run:
        return

    t_all = time.time()
    for opp_name, opp_path, instrument, sims in plan:
        seed += 1000
        games = args.book_games if instrument == "book" else args.free_games
        tag = f"{args.name}_vs_{opp_name}_{instrument}_{sims}"
        report_path = f"{out_dir}/{tag}.json"
        abs_report = os.path.join(ROOT, report_path)
        if os.path.exists(abs_report):
            print(f"SKIP {tag}", flush=True)
            d = json.load(open(abs_report))
        else:
            print(f"START {tag}", flush=True)
            t0 = time.time()
            d = run_match(
                args.model, opp_path or args.model, games, sims, seed,
                workers=args.workers, engine=args.engine,
                book=args.book if instrument == "book" else None,
                book_offset=args.book_offset if instrument == "book" else 0,
                game_log=(f"{out_dir}/{tag}.lines.jsonl"
                          if instrument == "book" else None))
            with open(abs_report, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=2)
            print(f"DONE  {tag} in {(time.time()-t0)/60:.1f}m", flush=True)

        white = d["a_as_white"]["score"]
        black = d["a_as_black"]["score"]
        cell = {
            "opponent": opp_name, "instrument": instrument, "sims": sims,
            "games": games, "white": white, "black": black,
            "report": report_path,
        }
        if opp_name == "self":
            # Aggregate is 0.5 by construction under a book and meaningless
            # free; the colour split is the whole content.
            cell["colour_split"] = round(white - black, 4)
        else:
            cell["aggregate"] = d["a_score"]
            cell["elo"] = round(elo(d["a_score"]), 1)
        od = d.get("opening_diversity") or {}
        cell["effective_unique"] = od.get("effective_unique_fraction")
        cells.append(cell)
        print(f"      W={white:.4f} B={black:.4f}"
              + (f" agg={d['a_score']:.4f} Elo={cell['elo']:+.1f}"
                 if "elo" in cell else
                 f" split={cell['colour_split']:+.4f}"), flush=True)

    out = {
        "model": args.model, "name": args.name,
        "anchor": {"name": args.anchor_name, "path": args.anchor},
        "book": args.book, "book_offset": args.book_offset,
        "sims": sims_list, "cells": cells,
        "elapsed_hours": round((time.time() - t_all) / 3600, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = args.report_path or f"benchmarks/model_report_{args.name}.json"
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path} ({out['elapsed_hours']}h)", flush=True)


if __name__ == "__main__":
    main()
