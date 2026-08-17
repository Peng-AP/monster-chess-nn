"""Play self-play games and export N examples of each outcome, in one pass.

`export_selfplay_replays.py` takes a finished match artifact and replays every
game in it, because the artifact stores only aggregates and there is no other
way to find which seed produced which outcome. That is the right design for
auditing a match that already happened -- but when the goal is simply "show me
five of each", it plays every game twice and plays far more of them than
needed. At 6400 sims that is hours.

This plays and records in a single pass, keeps what it needs, and stops as soon
as every category is full. It also reports the running tally, so the same run
that produces the examples measures the colour split for free.

    py -3 tools/selfplay_examples.py --model models/fresh_start_v21/best_value_net.pt \
        --sims 6400 --per-category 5 --out benchmarks/v21_6400_examples.html
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from export_selfplay_replays import (  # noqa: E402
    _init_worker, _play_recorded, classify_black_result, render_html)

CATEGORIES = ("black_loss", "black_win", "draw")
LABEL = {"black_loss": "White win", "black_win": "Black win", "draw": "Draw"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--sims", type=int, default=6400)
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-games", type=int, default=120,
                    help="stop even if a category is still short")
    ap.add_argument("--opening-temp-plies", type=int, default=16,
                    help="two deterministic engines replay one identical game "
                         "without sampled openings (benchmark.py:117)")
    ap.add_argument("--seed", type=int, default=64006400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--fpu-reduction", type=float, default=0.3)
    ap.add_argument("--policy-temperature", type=float, default=1.0)
    args = ap.parse_args()

    model_path = str(ROOT / args.model) if not Path(args.model).is_absolute() \
        else args.model
    kept: dict[str, list] = {key: [] for key in CATEGORIES}
    tally = {key: 0 for key in CATEGORIES}
    played = 0
    started = time.time()

    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(model_path, args.sims, "native", args.c_puct,
                           args.fpu_reduction, args.policy_temperature)) as pool:
        while played < args.max_games:
            if all(len(kept[k]) >= args.per_category for k in CATEGORIES):
                break
            batch = [(args.seed + played + i, args.opening_temp_plies, 0.5)
                     for i in range(args.workers)]
            for game in pool.imap_unordered(_play_recorded, batch):
                played += 1
                tally[game["category"]] += 1
                if len(kept[game["category"]]) < args.per_category:
                    kept[game["category"]].append(game)
            done = sum(min(len(kept[k]), args.per_category) for k in CATEGORIES)
            need = args.per_category * len(CATEGORIES)
            print(f"[{played} games] have {done}/{need} examples | "
                  f"White wins {tally['black_loss']} "
                  f"Black wins {tally['black_win']} draws {tally['draw']} | "
                  f"{(time.time() - started) / 60:.1f}m", flush=True)

    selected = []
    for category in CATEGORIES:
        for ordinal, game in enumerate(kept[category][:args.per_category], 1):
            game = dict(game)
            game["category_label"] = LABEL[category]
            game["title"] = f"{LABEL[category]} {ordinal}"
            selected.append(game)

    # The colour split falls out of the same games that produced the examples.
    decisive = tally["black_loss"] + tally["black_win"]
    payload = {
        "title": f"v21 self-play at {args.sims} sims",
        "model": args.model,
        "sims": args.sims,
        "games_played": played,
        "tally": {LABEL[k]: tally[k] for k in CATEGORIES},
        "white_share_of_decisive": (round(tally["black_loss"] / decisive, 4)
                                    if decisive else None),
        "games": selected,
    }
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(
        {k: v for k, v in payload.items() if k != "games"}, indent=2),
        encoding="utf-8")

    print(f"\n{played} games at {args.sims} sims in "
          f"{(time.time() - started) / 60:.1f}m")
    for key in CATEGORIES:
        print(f"   {LABEL[key]:11} {tally[key]:3d}  "
              f"({tally[key] / played:.1%})   exported {len(kept[key][:args.per_category])}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
