"""Build the first opening book and re-anchor the bar under it.

Switching the harness from temperature-sampled openings to a paired book
changes the measurement, so every score on record was produced under a regime
this one does not share. Nothing is comparable across the switch until one
measurement is repeated on both sides of it.

The repeat chosen here is v21b vs v21 at 800 games / 400 sims -- the largest
head-to-head on record (benchmarks/match_scratch_v21corpus_vs_fresh_start_v21_
20260807_202724.json, a_score 0.5450, CI [0.5104, 0.5796], W 0.670 B 0.420).
Same models, same game count, same simulations; only the openings and their
pairing differ.

Two things come out of it:

  * a MAPPING -- if the book score lands near 0.5450 the two regimes agree on
    strength and old numbers can be read across with care; if it does not, the
    book has shifted the operating point and only book-measured scores count.
  * the PAYOFF -- the paired standard error against the historical run's
    0.0177 at the same 800 games. That ratio is what decides whether the screen
    can drop to a third of its games, and it is measured rather than assumed.

    py -3 tools/book_rescore.py
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

BOOK = "books/v21b_p16.json"
BAR = "models/fresh_start_v21b/best_value_net.pt"
NUMBERED = "models/fresh_start_v21/best_value_net.pt"
REPORT = "benchmarks/book_rescore.json"

# The run being repeated. Its CI half-width is 0.0346, so SE = 0.01765.
HISTORICAL = {
    "artifact": "benchmarks/match_scratch_v21corpus_vs_fresh_start_v21"
                "_20260807_202724.json",
    "games": 800, "sims": 400, "a_score": 0.5450,
    "white": 0.670, "black": 0.420, "se": 0.0177,
}

GAMES = 800
SIMS = 400
ENTRIES = GAMES // 2          # each entry is played twice, colours reversed


def stage(name, cmd, marker=None):
    if marker and os.path.exists(os.path.join(ROOT, marker)):
        print(f"\n=== [{name}] already complete ({marker}) -- skipping ===",
              flush=True)
        return
    print(f"\n=== [{name}] {time.strftime('%H:%M:%S')} ===\n$ "
          f"{' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([PY, "-u"] + [str(c) for c in cmd], cwd=ROOT).returncode
    print(f"=== [{name}] exit={rc} after {(time.time() - t0) / 60:.1f}m ===",
          flush=True)
    if rc != 0:
        raise SystemExit(f"[{name}] failed with exit {rc}; stopping the chain")


def main():
    started = time.time()
    print("book re-anchor: repeating the largest head-to-head on record "
          "under paired openings", flush=True)

    stage("make-book", [
        "tools/make_book.py", "--model", BAR, "--entries", ENTRIES,
        "--plies", "16", "--sims", "700", "--temperature", "0.5",
        "--seed", "90000000", "--workers", "8", "--engine", "native",
        "--out", BOOK,
    ], marker=BOOK)

    stage("re-anchor", [
        "tools/match.py", "--model-a", BAR, "--model-b", NUMBERED,
        "--games", GAMES, "--sims", SIMS, "--engine", "native",
        "--workers", "8", "--book", BOOK, "--seed", "31415000",
        "--report-path", REPORT,
    ], marker=REPORT)

    out = json.load(open(os.path.join(ROOT, REPORT), encoding="utf-8"))
    paired = out.get("paired") or {}
    se = paired.get("se_paired")

    print("\n" + "=" * 68)
    print("RE-ANCHOR: v21b vs v21, 800 games @ 400 sims")
    print("=" * 68)
    print(f"{'':<14}{'sampled (historical)':>22}{'book (paired)':>18}")
    print(f"{'overall':<14}{HISTORICAL['a_score']:>22.4f}"
          f"{out['a_score']:>18.4f}")
    print(f"{'white':<14}{HISTORICAL['white']:>22.4f}"
          f"{out['a_as_white']['score']:>18.4f}")
    print(f"{'black':<14}{HISTORICAL['black']:>22.4f}"
          f"{out['a_as_black']['score']:>18.4f}")
    print(f"{'SE of mean':<14}{HISTORICAL['se']:>22.4f}"
          f"{se if se is not None else float('nan'):>18.4f}")

    if se:
        factor = HISTORICAL["se"] / se
        print(f"\nvariance reduction: {factor:.2f}x tighter at equal games "
              f"({paired['pairs']} pairs)")
        print(f"equivalent games saved: a sampled match needs "
              f"{GAMES * factor ** 2:.0f} games to match this precision")
    shift = out["a_score"] - HISTORICAL["a_score"]
    print(f"\noperating-point shift: {shift:+.4f} overall. Book and sampled "
          f"scores remain SEPARATE REGIMES regardless -- this maps them, it "
          f"does not merge them.")
    print(f"\ncomplete in {(time.time() - started) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
