"""Does fixing the opening change WHO WINS, or only how precisely we see it?

A book buys precision by deleting a skill from the measurement: with the first
N plies pre-played, no model gets credit for steering into positions it
understands. In this game that matters more than the ply count suggests --
16 plies is only turn_count 10 of 150, but it is about eleven White half-moves,
and White has just four pawns and a king to commit.

Whether that costs anything is an empirical question, not an argument. Hold
provenance and every other setting fixed, vary ONLY the book depth, and replay
the same head-to-head:

  * if the score is flat across depth, steering is not moving the measurement
    and the concern is theoretical -- take the precision.
  * if it drifts with depth, the book is measuring steering out, and the
    shallowest depth that still reduces variance is the right setting.

The variance reduction is read off the same runs, so this also prices the
trade rather than assuming a direction for it: deeper books remove more
opening-draw noise while deleting more steering, and the knee of those two
curves is the setting to keep.

The pair replayed is v21b vs v21 -- the largest head-to-head on record
(0.5450 over 800 games, SE 0.0177), so there is a known answer to drift
against.

    py -3 tools/book_depth_sweep.py
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

BAR = "models/fresh_start_v21b/best_value_net.pt"
NUMBERED = "models/fresh_start_v21/best_value_net.pt"
GEN7 = "models/candidates/gen7_scratch/screen_nominee.pt"

DEPTHS = [8, 16, 24]
GAMES = 200
SIMS = 400
ENTRIES = GAMES // 2

# Mixed provenance, held CONSTANT across depths so the only moving part is the
# depth. A single-lineage book would confound "deeper" with "more of v21b's
# taste", which is the other objection to books entirely.
PROVENANCE = [BAR, NUMBERED, GEN7]

HISTORICAL = {"a_score": 0.5450, "se": 0.0177, "games": 800,
              "white": 0.670, "black": 0.420}


def run(name, cmd, marker=None):
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
    for path in PROVENANCE:
        if not os.path.exists(os.path.join(ROOT, path)):
            raise SystemExit(f"missing provenance model: {path}")

    rows = []
    for depth in DEPTHS:
        book = f"books/mixed_p{depth}.json"
        report = f"benchmarks/book_depth_p{depth}.json"

        cmd = ["tools/make_book.py"]
        for model in PROVENANCE:
            cmd += ["--model", model]
        cmd += ["--entries", ENTRIES, "--plies", depth, "--sims", "700",
                "--temperature", "0.5", "--seed", 90100000 + depth,
                "--workers", "8", "--engine", "native", "--out", book]
        run(f"book-p{depth}", cmd, marker=book)

        run(f"match-p{depth}", [
            "tools/match.py", "--model-a", BAR, "--model-b", NUMBERED,
            "--games", GAMES, "--sims", SIMS, "--engine", "native",
            "--workers", "8", "--book", book, "--seed", 27180000 + depth,
            "--report-path", report,
        ], marker=report)

        out = json.load(open(os.path.join(ROOT, report), encoding="utf-8"))
        paired = out.get("paired") or {}
        rows.append({
            "depth": depth, "score": out["a_score"],
            "white": out["a_as_white"]["score"],
            "black": out["a_as_black"]["score"],
            "se_paired": paired.get("se_paired"),
            "pairs": paired.get("pairs"),
        })

    # Sampled openings at this game count, for the precision comparison.
    naive_se = (0.25 / GAMES) ** 0.5

    print("\n" + "=" * 74)
    print(f"BOOK DEPTH SWEEP -- v21b vs v21, {GAMES} games @ {SIMS} sims each")
    print("=" * 74)
    print(f"{'depth':>7}{'turns':>7}{'score':>9}{'white':>9}{'black':>9}"
          f"{'se_paired':>12}{'vs naive':>11}")
    for row in rows:
        factor = (naive_se / row["se_paired"]) if row["se_paired"] else 0.0
        print(f"{row['depth']:>7}{row['depth'] * 10 // 16:>7}"
              f"{row['score']:>9.4f}{row['white']:>9.4f}{row['black']:>9.4f}"
              f"{row['se_paired']:>12.4f}{factor:>10.2f}x")
    print(f"{'sampled':>7}{'-':>7}{HISTORICAL['a_score']:>9.4f}"
          f"{HISTORICAL['white']:>9.4f}{HISTORICAL['black']:>9.4f}"
          f"{HISTORICAL['se']:>12.4f}{'-':>11}"
          f"   ({HISTORICAL['games']} games)")

    scores = [r["score"] for r in rows]
    spread = max(scores) - min(scores)
    # Two independent 200-game paired samples differ by chance; the spread only
    # means something if it clears the noise of the runs being compared.
    worst_se = max((r["se_paired"] or 0.0) for r in rows)
    print(f"\nspread across depths: {spread:.4f} "
          f"(largest single-run SE {worst_se:.4f})")
    if spread <= 2 * worst_se:
        print("VERDICT: flat within noise -- depth is not moving the "
              "measurement, so deleting steering costs nothing measurable "
              "here. Choose depth on variance reduction alone.")
    else:
        drift = "monotonic" if scores == sorted(scores) or \
            scores == sorted(scores, reverse=True) else "non-monotonic"
        print(f"VERDICT: spread exceeds noise and is {drift} in depth -- the "
              f"book IS measuring steering out. Prefer the shallowest depth "
              f"whose variance reduction is still worth having.")

    path = os.path.join(ROOT, "benchmarks/book_depth_sweep.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "historical": HISTORICAL,
                   "naive_se_at_games": round(naive_se, 4),
                   "games": GAMES, "sims": SIMS,
                   "provenance": PROVENANCE,
                   "spread": round(spread, 4)}, fh, indent=2)
    print(f"\nwrote {path}")
    print(f"complete in {(time.time() - started) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
