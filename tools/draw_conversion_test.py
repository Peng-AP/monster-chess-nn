"""Can the repetition-aware engine convert the strong-model draws? (owner, 2026-10-12: "draw aware > solver")

Strong-vs-strong games on the repetition-aware engine (gen55's run) still end
in repetition draws: 1,004 of 4,440 at 3,200 and 349 of 1,040 at 12,800, most
of them completed by White's move. A short forced-capture search is the wrong
judge; the engine playing it out is the right one. This takes 100 of those
draws, starts 20 half-moves before the draw (Black to move), and has gen55 play
both sides with repetition awareness at 3,200 and at 12,800 simulations
(2 sampled half-moves so the paired games differ). Converted = Black captures
the king.

Waits for the opening census to finish (one GPU job at a time). Evidence:
benchmarks/game_value_20261012/draw_conversion/. Produces no model.
"""
import glob
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/game_value_20261012/draw_conversion"
GEN55 = "models/candidates/bootstrap_main_gen_0055/arena_selected.pt"
STRONG = ("gen_0053", "gen_0054", "gen_0055", "bootstrap_v29")
BACK = 20
COUNT = 100
SEED = 3_690_000_000


def norm(p):
    return str(p).replace(os.sep, "/")


def sample_draws():
    found = []
    for f in sorted(glob.glob(str(ROOT / "benchmarks/gen55_program/gen55_20261007/production/**/*.jsonl"),
                              recursive=True)):
        if not os.path.exists(f + ".manifest.json") or "/continuations/" in norm(f):
            continue
        s = json.load(open(f + ".manifest.json"))["settings"]
        if not all(any(x in norm(s[k]["path"]) for x in STRONG) for k in ("model_a", "model_b")):
            continue
        for line in open(f):
            r = json.loads(line)
            if "game" in r and r["game"]["termination"] == "repetition":
                found.append((norm(os.path.relpath(f, ROOT)), s["sims"], r["game"]["trajectory"]))
    random.Random(20261012).shuffle(found)
    entries, provenance = [], []
    for path, sims, traj in found:
        # Black to move, settled, about BACK half-moves before the end.
        for state in reversed(traj[:max(1, len(traj) - BACK)]):
            if state["fen"].split()[1] == "b" and not state.get("half"):
                entries.append(dict(fen=state["fen"], half=False, turn_count=state.get("turn_count", 0)))
                provenance.append(dict(source=path, original_sims=sims, plies_before_end=len(traj) - traj.index(state)))
                break
        if len(entries) == COUNT:
            break
    return entries, provenance


def main():
    os.chdir(ROOT)
    while "CENSUS ALL COMPLETE" not in (ROOT / "logs/opening_census.log").read_text(errors="ignore"):
        time.sleep(300)
    OUT.mkdir(parents=True, exist_ok=True)
    entries, provenance = sample_draws()
    book = OUT / "book.json"
    book.write_text(json.dumps(dict(schema_version=1, model="strong-model repetition draws, 20 half-moves before",
                                    plies=0, entries=entries), indent=1))
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=1))
    summary = {}
    for i, sims in enumerate((3200, 12800)):
        report, log = OUT / f"gen55_self_{sims}.json", OUT / f"gen55_self_{sims}.jsonl"
        if not report.exists():
            subprocess.run([sys.executable, "-u", "tools/match.py", "--model-a", GEN55, "--model-b", GEN55,
                            "--games", str(2 * len(entries)), "--sims", str(sims), "--sims-b", str(sims),
                            "--engine", "native", "--workers", "8", "--seed", str(SEED + i * 1_000_000),
                            "--book", str(book), "--book-temp-plies", "2", "--stall-timeout", "1800",
                            "--game-log", str(log), "--report-path", str(report), "--resume"], check=True)
        rows = [json.loads(l) for l in open(log) if '"game"' in l]
        ends = {}
        for r in rows:
            key = "Black wins" if r["white_score"] == 0 else "White wins" if r["white_score"] == 1 else \
                f"draw ({r['game']['termination']})"
            ends[key] = ends.get(key, 0) + 1
        summary[sims] = dict(games=len(rows), outcomes=ends)
        print(f"DRAW CONVERSION {sims}: {summary[sims]}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("DRAW CONVERSION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
