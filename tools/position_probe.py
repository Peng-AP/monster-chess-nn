"""Who can hold one position? Play it many times with several models against one opponent.

October 5, 2026: gen53 lost all 32 of its White games against v27 from one
position (reached after the 16 sampled opening plies), yet drew it 22/22 against
v28 and 7/7 against gen49. This plays that position with each candidate as
White against v27 as Black (and colours reversed), 10 times per side with 4
sampled plies after the position so the samples differ. If every candidate
loses as White, the position is lost and gen53's opening choice is the fault;
if some hold it, gen53's endgame play is.

Diagnostic only: it produces no training data, so v27 stays a held-out test.

    py -3 tools/position_probe.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

POSITION = {"fen": "rnbqkb1r/ppp2npp/4P3/8/8/4K3/2P5/8 w kq - 0 6", "half": True, "turn_count": 10}
OPPONENT = ("v27", "models/bootstrap_v27/best_value_net.pt")
CANDIDATES = [
    ("gen53", "models/candidates/bootstrap_main_gen_0053/arena_selected.pt"),
    ("gen52R", "models/candidates/bootstrap_main_gen_0052_ramp/arena_selected.pt"),
    ("v29", "models/bootstrap_v29/best_value_net.pt"),
    ("gen52LR", "models/candidates/bootstrap_main_gen_0052_large_ramp/arena_selected.pt"),
    ("gen52L", "models/candidates/bootstrap_main_gen_0052_large/arena_selected.pt"),
    ("v28", "models/bootstrap_v28/best_value_net.pt"),
]
PER_SIDE = 10
SEED = 4_200_000_000
OUT = os.path.join(ROOT, "benchmarks", "position_probe_20261005")


def main():
    global CANDIDATES, OUT
    import argparse
    from match import run_match
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", action="append", default=[], help="name=path (repeatable); replaces the default list")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.candidate:
        CANDIDATES = [tuple(c.split("=", 1)) for c in args.candidate]
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    book = os.path.join(OUT, "probe_book.json")
    with open(book, "w", encoding="utf-8") as fh:
        json.dump(dict(schema_version=1, model="hand-picked: gen53 vs v27 losing endpoint", plies=16,
                       entries=[dict(POSITION) for _ in range(PER_SIDE)]), fh, indent=2)
    summary = {}
    for i, (name, path) in enumerate(CANDIDATES):
        report = os.path.join(OUT, f"{name}_vs_{OPPONENT[0]}.json")
        if not os.path.exists(report):
            m = run_match(os.path.join(ROOT, path), os.path.join(ROOT, OPPONENT[1]), games=2 * PER_SIDE, sims=3200,
                          seed=SEED + i * 1_000_000, book=book, book_temp_plies=4, workers=8, engine="native",
                          stall_timeout=1800.0, game_log=report[:-5] + ".jsonl",
                          resume=os.path.exists(report[:-5] + ".jsonl"))
            with open(report, "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=2)
        m = json.load(open(report, encoding="utf-8"))
        w, b = m["a_as_white"], m["a_as_black"]
        summary[name] = dict(as_white=dict(wins=w["wins"], draws=w["draws"], losses=w["losses"], score=w["score"]),
                             as_black=dict(wins=b["wins"], draws=b["draws"], losses=b["losses"], score=b["score"]))
        print(f"PROBE {name} as White vs {OPPONENT[0]}: W/D/L {w['wins']}/{w['draws']}/{w['losses']} "
              f"({w['score']:.2f}); as Black {b['wins']}/{b['draws']}/{b['losses']}", flush=True)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(position=POSITION, opponent=OPPONENT[0], per_side=PER_SIDE, results=summary), fh, indent=2)
    print("PROBE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
