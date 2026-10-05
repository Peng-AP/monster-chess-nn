"""Which opponents expose a model's holes? (Input for choosing gen54's opponent pool.)

October 5, 2026 (owner: "My guess is broader opponents. But there is a limit to
worse play for the sake of another model"). An opponent is useful for training
if it finds lines the model loses, not merely if it is strong or weak. For each
candidate opponent this plays MODEL 100 games (50 per colour, sampled openings,
3,200 sims) and reports the score, the distinct-game score, the losses and the
number of DISTINCT losing endpoints (positions after the 16 sampled plies).

B2 and v27 are held out and never scanned: they must stay independent tests.

    py -3 tools/hole_scan.py --name gen53 --model models/candidates/bootstrap_main_gen_0053/arena_selected.pt
"""
import argparse
import collections
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

OPPONENTS = [
    ("v24", "models/bootstrap_v24/best_value_net.pt"),
    ("v25", "models/bootstrap_v25/best_value_net.pt"),
    ("v26", "models/bootstrap_v26/best_value_net.pt"),
    ("v28", "models/bootstrap_v28/best_value_net.pt"),
    ("gen48", "models/candidates/bootstrap_main_gen_0048/arena_selected.pt"),
    ("gen49", "models/candidates/bootstrap_main_gen_0049/arena_selected.pt"),
    ("v29", "models/bootstrap_v29/best_value_net.pt"),
    ("gen52A", "models/candidates/bootstrap_main_gen_0052/arena_selected.pt"),
    ("gen52B", "models/candidates/bootstrap_main_gen_0052_pool/arena_selected.pt"),
    ("gen52C", "models/candidates/bootstrap_main_gen_0052_poolcap/arena_selected.pt"),
    ("gen52L", "models/candidates/bootstrap_main_gen_0052_large/arena_selected.pt"),
    ("gen52LR", "models/candidates/bootstrap_main_gen_0052_large_ramp/arena_selected.pt"),
    ("gen52R", "models/candidates/bootstrap_main_gen_0052_ramp/arena_selected.pt"),
]
HELD_OUT = {"models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt", "models/bootstrap_v27/best_value_net.pt"}
GAMES, SIMS = 100, 3200
SEED_BASE, SEED_STRIDE = 4_150_000_000, 1_000_000   # 13 opponents -> below 2**32


def losing_lines(log):
    """Distinct opening endpoints in which MODEL (player a) lost, by colour."""
    out = {"white": collections.Counter(), "black": collections.Counter()}
    for line in open(log, encoding="utf-8"):
        if not line.strip().startswith("{"):
            continue
        r = json.loads(line)
        if "white_score" not in r:
            continue
        score = r["white_score"] if r["a_is_white"] else 1 - r["white_score"]
        if score < 0.5:
            out["white" if r["a_is_white"] else "black"][r["opening"]["fen"]] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    from match import run_match
    assert not {p for _, p in OPPONENTS} & HELD_OUT, "held-out models may not be scanned"
    out = os.path.join(ROOT, "benchmarks", f"hole_scan_{args.name}_20261005")
    os.makedirs(out, exist_ok=True)
    summary = {}
    for i, (opp, path) in enumerate(OPPONENTS):
        stem = os.path.join(out, f"{i:02d}_{args.name}_vs_{opp}")
        if not os.path.exists(stem + ".json"):
            print(f"SCAN {i + 1}/{len(OPPONENTS)}: {args.name} vs {opp}", flush=True)
            m = run_match(os.path.join(ROOT, args.model), os.path.join(ROOT, path), games=GAMES, sims=SIMS,
                          seed=SEED_BASE + i * SEED_STRIDE, opening_temp_plies=16, workers=args.workers,
                          engine="native", stall_timeout=1800.0, game_log=stem + ".jsonl",
                          resume=os.path.exists(stem + ".jsonl"))
            with open(stem + ".json.tmp", "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=2)
            os.replace(stem + ".json.tmp", stem + ".json")
        m = json.load(open(stem + ".json", encoding="utf-8"))
        lines = losing_lines(stem + ".jsonl")
        w, b = m["a_as_white"], m["a_as_black"]
        summary[opp] = dict(score=m["a_score"], as_white=w["score"], as_black=b["score"],
                            losses=dict(white=w["losses"], black=b["losses"]),
                            distinct_losing_endpoints=dict(white=len(lines["white"]), black=len(lines["black"])),
                            worst_endpoints=[dict(colour=c, fen=f, losses=n) for c in ("white", "black")
                                             for f, n in lines[c].most_common(3)])
        print(f"SCAN RESULT vs {opp}: {m['a_score']:.3f} | losses W {w['losses']} B {b['losses']} | distinct losing "
              f"endpoints W {len(lines['white'])} B {len(lines['black'])} ({time.strftime('%H:%M')})", flush=True)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(model=args.model, games=GAMES, sims=SIMS, opponents=summary), fh, indent=2)
    print("SCAN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
