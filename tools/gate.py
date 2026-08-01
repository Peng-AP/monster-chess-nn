"""The gate protocol, in one place. Run a candidate through every leg, emit one verdict.

    py -3 tools/gate.py --model models/candidates/v19_K/best_value_net.pt

Why this exists (DIRECTIVE Phase 0):

* Every arm must face *identical* legs. Hand-assembled gate drivers have twice
  produced runs that were not comparable, and once read benchmark.py's schema
  for a match.py file and scored every leg `None` (HANDOFF SS10.1). All legs
  here go through match.run_match, which is the single producer of the
  a_score / a_as_white / a_as_black shape.
* The thresholds are constants, not flags. The owner's binding rule is that a
  threshold is never weakened to let a recipe through, so there is deliberately
  no way to pass one on the command line.
* Per-side scores are the verdict; aggregates are reported but never decide a
  leg. Aggregates masking a per-side collapse has burned this project four
  times (law 8).

`--protocol quick` shrinks every leg for rehearsal (HANDOFF SS10.1: rehearse the
chain at tiny scale first). Quick runs are stamped `binding: false` and their
verdict is `REHEARSAL`, never PASS/FAIL -- a smoke test must not be able to
masquerade as a gate result.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

# --- The protocol. Constants on purpose; see the module docstring. ---
PER_SIDE_FLOOR = 0.40
V17_AGGREGATE_MIN = 0.50          # must be strictly beaten
SIMS = 400

INCUMBENT = os.path.join(ROOT, "models", "fresh_start_v17", "best_value_net.pt")
SPARRING = os.path.join(ROOT, "models", "rejected", "fresh_start_v18_ramp",
                        "best_value_net.pt")

# (leg name, opponent path or None for the heuristic anchor, games, decides_aggregate)
FULL_LEGS = [
    ("vs_v17", INCUMBENT, 40, True),
    ("vs_ramp", SPARRING, 40, False),
    ("anchor", None, 20, False),
]
QUICK_LEGS = [
    ("vs_v17", INCUMBENT, 4, True),
    ("vs_ramp", SPARRING, 4, False),
    ("anchor", None, 2, False),
]


def _side_score(block):
    """A side's score, or None when the leg produced no games on that side."""
    return block.get("score") if block else None


def evaluate_legs(legs):
    """Apply the protocol to finished legs. Returns (verdict, failures, totals).

    Split out from the run loop so the thresholds are unit-testable without
    playing a single game.
    """
    failures = []
    for name, leg in legs.items():
        for side in ("a_as_white", "a_as_black"):
            score = _side_score(leg.get(side))
            label = side.replace("a_as_", "")
            if score is None:
                failures.append(f"{name} {label} leg has no games")
            elif score < PER_SIDE_FLOOR:
                failures.append(
                    f"{name} {label} leg {score:.4f} < {PER_SIDE_FLOOR:.2f}")
    v17 = legs.get("vs_v17")
    if v17 is not None and v17["a_score"] <= V17_AGGREGATE_MIN:
        failures.append(
            f"vs_v17 aggregate {v17['a_score']:.4f} <= {V17_AGGREGATE_MIN:.2f}")

    # Process note SS12: read the per-side totals across ALL legs against the
    # noise floor before believing any direction. A single leg moving is one
    # game in ten and has produced a false win before (HANDOFF SS4.1).
    totals = {}
    for side in ("white", "black"):
        key = f"a_as_{side}"
        pts = sum(leg[key]["wins"] + 0.5 * leg[key]["draws"]
                  for leg in legs.values() if leg.get(key))
        n = sum(leg[key]["games"] for leg in legs.values() if leg.get(key))
        totals[side] = {
            "points": round(pts, 2), "games": n,
            "score": round(pts / n, 4) if n else None,
            # SE of a total in game-points if the true rate were 0.5.
            "se_points": round(math.sqrt(n * 0.25), 2) if n else None,
        }
    verdict = "PASS" if not failures else "FAIL"
    return verdict, failures, totals


def run_gate(model, protocol="full", seed=20260801, workers=None, sims=SIMS):
    spec = FULL_LEGS if protocol == "full" else QUICK_LEGS
    from match import run_match

    legs = {}
    for i, (name, opponent, games, _agg) in enumerate(spec):
        print(f"[gate] leg {name}: {games} games vs "
              f"{os.path.basename(os.path.dirname(opponent)) if opponent else 'heuristic'}",
              flush=True)
        t0 = time.time()
        # Distinct seed per leg so the legs are independent samples rather than
        # the same openings replayed against three opponents.
        legs[name] = run_match(model, opponent, games, sims, seed + 100 * i,
                               workers=workers)
        print(f"[gate]   {name}: a_score={legs[name]['a_score']} "
              f"W={legs[name]['a_as_white']['score']} "
              f"B={legs[name]['a_as_black']['score']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    verdict, failures, totals = evaluate_legs(legs)
    binding = protocol == "full"
    return {
        "candidate": os.path.basename(os.path.dirname(model)),
        "model": os.path.relpath(model, ROOT),
        "protocol": protocol,
        "binding": binding,
        "verdict": verdict if binding else "REHEARSAL",
        "raw_verdict": verdict,
        "failures": failures,
        "thresholds": {
            "per_side_floor": PER_SIDE_FLOOR,
            "v17_aggregate_min_exclusive": V17_AGGREGATE_MIN,
            "sims": sims,
        },
        "per_side_totals_across_legs": totals,
        "legs": legs,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="candidate .pt")
    ap.add_argument("--protocol", choices=("full", "quick"), default="full",
                    help="quick = tiny rehearsal, verdict is non-binding")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    if not os.path.exists(args.model):
        ap.error(f"no such model: {args.model}")

    out = run_gate(args.model, args.protocol, args.seed, args.workers)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "gate" if out["binding"] else "gate_rehearsal"
    path = os.path.join(
        args.out_dir,
        f"{tag}_{out['candidate']}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({k: v for k, v in out.items() if k != "legs"}, indent=2))
    print(f"\nVERDICT: {out['verdict']}")
    for reason in out["failures"]:
        print(f"  - {reason}")
    print(f"Saved to {path}")
    return 0 if out["raw_verdict"] == "PASS" else 1


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
