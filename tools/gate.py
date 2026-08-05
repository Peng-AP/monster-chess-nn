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
* **The bar is the strongest engine on record, `v19_B`.** E5 re-measured the
  ladder under captures-only scoring on the native engine; v19_B beat the
  numbered incumbent v19 by 0.575 on each of two independent 40-game reads.
  A candidate must beat v19_B on aggregate, clear the per-side floor on every
  leg, and then beat v19_B *again* on a fresh opening seed.
  `fresh_start_v18_ramp` remains a floor-bearing leg -- a distinct style.
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
AGGREGATE_MIN = 0.50              # must be strictly beaten
SIMS = 400

# Owner, 2026-08-01: "Every model should be better than the last, definitively.
# Last should be ramp."
#
# Until 2026-08-02 the bar was fresh_start_v18_ramp: v17 held the version
# number, but ramp was the strongest engine on record, so a candidate had to
# beat *it*.
# 2026-08-04 E5: captures-only native re-baseline. v19_B beat v19 0.575 twice
# on disjoint 40-game samples and led every shared comparison. The strongest
# engine, not the numbered release label, is the bar. No threshold moved.
BAR = "vs_v19_B"
AGGREGATE_LEGS = ("vs_v19_B", "vs_ramp")

NUMBERED_INCUMBENT = os.path.join(
    ROOT, "models", "fresh_start_v19", "best_value_net.pt")
BAR_MODEL = os.path.join(
    ROOT, "models", "candidates", "v19_B", "best_value_net.pt")
SPARRING = os.path.join(ROOT, "models", "rejected", "fresh_start_v18_ramp",
                        "best_value_net.pt")

# "Definitively" has a measured meaning here. Two independent 40-game samples
# of the SAME matchup (ramp vs v17) came out 0.575 and 0.725 on 2026-08-01 --
# per-leg variance is dominated by the sampled opening set, so one leg above
# 0.50 is not a definitive anything. A candidate that passes therefore replays
# the bar leg on a different opening seed and must clear it twice.
CONFIRM_LEG = "vs_v19_B_confirm"
CONFIRM_SEED_OFFSET = 424242

# (leg name, opponent path or None for the heuristic anchor, games)
FULL_LEGS = [
    ("vs_v19_B", BAR_MODEL, 40),
    ("vs_ramp", SPARRING, 40),
    ("anchor", None, 20),
]
QUICK_LEGS = [
    ("vs_v19_B", BAR_MODEL, 4),
    ("vs_ramp", SPARRING, 4),
    ("anchor", None, 2),
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
    for name in AGGREGATE_LEGS + (CONFIRM_LEG,):
        leg = legs.get(name)
        if leg is not None and leg["a_score"] <= AGGREGATE_MIN:
            failures.append(
                f"{name} aggregate {leg['a_score']:.4f} <= {AGGREGATE_MIN:.2f}")

    # The bar leg is not optional. A run that never played it cannot pass,
    # however good the rest looks.
    if legs.get(BAR) is None:
        failures.append(f"{BAR} leg was not played")

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


def run_gate(model, protocol="full", seed=20260801, workers=None, sims=SIMS,
             engine=None):
    spec = FULL_LEGS if protocol == "full" else QUICK_LEGS
    from match import run_match

    legs = {}

    def play(name, opponent, games, leg_seed):
        print(f"[gate] leg {name}: {games} games vs "
              f"{os.path.basename(os.path.dirname(opponent)) if opponent else 'heuristic'}",
              flush=True)
        t0 = time.time()
        legs[name] = run_match(model, opponent, games, sims, leg_seed,
                               workers=workers, engine=engine)
        print(f"[gate]   {name}: a_score={legs[name]['a_score']} "
              f"W={legs[name]['a_as_white']['score']} "
              f"B={legs[name]['a_as_black']['score']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    for i, (name, opponent, games) in enumerate(spec):
        play(name, opponent, games, seed + 100 * i)

    verdict, failures, totals = evaluate_legs(legs)

    # Only a candidate that has already cleared everything earns the
    # confirmation leg -- there is nothing to confirm about a failure, and the
    # 23 minutes are better spent on the next arm.
    if verdict == "PASS":
        # Replay the BAR leg specifically -- look its opponent up rather than
        # naming one, so moving the bar can never leave this confirming the
        # wrong model (it did, briefly, when the bar moved from ramp to v19).
        bar_spec = {n: (o, g) for n, o, g in spec}[BAR]
        print("[gate] provisional PASS -- replaying the bar leg on a fresh "
              "opening seed", flush=True)
        play(CONFIRM_LEG, bar_spec[0], bar_spec[1], seed + CONFIRM_SEED_OFFSET)
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
        "bar": BAR,
        "confirmed": CONFIRM_LEG in legs,
        "thresholds": {
            "per_side_floor": PER_SIDE_FLOOR,
            "aggregate_min_exclusive": AGGREGATE_MIN,
            "aggregate_legs": list(AGGREGATE_LEGS),
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
    ap.add_argument("--engine", choices=("python", "native"), default=None,
                    help="search engine for every leg; defaults to "
                         "MONSTER_ENGINE or python. Thresholds are untouched.")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--sims", type=int, default=SIMS,
                    help="search simulations per move (default: %(default)s); "
                         "verdict thresholds are unchanged")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    if not os.path.exists(args.model):
        ap.error(f"no such model: {args.model}")

    if args.sims <= 0:
        ap.error("--sims must be positive")

    out = run_gate(args.model, args.protocol, args.seed, args.workers,
                   sims=args.sims, engine=args.engine)

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
