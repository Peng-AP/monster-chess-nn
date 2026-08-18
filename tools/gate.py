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
* **The bar is the strongest promoted engine on record, `v23`.** A candidate
  must beat v23 on aggregate, clear the per-side floor on every leg, and then
  beat v23 *again* on a fresh opening seed.
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
# 2026-08-05: the owner promoted the Wide64 LC0 successor as v20 after two
# calibrated reads improved both colors over the approved 32-channel model.
# The strongest engine, not a historical label, is the bar. No threshold moved.
# 2026-08-06: the owner playtested the gen-5 teacher-3200 epoch-2 candidate
# ("very strong player") and promoted it as v21. It had already passed this
# gate against v20 and confirmed on a disjoint opening seed. The bar follows
# the strongest engine, so it moves to v21. No threshold moved.
# 2026-08-07: the owner promoted the from-scratch epoch-4 model as **v21b** --
# "it'll be the gate but I'm not impressed enough for it to be 22". So the bar
# and the version number separate again, exactly as they did when v17 held the
# number and v18_ramp was the bar. v21b cleared the first 200-game bar leg
# (pooled 0.5713 over 400 games, z=+2.85) and beat v21 over 800 games. No
# threshold moved.
# 2026-08-16: the owner explicitly promoted the fully gated Gen9 checkpoint as
# v22. It also beat the prior v21b bar directly over 400 paired games. The
# release and strength bar are aligned again; thresholds remain unchanged.
# 2026-08-17: the owner promoted the fully gated generation-15 checkpoint as
# v23, the first release of the BOOTSTRAP series. The version number continues
# from v22 so the release ladder stays comparable; the directory prefix changes
# because the lineage did. It is +130.4 Elo above v22 on a least-squares fit
# over all 15 pairs of the chain (largest residual 0.0226, no inversions), and
# scores 0.6567 against v22 directly over 600 games. Release and strength bar
# are aligned; thresholds remain unchanged.
BAR = "vs_v23"
AGGREGATE_LEGS = ("vs_v23", "vs_ramp")

NUMBERED_INCUMBENT = os.path.join(
    ROOT, "models", "bootstrap_v23", "best_value_net.pt")
BAR_MODEL = os.path.join(
    ROOT, "models", "bootstrap_v23", "best_value_net.pt")
SPARRING = os.path.join(ROOT, "models", "rejected", "fresh_start_v18_ramp",
                        "best_value_net.pt")

# "Definitively" has a measured meaning here. Two independent 40-game samples
# of the SAME matchup (ramp vs v17) came out 0.575 and 0.725 on 2026-08-01 --
# per-leg variance is dominated by the sampled opening set, so one leg above
# 0.50 is not a definitive anything. A candidate that passes therefore replays
# the bar leg on a different opening seed and must clear it twice.
CONFIRM_LEG = "vs_v23_confirm"
CONFIRM_SEED_OFFSET = 424242

# The bar played against ITSELF on the bar leg's own book block. Its true score
# is 0.5000 by construction, so whatever colour split it returns is the block's
# bias and nothing else.
#
# Added 2026-08-18 on the owner's approval, after a per-colour reading misled
# three times in one day. Measured then: gen16 against itself scored White
# 0.4437 over 800 games on one block (3.6 SE from even) and White 0.3000 on a
# 40-game block of the same book. Read against 0.4437 rather than 0.50, gen17's
# "alarming" 0.4338 was at par. Block bias also puts the absolute 0.40 White
# floor only 0.044 below neutral on a Black-favouring block.
#
# It is DIAGNOSTIC. It does not enter `legs`, the aggregate totals, or the
# verdict: calibrating a threshold could let a candidate through that the
# unchanged rule rejects, and no threshold moves without the owner saying so.
# It reuses the bar leg's block deliberately -- same openings is the point --
# so it costs no extra book entries, only time.
CALIBRATION_LEG = "bar_selfmatch"
CALIBRATION_SEED_OFFSET = 848484
SEED_BASE_FOR_TEST = 20260801   # the --seed default; named so tests share it

# (leg name, opponent path or None for the heuristic anchor, games)
#
# 2026-08-07: the bar leg went 40 -> 200 games. At 40 the per-side floor was
# checked on 20 games (SE 0.112), and on that day THREE candidates cleared the
# gate and then scored 0.4800 / 0.4825 / 0.5019 over 800 games each -- three
# false positives out of three passes (REPORT 23). At 200 the bar leg's
# per-side sample is 100 games (SE 0.05) and the confirmation replay matches
# it. No threshold moved; only the evidence behind them. The ramp and anchor
# legs stay small: they are floor checks that have never been the deciding
# leg, and enlarging them would triple gate cost for nothing.
#
# 2026-08-16: the bar leg went 200 -> 800 games, on the owner's instruction
# that a pass be "high power and absolutely confirmed". At 200 the aggregate
# SE is 0.022, so a candidate truly at 0.53 sits only 1.4 SE above the 0.50
# threshold -- barely better than a coin flip, which is why single blocks kept
# flipping verdicts. At 800 the aggregate SE is 0.011 (2.7 SE) and the
# per-colour SE is 0.0157, putting a true Black of 0.45 at 3.2 SE above the
# floor. A larger leg also spans four times as many book entries, which damps
# the block-to-block colour swing measured at 0.280-0.470 for V22 against
# itself. NO THRESHOLD MOVED -- only the evidence behind them, and it moved
# upward. The 2026-08-16 speedups are what made this affordable: 800 games at
# 400 sims is about 8 minutes.
FULL_LEGS = [
    ("vs_v23", BAR_MODEL, 800),
    ("vs_ramp", SPARRING, 40),
    ("anchor", None, 20),
]
QUICK_LEGS = [
    ("vs_v23", BAR_MODEL, 4),
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


def leg_seed_stride(spec):
    """Seed distance between legs, wide enough that none can replay another.

    `run_match` draws per-game seeds at leg_seed + i (White) and
    leg_seed + 1000 + i (Black), so the stride must clear both the leg's own
    game count and that 1000 offset. Exposed rather than inlined because
    `tests/test_match_seed_separation.py` has to check the SAME number the gate
    uses -- it previously hardcoded 100 and silently drifted once the bar leg
    grew, which is exactly the failure that test exists to prevent.
    """
    return max(100, 2 * (max(games for _n, _o, games in spec) + 1000))


def book_leg_offsets(spec, base_offset=0):
    """Allocate every NN leg a disjoint block of book entries.

    Returns ({leg_name: first_entry}, entries_required).

    Under a book, independence between legs lives in the ENTRY INDEX, not in
    the seed. Two legs at different seeds but the same offset replay identical
    openings and agree by construction, which would silently turn the
    confirmation replay -- the whole point of which is a fresh sample -- into a
    re-print of the bar leg. The confirmation is allocated last so it can never
    overlap a leg that ran before it.

    The anchor leg is deliberately absent: it keeps sampled openings because it
    is the heuristic yardstick, and its value is comparability with every
    anchor score on record. Heuristic tie-breaks already diversify it (see
    match.resolve_opening_temp_plies).
    """
    offsets, cursor = {}, int(base_offset)
    for name, opponent, games in spec:
        if opponent:
            offsets[name] = cursor
            cursor += games // 2
    offsets[CONFIRM_LEG] = cursor
    cursor += {name: games for name, _o, games in spec}[BAR] // 2
    return offsets, cursor


def run_gate(model, protocol="full", seed=20260801, workers=None, sims=SIMS,
             engine=None, bar_model=None, sparring_model=None,
             stall_timeout=600.0, book=None, book_offset=0):
    base_spec = FULL_LEGS if protocol == "full" else QUICK_LEGS
    bar_model = bar_model or BAR_MODEL
    sparring_model = sparring_model or SPARRING
    # Keep the named protocol and thresholds fixed while allowing an iterative
    # bootstrap run to point the bar at its current promoted champion.
    spec = []
    for name, opponent, games in base_spec:
        if name == BAR:
            opponent = bar_model
        elif name == "vs_ramp":
            opponent = sparring_model
        spec.append((name, opponent, games))
    from match import load_book, run_match

    legs = {}

    book_offsets = {}
    if book:
        book_offsets, needed = book_leg_offsets(spec, book_offset)
        n_entries = len(load_book(book)[0])
        if needed > n_entries:
            raise SystemExit(
                f"gate needs {needed} book entries but {book} has "
                f"{n_entries}; rebuild it with --entries {needed} or more")

    def play(name, opponent, games, leg_seed):
        print(f"[gate] leg {name}: {games} games vs "
              f"{os.path.basename(os.path.dirname(opponent)) if opponent else 'heuristic'}",
              flush=True)
        t0 = time.time()
        legs[name] = run_match(model, opponent, games, sims, leg_seed,
                               workers=workers, engine=engine,
                               stall_timeout=stall_timeout,
                               book=book if name in book_offsets else None,
                               book_offset=book_offsets.get(name, 0))
        print(f"[gate]   {name}: a_score={legs[name]['a_score']} "
              f"W={legs[name]['a_as_white']['score']} "
              f"B={legs[name]['a_as_black']['score']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Leg seeds must not overlap. run_match draws per-game seeds at
    # leg_seed + i (White) and leg_seed + 1000 + i (Black), so a stride of 100
    # is only safe while legs stay small: at ~100 games a leg's White seeds
    # reach the next leg's base, and past that its Black seeds collide too.
    # Today's legs are 40/40/20 and the sets are disjoint, but the margin is 81
    # -- small enough that raising a leg's game count would silently make two
    # legs replay the same openings and look like independent agreement.
    leg_stride = leg_seed_stride(spec)
    for i, (name, opponent, games) in enumerate(spec):
        play(name, opponent, games, seed + leg_stride * i)

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

    # Calibration last: it never changes the verdict, so a candidate that is
    # going to fail should not wait nine minutes to find out.
    calibration = None
    if protocol == "full":
        bar_spec = {n: (o, g) for n, o, g in spec}[BAR]
        print(f"[gate] leg {CALIBRATION_LEG}: {bar_spec[1]} games, bar vs "
              f"itself on the bar leg's own block (diagnostic)", flush=True)
        t0 = time.time()
        calibration = run_match(
            bar_model, bar_model, bar_spec[1], sims,
            seed + CALIBRATION_SEED_OFFSET,
            workers=workers, engine=engine, stall_timeout=stall_timeout,
            book=book if BAR in book_offsets else None,
            book_offset=book_offsets.get(BAR, 0))
        white = calibration["a_as_white"]["score"]
        black = calibration["a_as_black"]["score"]
        print(f"[gate]   {CALIBRATION_LEG}: W={white} B={black} "
              f"(true value 0.5/0.5 -- the gap is this block's bias) "
              f"({time.time() - t0:.0f}s)", flush=True)
        for name in (BAR, CONFIRM_LEG):
            leg = legs.get(name)
            if not leg:
                continue
            leg["calibrated"] = {
                "white": round(leg["a_as_white"]["score"] - white, 4),
                "black": round(leg["a_as_black"]["score"] - black, 4),
                "baseline_white": white, "baseline_black": black,
            }
            print(f"[gate]   {name} vs baseline: "
                  f"W={leg['calibrated']['white']:+.4f} "
                  f"B={leg['calibrated']['black']:+.4f}", flush=True)

    binding = protocol == "full"
    return {
        "calibration": calibration,
        "candidate": os.path.basename(os.path.dirname(model)),
        "model": os.path.relpath(model, ROOT),
        "protocol": protocol,
        "binding": binding,
        "verdict": verdict if binding else "REHEARSAL",
        "raw_verdict": verdict,
        "failures": failures,
        "bar": BAR,
        "bar_model": os.path.relpath(bar_model, ROOT),
        "sparring_model": os.path.relpath(sparring_model, ROOT),
        "confirmed": CONFIRM_LEG in legs,
        # None marks a verdict measured under temperature-sampled openings.
        # Book and non-book scores are different regimes and do not compare.
        "book": book,
        "book_base_offset": int(book_offset) if book else None,
        "book_offsets": book_offsets or None,
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
    ap.add_argument("--stall-timeout", type=float, default=600.0,
                    help="fail if no match game completes for this many seconds")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    ap.add_argument("--bar-model", default=BAR_MODEL,
                    help="current champion used for both bar legs")
    ap.add_argument("--sparring-model", default=SPARRING,
                    help="distinct-style floor-bearing opponent")
    ap.add_argument("--report-path", default=None,
                    help="write the report to this exact path")
    ap.add_argument("--book", default=None,
                    help="paired opening book (tools/make_book.py) for the "
                         "NN legs. The anchor leg keeps sampled openings. "
                         "Scores do NOT compare to non-book gate results.")
    ap.add_argument("--book-offset", type=int, default=0,
                    help="first book entry reserved for this gate (allows a "
                         "screen and gate to use disjoint blocks)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        ap.error(f"no such model: {args.model}")
    if not os.path.exists(args.bar_model):
        ap.error(f"no such bar model: {args.bar_model}")
    if not os.path.exists(args.sparring_model):
        ap.error(f"no such sparring model: {args.sparring_model}")

    if args.sims <= 0 or args.stall_timeout <= 0 or args.book_offset < 0:
        ap.error("--sims and --stall-timeout must be positive; "
                 "--book-offset must be non-negative")

    out = run_gate(args.model, args.protocol, args.seed, args.workers,
                   sims=args.sims, engine=args.engine,
                   bar_model=args.bar_model,
                   sparring_model=args.sparring_model,
                   stall_timeout=args.stall_timeout, book=args.book,
                   book_offset=args.book_offset)

    if args.report_path:
        path = os.path.abspath(args.report_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
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
