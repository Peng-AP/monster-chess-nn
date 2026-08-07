"""Run a queue of training arms end to end, unattended, until a deadline.

Each arm is train -> gate -> record. The point is that the box never idles
between them and no arm needs a human to start it. Every arm trains from the
same incumbent and gates against the same bar, so the results are comparable to
each other and not just to the baseline.

Deliberate properties:

* **Resumable.** Results append to the summary after every arm, and a rerun
  skips arms already recorded. A crash costs one arm.
* **Deadline-aware.** An arm is only started if its estimated cost fits in the
  remaining budget, so a 10-hour window does not end with a half-trained
  candidate and nothing to show.
* **No threshold touching.** Gating goes through tools/gate.py unmodified;
  this driver chooses what to run, never what counts as passing.

    py -3 tools/overnight_sweep.py --deadline-hours 10

Arms are defined in ARMS below. Editing that list is the intended interface.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INCUMBENT = "models/fresh_start_v21/best_value_net.pt"
BASE = "data/processed/bootstrap_replay_main_gen_0005_teacher3200_full"
CONV_W4 = "data/processed/replay_gen5_plus_conversions_w4"
CONV_W1 = "data/processed/replay_gen5_plus_conversions_w1"

# The recipe v21 was trained with. Arms differ from it by their `extra` only,
# so any difference in outcome is attributable to that one change.
RECIPE = [
    "--epochs", "12", "--patience", "5", "--batch-size", "256",
    "--memory-map-data", "--lr", "0.0001", "--lr-gamma", "0.95",
    "--policy-loss-weight", "1.0", "--weight-decay", "0.0001",
    "--ema-decay", "0.999", "--grad-clip", "1.0",
    "--warmup-epochs", "1", "--warmup-start-factor", "0.1",
    "--seed", "42", "--target", "game_result", "--value-head", "scalar",
    "--select-metric", "decisive", "--select-relative-to-resume",
    "--max-policy-ce-regression", "0.01", "--max-side-top1-drop", "0.01",
    "--save-selection-snapshots", "--stem-channels", "64",
    "--policy-head", "attention", "--policy-attention-channels", "64",
]

ARMS = [
    # The only signal that ever moved Black for a principled reason: a target
    # that is unambiguous exactly where the ramp is not. It previously failed
    # by REPLACING the value head; here it is auxiliary at low weight.
    {"name": "capture_wdl_w003", "data": BASE,
     "extra": ["--aux-wdl-head", "--wdl-target", "capture_result",
               "--wdl-loss-weight", "0.03"]},
    # BLACK_WEIGHT_BALANCED has sat in data_processor at 1.75 and every recipe
    # has run with 1.0. A direct lever on the asymmetry needing no new data.
    {"name": "black_policy_w175", "data": BASE,
     "extra": ["--black-policy-weight", "1.75"]},
    # v22 used weight 4 and degraded from epoch 2. Weight 1 separates "the
    # conversion data is harmful" from "4x was too aggressive".
    {"name": "conversions_w1", "data": CONV_W1, "extra": []},
    # Same corpus as v22 at a third the learning rate. v22's decisive metric
    # peaked at epoch 1 then fell for five straight epochs, which is what an
    # over-hot LR on a corpus with a new source looks like.
    {"name": "conversions_w4_lowlr", "data": CONV_W4,
     "extra": ["--lr", "0.00003"]},
    # Combination, run last: only interesting if a component moved.
    {"name": "capture_wdl_plus_black_policy", "data": BASE,
     "extra": ["--aux-wdl-head", "--wdl-target", "capture_result",
               "--wdl-loss-weight", "0.03",
               "--black-policy-weight", "1.75"]},
]

# Phase 2. Arms ran ~15 min each rather than the estimated 85, leaving hours of
# the window idle, so the spare capacity goes to the one thing that has to be
# settled: capture_wdl_w003 passed by 0.34 SE, which is a pass, not a
# measurement. Replicates on fresh seeds are worth more than any new idea --
# a thin result that repeats three times is real, and one that does not is
# noise we would otherwise have promoted.
WDL = ["--aux-wdl-head", "--wdl-target", "capture_result",
       "--wdl-loss-weight", "0.03"]
PHASE2 = [
    # Same recipe, different seed. The only question that matters right now.
    {"name": "capture_wdl_w003_seed43", "data": BASE, "extra": WDL + ["--seed", "43"]},
    {"name": "capture_wdl_w003_seed44", "data": BASE, "extra": WDL + ["--seed", "44"]},
    # Is 0.03 a peak or a point on a flat line?
    {"name": "capture_wdl_w001", "data": BASE,
     "extra": ["--aux-wdl-head", "--wdl-target", "capture_result",
               "--wdl-loss-weight", "0.01"]},
    {"name": "capture_wdl_w006", "data": BASE,
     "extra": ["--aux-wdl-head", "--wdl-target", "capture_result",
               "--wdl-loss-weight", "0.06"]},
    {"name": "capture_wdl_w010", "data": BASE,
     "extra": ["--aux-wdl-head", "--wdl-target", "capture_result",
               "--wdl-loss-weight", "0.10"]},
    # A third replicate replaced the planned capture-WDL-on-conversions arm.
    # With the conversion corpus now 0/3 at the gate, an arm combining it with
    # capture_wdl would be confounded: a failure could not distinguish a bad
    # signal from a bad corpus, and an uninterpretable arm is worth less than
    # another clean read on the only result that has passed.
    {"name": "capture_wdl_w003_seed45", "data": BASE, "extra": WDL + ["--seed", "45"]},
]
ARMS = ARMS + PHASE2

# Measured 2026-08-07: arms run ~15 min (early stopping at epoch 6 plus a
# ~7 min gate), not the 85 first guessed. Leaving the estimate high would make
# the deadline guard refuse to start arms it had time for.
ESTIMATED_ARM_MINUTES = 25


def load_summary(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("arms", [])
    except Exception:
        return []


def save_summary(path, arms, started):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"incumbent": INCUMBENT,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S",
                                            time.localtime(started)),
                   "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "arms": arms}, fh, indent=2)


def run(cmd, label):
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    print(f"=== {label} exit={proc.returncode} "
          f"({(time.time() - t0) / 60:.1f}m) ===", flush=True)
    return proc.returncode


def gate_result(model_dir):
    """Newest gate artifact for this arm, as (verdict, black_confirm)."""
    import glob
    hits = sorted(glob.glob(os.path.join(
        ROOT, "benchmarks", f"gate_{os.path.basename(model_dir)}_*.json")))
    if not hits:
        return None
    with open(hits[-1], encoding="utf-8") as fh:
        d = json.load(fh)
    legs = d.get("legs") or {}
    out = {"verdict": d.get("verdict"), "artifact": os.path.basename(hits[-1]),
           "failures": d.get("failures", [])}
    for name, leg in legs.items():
        if leg:
            out[name] = {"overall": leg.get("a_score"),
                         "white": leg["a_as_white"]["score"],
                         "black": leg["a_as_black"]["score"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-hours", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary", default=os.path.join(
        ROOT, "benchmarks", "overnight_sweep.json"))
    ap.add_argument("--only", action="append", default=[],
                    help="run only these arm names (repeatable)")
    args = ap.parse_args()

    started = time.time()
    deadline = started + args.deadline_hours * 3600
    done = load_summary(args.summary)
    done_names = {a["name"] for a in done}
    arms = [a for a in ARMS if not args.only or a["name"] in args.only]

    print(f"sweep: {len(arms)} arms, deadline {args.deadline_hours}h, "
          f"incumbent {INCUMBENT}", flush=True)
    if done_names:
        print(f"resuming; already recorded: {sorted(done_names)}", flush=True)

    for arm in arms:
        if arm["name"] in done_names:
            print(f"[skip] {arm['name']} already recorded", flush=True)
            continue
        remaining = (deadline - time.time()) / 60
        if remaining < ESTIMATED_ARM_MINUTES:
            print(f"[stop] {remaining:.0f}m left, an arm needs "
                  f"~{ESTIMATED_ARM_MINUTES}m -- not starting {arm['name']}",
                  flush=True)
            break
        if not os.path.isdir(os.path.join(ROOT, arm["data"])):
            print(f"[skip] {arm['name']}: corpus missing ({arm['data']})",
                  flush=True)
            continue

        model_dir = f"models/candidates/{arm['name']}"
        rc = run([sys.executable, "-u", "src/train.py",
                  "--data-dir", arm["data"], "--model-dir", model_dir,
                  "--resume-from", INCUMBENT] + RECIPE + arm["extra"],
                 f"train {arm['name']}")
        record = {"name": arm["name"], "data": arm["data"],
                  "extra": arm["extra"], "train_exit": rc,
                  "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if rc == 0:
            run([sys.executable, "-u", "tools/gate.py",
                 "--model", f"{model_dir}/best_value_net.pt",
                 "--protocol", "full", "--engine", "native",
                 "--workers", str(args.workers)], f"gate {arm['name']}")
            record["gate"] = gate_result(model_dir)
        done.append(record)
        save_summary(args.summary, done, started)
        v = (record.get("gate") or {}).get("verdict", "no-gate")
        print(f"\n>>> {arm['name']}: {v}  "
              f"({(time.time() - started) / 3600:.1f}h elapsed)\n", flush=True)

    print("\n" + "=" * 72)
    print(f"{'arm':34} {'verdict':8} {'confirm B':>10} {'confirm all':>12}")
    print("-" * 72)
    for a in done:
        g = a.get("gate") or {}
        c = g.get("vs_v21_confirm") or {}
        print(f"{a['name']:34} {str(g.get('verdict')):8} "
              f"{c.get('black', float('nan')):10.3f} "
              f"{c.get('overall', float('nan')):12.3f}")
    print("=" * 72)
    print(f"summary -> {args.summary}")


if __name__ == "__main__":
    main()
