"""Train and gate the v19 arms, one after another, unattended.

    py -3 -u phase3_driver.py --arms control,O,K,B

Every arm shares one recipe -- ramp labels (r50h60), scalar head, 15ch,
seed 42, 30 epochs / patience 10 (M1 measured best epoch 24 with an 80 cap, so
there is no headroom to buy) -- and differs only in its corpus. That is the
whole design: the corpus is the variable, and the ladder is built so each rung
attributes one change.

    control  combined_v17           the v18-era corpus, no owner games added
    O        combined_v19_base      + 27 owner games            (what D4 bought)
    K        combined_v19_K         + ps_monster, policy only   (knowledge)
    B        combined_v19_B         + ps_monster, full value    (belief)

K and B differ from each other by a single stamp, so K-vs-B is the
knowledge-vs-belief fork from HANDOFF SS7.1 with everything else held fixed.

Gating is `tools/gate.py`, unchanged and unweakened: the bar is ramp, the
per-side floor is 0.40, and a passing arm replays the bar leg on a fresh
opening seed. Exit status of a gate is not treated as failure of the run --
arms are *expected* to fail, that is what a gate is for.

Traps this driver is written around (HANDOFF SS10.1):
  * no stdout=PIPE on children -- output is inherited into the log so a
    multi-hour run stays watchable;
  * no sys.exit() inside the try -- SystemExit is a BaseException and would
    log the failure marker on a clean run;
  * each stage writes its own artifact, and the driver reads the artifact
    rather than parsing stdout.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = [sys.executable, "-u"]

# arm -> processed corpus
ARMS = {
    "control": "data/processed/combined_v17_r50h60",
    "O": "data/processed/combined_v19_base_r50h60",
    "K": "data/processed/combined_v19_K_r50h60",
    "B": "data/processed/combined_v19_B_r50h60",
}

# The frozen recipe. Only --data-dir and --model-dir vary between arms.
RECIPE = [
    "--epochs", "30", "--patience", "10",
    "--batch-size", "256", "--lr", "0.002",
    "--policy-loss-weight", "1.0", "--weight-decay", "0.0001",
    "--grad-clip", "1.0", "--warmup-epochs", "3",
    "--warmup-start-factor", "0.1", "--seed", "42",
    "--target", "game_result", "--value-head", "scalar",
    "--select-metric", "decisive", "--stem-channels", "64",
]


def run(cmd, cwd=ROOT):
    """Run a child with inherited stdout. Returns its exit code."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd)
    print(f"[driver] exit={proc.returncode} in {time.time() - t0:.0f}s", flush=True)
    return proc.returncode


def newest(pattern_dir, prefix, after):
    """Newest file in pattern_dir starting with prefix and modified after `after`."""
    best, best_m = None, after
    if not os.path.isdir(pattern_dir):
        return None
    for name in os.listdir(pattern_dir):
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        path = os.path.join(pattern_dir, name)
        m = os.path.getmtime(path)
        if m > best_m:
            best, best_m = path, m
    return best


def train_arm(arm, model_dir, skip_if_present):
    data_dir = ARMS[arm]
    if not os.path.isdir(os.path.join(ROOT, data_dir)):
        print(f"[driver] SKIP {arm}: {data_dir} does not exist", flush=True)
        return False
    if skip_if_present and os.path.exists(
            os.path.join(ROOT, model_dir, "best_value_net.pt")):
        print(f"[driver] {arm}: model already present, skipping training", flush=True)
        return True
    code = run(PY + ["src/train.py", "--data-dir", data_dir,
                     "--model-dir", model_dir] + RECIPE)
    return code == 0


def gate_arm(arm, model_dir):
    model = os.path.join(model_dir, "best_value_net.pt")
    if not os.path.exists(os.path.join(ROOT, model)):
        print(f"[driver] SKIP gate for {arm}: no model at {model}", flush=True)
        return None
    floor = time.time() - 1
    # Non-zero exit just means the arm did not pass; that is a result.
    run(PY + ["tools/gate.py", "--model", model, "--protocol", "full"])
    return newest(os.path.join(ROOT, "benchmarks"), "gate_", floor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="control,O,K,B")
    ap.add_argument("--skip-trained", action="store_true",
                    help="reuse an existing best_value_net.pt for an arm")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; known: {', '.join(ARMS)}")

    print(f"[driver] arms: {', '.join(arms)}", flush=True)
    for arm in arms:
        exists = os.path.isdir(os.path.join(ROOT, ARMS[arm]))
        print(f"  {arm:8s} {ARMS[arm]:45s} {'ok' if exists else 'MISSING'}", flush=True)
    if args.dry_run:
        print("[driver] dry run, nothing executed", flush=True)
        return

    results = {}
    for arm in arms:
        model_dir = f"models/candidates/v19_{arm}"
        print(f"\n{'=' * 70}\n[driver] ARM {arm}\n{'=' * 70}", flush=True)
        if not train_arm(arm, model_dir, args.skip_trained):
            results[arm] = {"error": "training failed or corpus missing"}
            continue
        verdict_path = gate_arm(arm, model_dir)
        if verdict_path and os.path.exists(verdict_path):
            with open(verdict_path, encoding="utf-8") as f:
                v = json.load(f)
            results[arm] = {
                "verdict": v.get("verdict"),
                "failures": v.get("failures"),
                "legs": {name: {"a_score": leg["a_score"],
                                "W": leg["a_as_white"]["score"],
                                "B": leg["a_as_black"]["score"]}
                         for name, leg in v.get("legs", {}).items()},
                "artifact": os.path.relpath(verdict_path, ROOT),
            }
        else:
            results[arm] = {"error": "no gate artifact produced"}
        print(f"\n[driver] {arm} -> {json.dumps(results[arm])}", flush=True)

    out = os.path.join(ROOT, "benchmarks",
                       f"phase3_summary_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[driver] SUMMARY\n{json.dumps(results, indent=2)}", flush=True)
    print(f"[driver] saved {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 - driver must log why it died
        print(f"[driver] CHAIN ABORTED: {type(exc).__name__}: {exc}", flush=True)
        raise
