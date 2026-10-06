"""Prepare, train, and gate the post-E5 three-arm data ladder.

    py -3 -u tools/post_e5_driver.py

E5 made the evaluation zero point comparable and established ``v19_B`` as the
bar.  This ladder now changes one data input at a time while holding the v19
training recipe fixed:

    base      combined_v19_K              frozen-corpus control
    e1500     base + ps_monster_e1500      policy-only human teacher
    owner41   base + 41 owner games        normal policy and value labels

Every binding gate uses the native engine at the measured 3,200-simulation
operating-point knee.  Gate thresholds remain constants in ``tools/gate.py``;
this driver changes search effort, never a verdict threshold.

The driver is deliberately restart-safe.  It reuses a complete prepared
corpus, refuses a partial one, and only reuses a trained model when
``--skip-trained`` is explicit.  Child output is inherited so ``tools/runs.py``
can expose live progress for the unattended chain.
"""
import argparse
import json
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = [sys.executable, "-u"]
REFERENCE_RAW = "data/raw/combined_v19_K"
GATE_SIMS = 3200
GATE_SEED = 20260801

ARMS = {
    "base": {
        "raw": REFERENCE_RAW,
        "processed": "data/processed/combined_v19_K_r50h60",
        "model": "models/candidates/post_e5_base",
    },
    "e1500": {
        "raw": "data/raw/combined_v19_K_e1500",
        "processed": "data/processed/combined_v19_K_e1500_r50h60",
        "model": "models/candidates/post_e5_e1500",
    },
    "owner41": {
        "raw": "data/raw/combined_v19_K_owner41",
        "processed": "data/processed/combined_v19_K_owner41_r50h60",
        "model": "models/candidates/post_e5_owner41",
    },
}

# Frozen from the v19 K run. Only --data-dir and --model-dir vary.
RECIPE = [
    "--epochs", "30", "--patience", "10",
    "--batch-size", "256", "--lr", "0.002",
    "--policy-loss-weight", "1.0", "--weight-decay", "0.0001",
    "--grad-clip", "1.0", "--warmup-epochs", "3",
    "--warmup-start-factor", "0.1", "--seed", "42",
    "--target", "game_result", "--value-head", "scalar",
    "--select-metric", "decisive", "--stem-channels", "64",
]

PROCESS_RECIPE = [
    "--seed", "42", "--value-floor", "0.5", "--value-horizon", "60",
    "--value-discount-mode", "near_mate", "--channels", "15",
]

PROCESSED_FILES = (
    "positions.npy", "mcts_values.npy", "game_results.npy", "policies.npy",
    "policy_weights.npy", "value_weights.npy", "splits.npz",
    "split_game_ids.json",
)


def absolute(path):
    return os.path.join(ROOT, path)


def run(cmd):
    """Run a child with inherited output and return its exit code."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    print(f"[post-e5] exit={proc.returncode} in {time.time() - started:.0f}s",
          flush=True)
    return proc.returncode


def newest(directory, prefix, after):
    best = None
    best_mtime = after
    if not os.path.isdir(directory):
        return None
    for name in os.listdir(directory):
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        mtime = os.path.getmtime(path)
        if mtime > best_mtime:
            best, best_mtime = path, mtime
    return best


def _validate_e1500_manifest(raw_dir):
    manifest_path = os.path.join(raw_dir, "corpus_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"existing e1500 corpus has no manifest: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("merged_sources", [])
    expected = [entry for entry in entries
                if entry.get("source") == "data/raw/ps_monster_e1500"
                and entry.get("as") == "ps_monster_e1500"]
    if len(expected) != 1 or expected[0].get("value_weight") != 0.0:
        raise RuntimeError("existing e1500 corpus is not the policy-only arm")
    if not expected[0].get("dedupe_against_base"):
        raise RuntimeError("existing e1500 corpus did not deduplicate the base")


def prepare_raw(arm):
    spec = ARMS[arm]
    raw_dir = absolute(spec["raw"])
    if arm == "e1500":
        if not os.path.exists(raw_dir):
            code = run(PY + [
                "tools/merge_source.py",
                "--base-dir", REFERENCE_RAW,
                "--source", "data/raw/ps_monster_e1500",
                "--as", "ps_monster_e1500",
                "--value-weight", "0",
                "--dedupe-against-base",
                "--out-dir", spec["raw"],
            ])
            if code:
                raise RuntimeError("e1500 corpus merge failed")
        _validate_e1500_manifest(raw_dir)
    elif not os.path.isdir(raw_dir):
        raise RuntimeError(f"missing raw corpus for {arm}: {spec['raw']}")


def audit_raw(arm):
    if arm == "base":
        return
    code = run(PY + [
        "tools/pretrain_check.py", ARMS[arm]["raw"],
        "--reference", REFERENCE_RAW,
        "--value-discount-mode", "near_mate",
        "--value-floor", "0.5", "--value-horizon", "60",
    ])
    if code:
        raise RuntimeError(f"pretrain audit failed for {arm}")


def prepare_processed(arm):
    spec = ARMS[arm]
    processed = absolute(spec["processed"])
    if os.path.exists(processed):
        missing = [name for name in PROCESSED_FILES
                   if not os.path.isfile(os.path.join(processed, name))]
        if missing:
            raise RuntimeError(
                f"partial processed corpus for {arm}; missing {', '.join(missing)}")
        print(f"[post-e5] {arm}: reusing complete {spec['processed']}", flush=True)
        return
    code = run(PY + [
        "src/data_processor.py", "--raw-dir", spec["raw"],
        "--output-dir", spec["processed"],
    ] + PROCESS_RECIPE)
    if code:
        raise RuntimeError(f"data processing failed for {arm}")


def prepare_arm(arm):
    prepare_raw(arm)
    audit_raw(arm)
    prepare_processed(arm)


def train_arm(arm, skip_trained=False):
    spec = ARMS[arm]
    model_dir = absolute(spec["model"])
    checkpoint = os.path.join(model_dir, "best_value_net.pt")
    if os.path.isfile(checkpoint):
        if skip_trained:
            print(f"[post-e5] {arm}: reusing existing checkpoint", flush=True)
            return
        raise RuntimeError(
            f"checkpoint already exists for {arm}; pass --skip-trained to reuse it")
    if os.path.exists(model_dir):
        raise RuntimeError(f"partial model directory exists for {arm}: {spec['model']}")
    code = run(PY + [
        "src/train.py", "--data-dir", spec["processed"],
        "--model-dir", spec["model"],
    ] + RECIPE)
    if code or not os.path.isfile(checkpoint):
        raise RuntimeError(f"training failed for {arm}")


def gate_arm(arm, workers=None):
    spec = ARMS[arm]
    checkpoint = os.path.join(spec["model"], "best_value_net.pt")
    started = time.time() - 1
    command = PY + [
        "tools/gate.py", "--model", checkpoint, "--protocol", "full",
        "--engine", "native", "--sims", str(GATE_SIMS),
        "--seed", str(GATE_SEED),
    ]
    if workers is not None:
        command += ["--workers", str(workers)]
    # A non-zero gate exit is an experimental result, not a chain failure.
    run(command)
    artifact = newest(absolute("benchmarks"), f"gate_{arm}", started)
    # Artifact candidates are named from the model directory.
    if artifact is None:
        artifact = newest(absolute("benchmarks"), f"gate_post_e5_{arm}", started)
    if artifact is None:
        raise RuntimeError(f"gate produced no artifact for {arm}")
    with open(artifact, encoding="utf-8") as handle:
        verdict = json.load(handle)
    if verdict.get("thresholds", {}).get("sims") != GATE_SIMS:
        raise RuntimeError(f"gate artifact for {arm} did not use {GATE_SIMS} sims")
    return artifact, verdict


def compact_verdict(artifact, verdict):
    return {
        "verdict": verdict.get("verdict"),
        "failures": verdict.get("failures"),
        "bar": verdict.get("bar"),
        "sims": verdict.get("thresholds", {}).get("sims"),
        "legs": {
            name: {
                "a_score": leg.get("a_score"),
                "W": leg.get("a_as_white", {}).get("score"),
                "B": leg.get("a_as_black", {}).get("score"),
            }
            for name, leg in verdict.get("legs", {}).items()
        },
        "artifact": os.path.relpath(artifact, ROOT).replace("\\", "/"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", default="base,e1500,owner41")
    parser.add_argument("--skip-trained", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        parser.error(f"unknown arms: {', '.join(unknown)}")

    print(f"[post-e5] arms: {', '.join(arms)}", flush=True)
    print(f"[post-e5] binding gate: native, {GATE_SIMS} sims, bar v19_B",
          flush=True)
    for arm in arms:
        spec = ARMS[arm]
        print(f"  {arm:8s} {spec['raw']} -> {spec['model']}", flush=True)
    if args.dry_run:
        print("[post-e5] dry run, nothing executed", flush=True)
        return

    for arm in arms:
        print(f"\n{'=' * 72}\n[post-e5] PREPARE {arm}\n{'=' * 72}", flush=True)
        prepare_arm(arm)
    if args.prepare_only:
        print("[post-e5] preparation complete", flush=True)
        return

    results = {}
    for arm in arms:
        print(f"\n{'=' * 72}\n[post-e5] TRAIN + GATE {arm}\n{'=' * 72}",
              flush=True)
        train_arm(arm, args.skip_trained)
        artifact, verdict = gate_arm(arm, args.workers)
        results[arm] = compact_verdict(artifact, verdict)
        print(f"[post-e5] {arm} -> {json.dumps(results[arm])}", flush=True)

    summary = {
        "campaign": "post_e5_data_ladder",
        "arms": results,
        "gate": {"engine": "native", "sims": GATE_SIMS,
                 "seed": GATE_SEED, "bar": gate.BAR},
        "training_recipe": RECIPE,
        "processing_recipe": PROCESS_RECIPE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = absolute(
        f"benchmarks/post_e5_ladder_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\n[post-e5] SUMMARY\n{json.dumps(summary, indent=2)}", flush=True)
    print(f"[post-e5] saved {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 - persist the abort reason
        print(f"[post-e5] CHAIN ABORTED: {type(exc).__name__}: {exc}", flush=True)
        raise
