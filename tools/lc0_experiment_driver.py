"""Prepare and screen one-variable LC0-inspired training arms.

This is deliberately a non-binding screen. Every arm uses the exact full-value
``v19_B`` corpus and frozen training recipe; only the named flag changes.
Candidate A is compared with ``v19_B``, with Black score as the primary ranking
key and the established White/aggregate safeguards retained.

    py -3 -u tools/lc0_experiment_driver.py --arms control,moves_left,legal_mask,ema
"""
import argparse
import filecmp
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from config import DEFAULT_GAME_WORKERS  # noqa: E402
from gate import AGGREGATE_MIN, PER_SIDE_FLOOR  # noqa: E402

PY = [sys.executable, "-u"]
RAW = "data/raw/combined_v19_B"
PROCESSED = "data/processed/combined_v19_B_r50h60_aux"
PROCESSED_EXACT = "data/processed/combined_v19_B_r50h60_aux_exact"
REFERENCE_PROCESSED = "data/processed/combined_v19_B_r50h60"
CONTROL = "models/candidates/v19_B/best_value_net.pt"
MODEL_PREFIX = "lc0b_exact"

PROCESS_RECIPE = [
    "--seed", "42", "--value-floor", "0.5", "--value-horizon", "60",
    "--value-discount-mode", "near_mate", "--channels", "15",
]
TRAIN_RECIPE = [
    "--epochs", "30", "--patience", "10",
    "--batch-size", "256", "--lr", "0.002",
    "--policy-loss-weight", "1.0", "--weight-decay", "0.0001",
    "--grad-clip", "1.0", "--warmup-epochs", "3",
    "--warmup-start-factor", "0.1", "--seed", "42",
    "--target", "game_result", "--value-head", "scalar",
    "--select-metric", "decisive", "--stem-channels", "64",
]
ARMS = {
    "control": [],
    "moves_left": ["--moves-left-head", "--moves-left-loss-weight", "0.01"],
    "legal_mask": ["--legal-policy-mask"],
    "attention": ["--policy-head", "attention"],
    "ema": ["--ema-decay", "0.999"],
    "se": ["--use-se-blocks"],
}
ARM_DATA = {
    "control": REFERENCE_PROCESSED,
    "moves_left": PROCESSED_EXACT,
    "legal_mask": PROCESSED,
    "attention": REFERENCE_PROCESSED,
    "ema": REFERENCE_PROCESSED,
    "se": REFERENCE_PROCESSED,
}
REQUIRED_PROCESSED = (
    "positions.npy", "mcts_values.npy", "game_results.npy", "policies.npy",
    "policy_weights.npy", "value_weights.npy", "moves_left.npy",
    "moves_left_weights.npy", "legal_masks_packed.npy", "splits.npz",
    "split_game_ids.json",
)


def absolute(path):
    return os.path.join(ROOT, path)


def run(command):
    print(f"\n$ {' '.join(str(item) for item in command)}", flush=True)
    started = time.time()
    proc = subprocess.run([str(item) for item in command], cwd=ROOT)
    print(f"[lc0] exit={proc.returncode} in {time.time() - started:.0f}s",
          flush=True)
    if proc.returncode:
        raise RuntimeError(f"command failed with exit {proc.returncode}")


def prepare_processed():
    directory = absolute(PROCESSED)
    if os.path.isdir(directory):
        missing = [name for name in REQUIRED_PROCESSED
                   if not os.path.isfile(os.path.join(directory, name))]
        if missing:
            raise RuntimeError(
                "partial auxiliary corpus; missing " + ", ".join(missing))
        print(f"[lc0] reuse complete {PROCESSED}", flush=True)
    else:
        run(PY + [
            "src/data_processor.py", "--raw-dir", RAW,
            "--output-dir", PROCESSED,
        ] + PROCESS_RECIPE)
    validate_processed()
    prepare_exact_auxiliary()


def validate_processed():
    """Prove that auxiliary processing preserved the successful B formula."""
    import numpy as np

    reference = absolute(REFERENCE_PROCESSED)
    auxiliary = absolute(PROCESSED)
    identical = (
        "positions.npy", "policies.npy", "game_results.npy",
        "mcts_values.npy", "value_weights.npy", "splits.npz",
    )
    changed = [name for name in identical if not filecmp.cmp(
        os.path.join(reference, name), os.path.join(auxiliary, name),
        shallow=False)]
    if changed:
        raise RuntimeError(
            "auxiliary corpus changed v19_B foundation arrays: "
            + ", ".join(changed))

    ref_policy = np.load(os.path.join(reference, "policy_weights.npy"),
                         mmap_mode="r")
    aux_policy = np.load(os.path.join(auxiliary, "policy_weights.npy"),
                         mmap_mode="r")
    delta = ref_policy != aux_policy
    if (int(delta.sum()) != 16
            or not np.all(ref_policy[delta] == 1.0)
            or not np.all(aux_policy[delta] == 0.0)):
        raise RuntimeError(
            "unexpected policy-weight delta from v19_B; expected only 16 "
            "mirrored rows from 8 illegal source targets")
    value_weights = np.load(os.path.join(auxiliary, "value_weights.npy"),
                            mmap_mode="r")
    if not np.all(value_weights == 1.0):
        raise RuntimeError(
            "v19_B full-value formula was not preserved: masked value rows")
    print("[lc0] v19_B foundation verified: core arrays identical, all "
          f"{len(value_weights)} value rows enabled; 8 illegal source "
          "policy targets (16 mirrored rows) disabled", flush=True)


def prepare_exact_auxiliary():
    """Add auxiliary sidecars without changing any historical B train array."""
    directory = absolute(PROCESSED_EXACT)
    os.makedirs(directory, exist_ok=True)
    reference_names = (
        "positions.npy", "mcts_values.npy", "game_results.npy",
        "policies.npy", "policy_weights.npy", "value_weights.npy",
        "splits.npz",
    )
    auxiliary_names = (
        "moves_left.npy", "moves_left_weights.npy", "legal_masks_packed.npy",
        "split_game_ids.json",
    )
    for name in reference_names + auxiliary_names:
        source_root = REFERENCE_PROCESSED if name in reference_names else PROCESSED
        source = absolute(os.path.join(source_root, name))
        destination = os.path.join(directory, name)
        if not os.path.exists(destination):
            os.link(source, destination)
        if not filecmp.cmp(source, destination, shallow=False):
            raise RuntimeError(f"exact auxiliary corpus mismatch: {name}")
    print("[lc0] exact auxiliary corpus verified: historical v19_B training "
          "arrays plus aligned moves-left/legal-mask sidecars", flush=True)


def train_arm(name, skip_trained=False):
    if name == "control":
        print("[lc0] control: reuse historical v19_B checkpoint", flush=True)
        return absolute(CONTROL)
    model_dir = f"models/candidates/{MODEL_PREFIX}_{name}"
    checkpoint = absolute(os.path.join(model_dir, "best_value_net.pt"))
    if os.path.isfile(checkpoint):
        if skip_trained:
            print(f"[lc0] {name}: reuse existing checkpoint", flush=True)
            return checkpoint
        raise RuntimeError(
            f"{model_dir} already trained; pass --skip-trained to reuse")
    if os.path.exists(absolute(model_dir)):
        raise RuntimeError(f"partial model directory exists: {model_dir}")
    run(PY + [
        "src/train.py", "--data-dir", ARM_DATA[name], "--model-dir", model_dir,
    ] + TRAIN_RECIPE + ARMS[name])
    if not os.path.isfile(checkpoint):
        raise RuntimeError(f"training produced no checkpoint for {name}")
    return checkpoint


def screen_arm(name, checkpoint, games, sims, workers, seed):
    from match import run_match

    # Shared openings make color-normalized deltas against the control arm
    # meaningful. A later binding confirmation uses a fresh seed.
    arm_seed = seed
    match = run_match(
        checkpoint, absolute(CONTROL), games, sims, arm_seed,
        workers=workers, engine="native")
    black = match["a_as_black"]["score"]
    white = match["a_as_white"]["score"]
    aggregate = match["a_score"]
    return {
        "arm": name,
        "changed_flags": ARMS[name],
        "checkpoint": os.path.relpath(checkpoint, ROOT).replace("\\", "/"),
        "black_score": black,
        "white_score": white,
        "aggregate_score": aggregate,
        "screen_pass": (black >= PER_SIDE_FLOOR and white >= PER_SIDE_FLOOR
                        and aggregate > AGGREGATE_MIN),
        "match": match,
    }


def save(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", default="control,moves_left,legal_mask,ema,attention,se")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--workers", type=int, default=DEFAULT_GAME_WORKERS)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--skip-trained", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        parser.error("unknown arms: " + ", ".join(unknown))
    if args.games < 2 or args.sims <= 0 or args.workers <= 0:
        parser.error("games must be >=2; sims and workers must be >0")
    if not os.path.isfile(absolute(CONTROL)):
        raise FileNotFoundError(f"missing control checkpoint: {CONTROL}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = absolute(args.out) if args.out else absolute(
        f"benchmarks/lc0b_exact_training_sweep_{stamp}.json")
    payload = {
        "experiment": "lc0b_one_variable_training_sweep",
        "binding": False,
        "status": "running",
        "raw": RAW,
        "processed": PROCESSED,
        "processed_exact": PROCESSED_EXACT,
        "reference_processed": REFERENCE_PROCESSED,
        "control": CONTROL,
        "priority": "black_score",
        "safeguards": {"black_min": PER_SIDE_FLOOR,
                       "white_min": PER_SIDE_FLOOR,
                       "aggregate_exclusive": AGGREGATE_MIN},
        "training_recipe": TRAIN_RECIPE,
        "processing_recipe": PROCESS_RECIPE,
        "arm_data": ARM_DATA,
        "arms": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save(out, payload)
    prepare_processed()
    if args.prepare_only:
        payload["status"] = "prepared"
        save(out, payload)
        return

    for name in arms:
        print(f"\n{'=' * 72}\n[lc0] ARM {name}\n{'=' * 72}", flush=True)
        checkpoint = train_arm(name, skip_trained=args.skip_trained)
        if args.train_only:
            result = {
                "arm": name, "changed_flags": ARMS[name],
                "checkpoint": os.path.relpath(checkpoint, ROOT).replace("\\", "/"),
            }
        else:
            result = screen_arm(
                name, checkpoint, args.games, args.sims, args.workers,
                args.seed)
            print(f"[lc0] {name}: Black={result['black_score']:.3f} "
                  f"White={result['white_score']:.3f} "
                  f"all={result['aggregate_score']:.3f}", flush=True)
        payload["arms"].append(result)
        save(out, payload)

    if not args.train_only:
        control_result = next(
            (item for item in payload["arms"] if item["arm"] == "control"),
            None,
        )
        if control_result is not None:
            payload["self_calibration"] = {
                "black_score": control_result["black_score"],
                "white_score": control_result["white_score"],
                "aggregate_score": control_result["aggregate_score"],
                "seed": args.seed,
            }
            for item in payload["arms"]:
                item["delta_vs_self_calibration"] = {
                    "black": item["black_score"] - control_result["black_score"],
                    "white": item["white_score"] - control_result["white_score"],
                    "aggregate": (item["aggregate_score"]
                                  - control_result["aggregate_score"]),
                }
        payload["ranking"] = [
            arm["arm"] for arm in sorted(
                payload["arms"],
                key=lambda item: (
                    item.get("delta_vs_self_calibration", {}).get(
                        "black", item["black_score"]),
                    item["black_score"], item["aggregate_score"],
                    item["white_score"],
                ),
                reverse=True,
            )
        ]
    payload["status"] = "completed"
    payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(out, payload)
    print(f"[lc0] saved {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
