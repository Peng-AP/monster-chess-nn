"""Run one legacy accumulated, from-scratch bootstrap generation.

New unattended generations should use ``src/iterate.py``. This driver remains
for exact Gen7--Gen10 campaign reproduction; its outputs are now audited so a
reproduction cannot repeat the historical one-row teacher loss.

The loop's original fine-tuning step and offline-only checkpoint choice both
discarded measured playing strength.  This driver keeps the recipe that fixed
those apparatus faults: generate from the supplied incumbent, accumulate the
new corpus, train fresh at the successful v20 learning rate, select preserved
epochs by calibrated play, and gate the nominee against that same incumbent.

``--extra-source`` explicitly carries successful or informative prior
generations forward.  Nothing infers ancestry from directory names: the log
therefore records the exact data list and incumbent used by each generation.
An optional paired book gives the screen and gate disjoint position blocks.

Stages run in order and the driver **stops on the first failure** -- an
unattended chain that carries on past a broken stage produces a confident
result from nothing. Completed stages are skipped on a rerun.
"""
import argparse
import os
import subprocess
import sys
import time

from stage_artifacts import generation_complete, reanalysis_complete

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

_ap = argparse.ArgumentParser()
_ap.add_argument("--generation", type=int, required=True)
_ap.add_argument("--incumbent", required=True,
                 help="model that generates, and that the gate bars against")
_ap.add_argument("--games", type=int, default=500)
_ap.add_argument("--extra-source", action="append", default=[],
                 help="NAME=PATH accumulated corpus source (repeatable)")
_ap.add_argument("--book", default=None,
                 help="paired opening book used for checkpoint selection and "
                      "the gate; their entry blocks are disjoint")
_ap.add_argument("--book-offset", type=int, default=0,
                 help="first book entry reserved for this generation")
_args = _ap.parse_args()
if _args.book_offset < 0:
    _ap.error("--book-offset must be non-negative")

GEN = _args.generation
RAW = f"iterations/gen_{GEN:04d}/raw"
REANALYSIS = f"{RAW}/reanalysis"
PROCESSED = f"data/processed/bootstrap_new_main_gen_{GEN:04d}"
CORPUS = f"data/processed/bootstrap_replay_main_gen_{GEN:04d}"
MODEL_DIR = f"models/candidates/gen{GEN}_scratch"
INCUMBENT = _args.incumbent

# v20's from-scratch recipe: the one that produced a model beating five
# generations of fine-tuning.
RECIPE = [
    "--epochs", "30", "--patience", "10", "--batch-size", "256",
    "--memory-map-data", "--lr", "0.002", "--lr-gamma", "0.95",
    "--policy-loss-weight", "1.0", "--black-policy-weight", "1.0",
    "--weight-decay", "0.0001", "--ema-decay", "0.999", "--grad-clip", "1.0",
    "--warmup-epochs", "3", "--warmup-start-factor", "0.1", "--seed", "42",
    "--target", "game_result", "--value-head", "scalar",
    "--select-metric", "decisive", "--save-selection-snapshots",
    "--stem-channels", "64", "--policy-head", "attention",
    "--policy-attention-channels", "64",
]

SOURCES = [
    "anchor=data/processed/combined_v19_B_r50h60_capture",
    "gen_0002=data/processed/bootstrap_new_main_gen_0002",
    "gen_0003=data/processed/bootstrap_new_main_gen_0003",
    "gen_0004=data/processed/bootstrap_new_main_gen_0004",
    "gen_0005_teacher3200=data/processed/bootstrap_new_main_gen_0005_teacher3200",
] + _args.extra_source + [f"gen_{GEN:04d}={PROCESSED}"]


def done(marker):
    return os.path.exists(os.path.join(ROOT, marker))


def stage(name, cmd, marker=None, validator=None):
    complete = validator() if validator else (marker and done(marker))
    if complete:
        description = marker or "validated artifact"
        print(f"\n=== [{name}] already complete ({description}) -- skipping ===",
              flush=True)
        return
    print(f"\n=== [{name}] {time.strftime('%H:%M:%S')} ===\n$ "
          f"{' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([PY, "-u"] + [str(c) for c in cmd], cwd=ROOT).returncode
    mins = (time.time() - t0) / 60
    print(f"=== [{name}] exit={rc} after {mins:.1f}m ===", flush=True)
    if rc != 0:
        raise SystemExit(f"[{name}] failed with exit {rc}; stopping the chain")


def main():
    started = time.time()
    print(f"generation {GEN}: from-scratch training, play-based selection, "
          f"gate against {INCUMBENT}", flush=True)

    stage("generate", [
        "src/data_generation.py", "--engine", "native",
        "--num-games", str(_args.games), "--simulations", "700", "--workers", "8",
        "--stall-timeout", "900", "--use-model", INCUMBENT,
        "--record-all-plies", "--seed", str(70000000 + GEN * 1009),
        "--output-dir", RAW,
    ], marker=f"{RAW}/generation_summary.json",
       validator=lambda: generation_complete(
           os.path.join(ROOT, RAW), _args.games))

    stage("reanalyze", [
        "tools/reanalyze.py", "--source-dir", RAW, "--model", INCUMBENT,
        "--output-dir", REANALYSIS, "--sample", "8000", "--keep", "4000",
        "--black-fraction", "0.6", "--simulations", "3200",
        "--engine", "native", "--workers", "8", "--seed", str(70000000 + GEN * 1009 + 1),
        "--stall-timeout", "900",
    ], marker=f"{REANALYSIS}/reanalysis_summary.json",
       validator=lambda: reanalysis_complete(
           os.path.join(ROOT, REANALYSIS), 8000, 4000, 3200))

    stage("process", [
        "src/data_processor.py", "--raw-dir", RAW,
        "--output-dir", PROCESSED, "--value-floor", "0.5",
        "--value-horizon", "60", "--value-discount-mode", "near_mate",
        "--min-nonhuman-plies", "0", "--max-generation-age", "0",
        "--channels", "15",
    ], marker=f"{PROCESSED}/positions.npy")

    stage("audit", [
        "tools/audit_generation_data.py", "--raw-dir", RAW,
        "--reanalysis-dir", REANALYSIS, "--processed-dir", PROCESSED,
        "--expected-teachers", "4000", "--expected-black-fraction", "0.6",
        "--value-floor", "0.5", "--value-horizon", "60",
    ], marker=f"{PROCESSED}/generation_audit.json")

    compose = ["tools/compose_processed.py"]
    for s in SOURCES:
        compose += ["--source", s]
    compose += ["--output-dir", CORPUS, "--balance-alpha", "0.5",
                "--balance-seed", "807096"]
    stage("compose", compose, marker=f"{CORPUS}/positions.npy")

    stage("train", [
        "src/train.py", "--data-dir", CORPUS, "--model-dir", MODEL_DIR,
    ] + RECIPE, marker=f"{MODEL_DIR}/best_value_net.pt")

    screen = [
        "tools/checkpoint_screen.py", "--model-dir", MODEL_DIR,
        "--incumbent", INCUMBENT,
        "--output-model", f"{MODEL_DIR}/screen_nominee.pt",
        "--report-path", f"benchmarks/screen_gen{GEN}.json",
    ]
    if _args.book:
        screen += ["--book", _args.book, "--book-offset",
                   str(_args.book_offset)]
    stage("screen", screen, marker=f"benchmarks/screen_gen{GEN}.json")

    nominee = f"{MODEL_DIR}/screen_nominee.pt"
    if not done(nominee):
        nominee = f"{MODEL_DIR}/best_value_net.pt"
        print(f"\n(no screen nominee; gating {nominee})", flush=True)
    gate = [
        "tools/gate.py", "--model", nominee, "--protocol", "full",
        "--engine", "native", "--workers", "8",
        "--bar-model", INCUMBENT,
        "--report-path", f"benchmarks/gate_gen{GEN}.json",
    ]
    # The default screen consumes 20 probe entries plus 100 final entries.
    # Start the binding gate after them so selection cannot train on its test.
    if _args.book:
        gate += ["--book", _args.book, "--book-offset",
                 str(_args.book_offset + 120)]
    stage("gate", gate)

    print(f"\ngeneration {GEN} complete in "
          f"{(time.time() - started) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
