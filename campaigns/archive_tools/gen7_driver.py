"""Generation 7, run with every apparatus lesson from 2026-08-07 applied.

The loop's training step was the defect, not its data. Measured that day:
training a fresh network on the accumulated corpus beat the fine-tuned champion
by ~31 Elo (800 games), and inside a from-scratch run the offline checkpoint
metric discarded a further ~52 Elo by preferring epoch 4 to epoch 12. Both are
apparatus faults; neither is about the game.

So this generation:

  * generates from **e4**, the strongest model on record (gate-passed at the
    200-game bar leg, pooled 0.5713 over 400 games, z=+2.85)
  * **accumulates** onto the gen-5 corpus, matching how gen 2-5 were built
  * trains **from scratch** at lr 2e-3 rather than resuming at 1e-4
  * selects the checkpoint by **play** at 200-game finals, not validation fit
  * gates the nominee against **e4** rather than v21, because the bar is the
    strongest engine on record

The question it answers is whether the loop *compounds*: one fresh-training
generation already beat v21, and if generation 7 then beats e4 by a similar
margin the loop works and only ever needed a competent training step.

Stages run in order and the driver **stops on the first failure** -- an
unattended chain that carries on past a broken stage produces a confident
result from nothing. Completed stages are skipped on a rerun.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

GEN = 7
RAW = f"iterations/gen_{GEN:04d}/raw"
REANALYSIS = f"{RAW}/reanalysis"
PROCESSED = f"data/processed/bootstrap_new_main_gen_{GEN:04d}"
CORPUS = f"data/processed/bootstrap_replay_main_gen_{GEN:04d}"
MODEL_DIR = f"models/candidates/gen{GEN}_scratch"
INCUMBENT = "models/candidates/scratch_v21corpus/best_value_net.pt"   # e4

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
    f"gen_{GEN:04d}={PROCESSED}",
]


def done(marker):
    return os.path.exists(os.path.join(ROOT, marker))


def stage(name, cmd, marker=None):
    if marker and done(marker):
        print(f"\n=== [{name}] already complete ({marker}) -- skipping ===",
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
          f"gate against e4", flush=True)

    stage("generate", [
        "src/data_generation.py", "--engine", "native",
        "--num-games", "500", "--simulations", "700", "--workers", "8",
        "--stall-timeout", "900", "--use-model", INCUMBENT,
        "--record-all-plies", "--seed", "70007000",
        "--output-dir", RAW,
    ], marker=RAW)

    stage("reanalyze", [
        "tools/reanalyze.py", "--source-dir", RAW, "--model", INCUMBENT,
        "--output-dir", REANALYSIS, "--sample", "8000", "--keep", "4000",
        "--black-fraction", "0.6", "--simulations", "3200",
        "--engine", "native", "--workers", "8", "--seed", "70007001",
        "--stall-timeout", "900",
    ], marker=REANALYSIS)

    stage("process", [
        "src/data_processor.py", "--raw-dir", RAW,
        "--output-dir", PROCESSED, "--value-floor", "0.5",
        "--value-horizon", "60", "--value-discount-mode", "near_mate",
        "--channels", "15",
    ], marker=f"{PROCESSED}/positions.npy")

    compose = ["tools/compose_processed.py"]
    for s in SOURCES:
        compose += ["--source", s]
    compose += ["--output-dir", CORPUS, "--balance-alpha", "0.5",
                "--balance-seed", "807096"]
    stage("compose", compose, marker=f"{CORPUS}/positions.npy")

    stage("train", [
        "src/train.py", "--data-dir", CORPUS, "--model-dir", MODEL_DIR,
    ] + RECIPE, marker=f"{MODEL_DIR}/best_value_net.pt")

    stage("screen", [
        "tools/checkpoint_screen.py", "--model-dir", MODEL_DIR,
        "--incumbent", INCUMBENT,
        "--output-model", f"{MODEL_DIR}/screen_nominee.pt",
        "--report-path", f"benchmarks/screen_gen{GEN}.json",
    ], marker=f"benchmarks/screen_gen{GEN}.json")

    nominee = f"{MODEL_DIR}/screen_nominee.pt"
    if not done(nominee):
        nominee = f"{MODEL_DIR}/best_value_net.pt"
        print(f"\n(no screen nominee; gating {nominee})", flush=True)
    stage("gate", [
        "tools/gate.py", "--model", nominee, "--protocol", "full",
        "--engine", "native", "--workers", "8",
        "--bar-model", INCUMBENT,
        "--report-path", f"benchmarks/gate_gen{GEN}.json",
    ])

    print(f"\ngeneration {GEN} complete in "
          f"{(time.time() - started) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
