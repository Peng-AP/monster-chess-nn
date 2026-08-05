"""Black-first multi-fidelity hyperparameter tuning on the exact v19_B data.

The objective is calibrated arena play, not validation loss. Each fidelity
rung trains from the same seed and from scratch to its epoch budget; this costs
some repeated early epochs, but avoids optimizer-reset and resume-state bias.
Weak trials are pruned by successive halving after the 6- and 12-epoch rungs.

Examples:
    py -3 -u tools/tune_training.py --smoke --trials 1
    py -3 -u tools/tune_training.py --trials 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
import time

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from config import DEFAULT_GAME_WORKERS  # noqa: E402
from match import run_match  # noqa: E402

DATA = "data/processed/combined_v19_B_r50h60"
BAR = "models/candidates/v19_B/best_value_net.pt"
DEFAULT_STUDY = "v19b_training_hpo"
DEFAULT_MODEL_ROOT = "models/tuning/v19b_training_hpo"
DEFAULT_LOG_ROOT = "logs/hpo/v19b_training_hpo"
DEFAULT_STORAGE = "logs/hpo/v19b_training_hpo.sqlite3"
WHITE_DELTA_FLOOR = -0.05
WHITE_PENALTY = 2.0
AGGREGATE_WEIGHT = 0.25
BASELINE_PARAMS = {
    "lr": 0.002,
    "lr_gamma": 0.95,
    "weight_decay": 0.0001,
    "policy_loss_weight": 1.0,
    "warmup_epochs": 3,
    "grad_clip": 1.0,
    "batch_size": 256,
}


@dataclass(frozen=True)
class Stage:
    index: int
    epochs: int
    games: int
    sims: int
    seed: int


STAGES = (
    Stage(1, 6, 16, 200, 20260820),
    Stage(2, 12, 40, 400, 20360820),
    Stage(3, 30, 80, 800, 20460820),
)
SMOKE_STAGES = (Stage(1, 1, 2, 1, 90260820),)


def absolute(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def suggest_parameters(trial) -> dict:
    """Small, training-only search space; architecture and data stay frozen."""
    return {
        "lr": trial.suggest_float("lr", 3e-4, 4e-3, log=True),
        "lr_gamma": trial.suggest_float("lr_gamma", 0.90, 0.99),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 3e-3, log=True),
        "policy_loss_weight": trial.suggest_float(
            "policy_loss_weight", 0.35, 2.5, log=True),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 1, 5),
        "grad_clip": trial.suggest_categorical(
            "grad_clip", [0.0, 0.5, 1.0, 2.0]),
        "batch_size": trial.suggest_categorical(
            "batch_size", [128, 256, 512]),
    }


def arena_objective(candidate: dict, calibration: dict) -> dict:
    black_delta = (candidate["a_as_black"]["score"]
                   - calibration["a_as_black"]["score"])
    white_delta = (candidate["a_as_white"]["score"]
                   - calibration["a_as_white"]["score"])
    aggregate_delta = candidate["a_score"] - calibration["a_score"]
    white_shortfall = max(0.0, WHITE_DELTA_FLOOR - white_delta)
    score = (black_delta + AGGREGATE_WEIGHT * aggregate_delta
             - WHITE_PENALTY * white_shortfall)
    return {
        "score": float(score),
        "black_delta": float(black_delta),
        "white_delta": float(white_delta),
        "aggregate_delta": float(aggregate_delta),
        "white_shortfall": float(white_shortfall),
    }


def training_command(params: dict, stage: Stage, model_dir: Path) -> list[str]:
    patience = 10 if stage.epochs >= 30 else stage.epochs + 1
    return [
        sys.executable, "-u", "src/train.py",
        "--data-dir", DATA,
        "--model-dir", str(model_dir.relative_to(ROOT)),
        "--epochs", str(stage.epochs),
        "--patience", str(patience),
        "--batch-size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
        "--lr-gamma", str(params["lr_gamma"]),
        "--policy-loss-weight", str(params["policy_loss_weight"]),
        "--weight-decay", str(params["weight_decay"]),
        "--grad-clip", str(params["grad_clip"]),
        "--warmup-epochs", str(params["warmup_epochs"]),
        "--warmup-start-factor", "0.1",
        "--seed", "42",
        "--target", "game_result",
        "--value-head", "scalar",
        "--select-metric", "decisive",
        "--stem-channels", "64",
    ]


class TrainingTuner:
    def __init__(self, args, stages):
        self.args = args
        self.stages = stages
        self.bar = absolute(BAR)
        self.model_root = absolute(args.model_root)
        self.log_root = absolute(args.log_root)
        self.calibration_root = self.log_root / "calibration"
        self.bar_hash = sha256(self.bar)

    def calibration(self, stage: Stage) -> dict:
        path = self.calibration_root / f"stage_{stage.index:02d}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "bar_sha256": self.bar_hash,
                "games": stage.games,
                "sims": stage.sims,
                "seed": stage.seed,
            }
            if all(payload.get(key) == value for key, value in expected.items()):
                print(f"[hpo] reuse stage-{stage.index} self-calibration", flush=True)
                return payload["match"]
            raise RuntimeError(f"stale calibration metadata: {path}")

        print(f"[hpo] stage-{stage.index} self-calibration: "
              f"{stage.games} games @ {stage.sims} sims", flush=True)
        match = run_match(
            str(self.bar), str(self.bar), stage.games, stage.sims, stage.seed,
            workers=self.args.workers, engine="native")
        save_json(path, {
            "bar": BAR,
            "bar_sha256": self.bar_hash,
            "games": stage.games,
            "sims": stage.sims,
            "seed": stage.seed,
            "match": match,
        })
        return match

    def train(self, trial, params: dict, stage: Stage) -> tuple[Path, dict]:
        trial_dir = self.model_root / f"trial_{trial.number:05d}"
        stage_dir = trial_dir / f"stage_{stage.index:02d}_e{stage.epochs:02d}"
        config_path = trial_dir / "config.json"
        config = {"trial": trial.number, "params": params, "data": DATA,
                  "seed": 42}
        if config_path.exists():
            recorded = json.loads(config_path.read_text(encoding="utf-8"))
            if recorded != config:
                raise RuntimeError(f"trial config mismatch: {config_path}")
        else:
            save_json(config_path, config)

        checkpoint = stage_dir / "best_value_net.pt"
        metadata_files = sorted(stage_dir.glob("train_run_*.json"))
        if checkpoint.is_file() and metadata_files:
            print(f"[hpo] trial {trial.number} stage {stage.index}: "
                  "reuse complete training", flush=True)
            return checkpoint, json.loads(
                metadata_files[-1].read_text(encoding="utf-8"))
        if stage_dir.exists():
            raise RuntimeError(f"partial training directory: {stage_dir}")

        stage_dir.mkdir(parents=True)
        log_path = self.log_root / f"trial_{trial.number:05d}_stage_{stage.index:02d}.log"
        command = training_command(params, stage, stage_dir)
        print(f"[hpo] trial {trial.number} stage {stage.index}: "
              f"train {stage.epochs} epochs", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(
                f"training failed ({proc.returncode}); see {log_path}")
        metadata_files = sorted(stage_dir.glob("train_run_*.json"))
        if not checkpoint.is_file() or len(metadata_files) != 1:
            raise RuntimeError(f"incomplete training output: {stage_dir}")
        return checkpoint, json.loads(
            metadata_files[0].read_text(encoding="utf-8"))

    def evaluate(self, trial, checkpoint: Path, stage: Stage,
                 calibration: dict) -> dict:
        trial_dir = self.model_root / f"trial_{trial.number:05d}"
        path = trial_dir / f"stage_{stage.index:02d}_match.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            print(f"[hpo] trial {trial.number} stage {stage.index}: "
                  "reuse match", flush=True)
            return payload
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash == self.bar_hash:
            print(f"[hpo] trial {trial.number} stage {stage.index}: exact "
                  "v19_B checkpoint; reuse self-calibration", flush=True)
            match = calibration
        else:
            print(f"[hpo] trial {trial.number} stage {stage.index}: "
                  f"screen {stage.games} games @ {stage.sims} sims", flush=True)
            match = run_match(
                str(checkpoint), str(self.bar), stage.games, stage.sims,
                stage.seed, workers=self.args.workers, engine="native")
        objective = arena_objective(match, calibration)
        payload = {
            "trial": trial.number,
            "stage": asdict(stage),
            "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": checkpoint_hash,
            "calibration": calibration,
            "match": match,
            "objective": objective,
        }
        save_json(path, payload)
        return payload

    def objective(self, trial):
        params = suggest_parameters(trial)
        final_score = None
        for stage in self.stages:
            calibration = self.calibration(stage)
            checkpoint, metadata = self.train(trial, params, stage)
            result = self.evaluate(trial, checkpoint, stage, calibration)
            metrics = result["objective"]
            final_score = metrics["score"]
            trial.set_user_attr(f"stage_{stage.index}_artifact", str(
                (self.model_root / f"trial_{trial.number:05d}"
                 / f"stage_{stage.index:02d}_match.json").relative_to(ROOT)
            ).replace("\\", "/"))
            trial.set_user_attr(f"stage_{stage.index}_best_epoch",
                                metadata["best_epoch"])
            for name, value in metrics.items():
                trial.set_user_attr(f"stage_{stage.index}_{name}", value)
            trial.report(final_score, step=stage.index)
            print(f"[hpo] trial {trial.number} stage {stage.index}: "
                  f"objective={final_score:+.4f} "
                  f"dB={metrics['black_delta']:+.3f} "
                  f"dW={metrics['white_delta']:+.3f} "
                  f"dAll={metrics['aggregate_delta']:+.3f}", flush=True)
            if stage is not self.stages[-1] and trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned after fidelity stage {stage.index}")
        return final_score


def storage_url(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.resolve().as_posix()


def save_summary(study, args, stages) -> Path:
    completed = [trial for trial in study.trials
                 if trial.state == optuna.trial.TrialState.COMPLETE]
    payload = {
        "study": study.study_name,
        "storage": str(absolute(args.storage).relative_to(ROOT)),
        "data": DATA,
        "bar": BAR,
        "objective": {
            "formula": "black_delta + 0.25*aggregate_delta - 2*white_shortfall",
            "white_delta_floor": WHITE_DELTA_FLOOR,
        },
        "stages": [asdict(stage) for stage in stages],
        "trial_counts": {
            state.name: sum(trial.state == state for trial in study.trials)
            for state in optuna.trial.TrialState
        },
        "best": ({"number": study.best_trial.number,
                  "value": study.best_value,
                  "params": study.best_params,
                  "user_attrs": study.best_trial.user_attrs}
                 if completed else None),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    suffix = "smoke" if args.smoke else time.strftime("%Y%m%d_%H%M%S")
    path = ROOT / "benchmarks" / f"hpo_v19b_training_{suffix}.json"
    save_json(path, payload)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=20,
                        help="additional Optuna trials to run")
    parser.add_argument("--study-name", default=DEFAULT_STUDY)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--workers", type=int, default=DEFAULT_GAME_WORKERS)
    parser.add_argument("--timeout-hours", type=float, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="one-epoch, two-game wiring check")
    args = parser.parse_args()
    if args.trials <= 0 or args.workers <= 0:
        parser.error("trials and workers must be > 0")
    if not absolute(DATA).is_dir():
        raise FileNotFoundError(f"missing exact B corpus: {DATA}")
    if not absolute(BAR).is_file():
        raise FileNotFoundError(f"missing v19_B checkpoint: {BAR}")

    stages = SMOKE_STAGES if args.smoke else STAGES
    if args.smoke:
        args.study_name += "_smoke"
        args.storage = str(Path(args.storage).with_name(
            Path(args.storage).stem + "_smoke.sqlite3"))
        args.model_root += "_smoke"
        args.log_root += "_smoke"

    sampler = optuna.samplers.TPESampler(
        seed=20260804, n_startup_trials=8)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url(absolute(args.storage)),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(BASELINE_PARAMS, user_attrs={
            "role": "historical_v19_B_recipe_anchor",
        })
    tuner = TrainingTuner(args, stages)
    timeout = (args.timeout_hours * 3600
               if args.timeout_hours is not None else None)
    study.optimize(tuner.objective, n_trials=args.trials, timeout=timeout,
                   gc_after_trial=True)
    summary = save_summary(study, args, stages)
    print(f"[hpo] saved {summary.relative_to(ROOT)}", flush=True)
    if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        print(f"[hpo] best trial={study.best_trial.number} "
              f"objective={study.best_value:+.4f}", flush=True)


if __name__ == "__main__":
    main()
