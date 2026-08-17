"""Resumable self-bootstrap pipeline.

One generation is a reproducible state machine:

    generate -> reanalyze -> process -> compose -> train -> checkpoint_screen
             -> offline_gate -> binding_gate -> high_fidelity_gate
             -> self_skew -> promote

The current champion is never overwritten.  A passing candidate is archived
under ``models/bootstrap/champions`` and selected through ``champion.json``.
Numbered release checkpoints remain immutable.

Use ``--dry-run`` first.  Long executions should be launched through
``tools/runs.py`` so their combined output is also visible in ``logs/``.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import re
from pathlib import Path

import numpy as np

from config import (
    PROJECT_ROOT,
    RANDOM_SEED,
    ITERATE_ARENA_SIMS,
    DEFAULT_GAME_WORKERS,
)

ROOT = Path(PROJECT_ROOT)
PY = sys.executable
DEFAULT_RUN_ROOT = ROOT / "iterations"
DEFAULT_CHAMPION = ROOT / "models" / "bootstrap_v23" / "best_value_net.pt"
DEFAULT_ANCHOR_DATA = ROOT / "data" / "processed" / "combined_v19_B_r50h60_capture"
DEFAULT_SPARRING = (
    ROOT / "models" / "rejected" / "fresh_start_v18_ramp" / "best_value_net.pt")
SCRATCH_EPOCHS = 30
SCRATCH_PATIENCE = 10
SCRATCH_LR = 0.002
SCRATCH_WARMUP_EPOCHS = 3
BOOTSTRAP_GAMES = 500
BOOTSTRAP_SIMS = 700
BOOTSTRAP_REANALYSIS_SAMPLE = 8000
BOOTSTRAP_REANALYSIS_KEEP = 4000
BOOTSTRAP_REANALYSIS_SIMS = 3200
BOOTSTRAP_VALUE_FLOOR = 0.5
BOOTSTRAP_VALUE_HORIZON = 60
TEACHER_POLICY_MULTIPLIER = 4.0
BOOTSTRAP_MODELS = ROOT / "models" / "bootstrap"
CHAMPIONS_DIR = BOOTSTRAP_MODELS / "champions"
CHAMPION_POINTER = BOOTSTRAP_MODELS / "champion.json"

PHASES = (
    "generate",
    "reanalyze",
    "process",
    "compose",
    "train",
    "checkpoint_screen",
    "offline_gate",
    "binding_gate",
    "high_fidelity_gate",
    "self_skew",
    "promote",
)


def _rel(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _absolute(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def _load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _pid_alive(pid):
    if not pid:
        return False
    try:
        if os.name == "nt":
            output = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10).stdout
            return str(pid) in output
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _next_generation(run_root):
    best = 0
    if not Path(run_root).exists():
        return 1
    for path in Path(run_root).glob("gen_*"):
        suffix = path.name.removeprefix("gen_")
        if suffix.isdigit():
            best = max(best, int(suffix))
    return best + 1


def _latest_generation(run_root):
    next_generation = _next_generation(run_root)
    return next_generation - 1 if next_generation > 1 else None


def _resolve_champion(explicit=None):
    if explicit:
        return _absolute(explicit)
    pointer = _load_json(CHAMPION_POINTER)
    if pointer and pointer.get("checkpoint"):
        checkpoint = _absolute(pointer["checkpoint"])
        if checkpoint.exists():
            return checkpoint
    return DEFAULT_CHAMPION.resolve()


def _checkpoint_spec(checkpoint):
    """Infer every flag needed to rebuild the champion architecture fresh."""
    import torch
    from train import (
        infer_backbone_architecture,
        infer_input_channels,
        infer_moves_left_head_config,
        infer_policy_head_config,
        infer_promotion_policy,
        infer_se_config,
        infer_side_policy_adapters,
        infer_spatial_value_head_config,
        infer_wdl_head_config,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy_type, _policy_channels, attention_channels = infer_policy_head_config(state)
    stem_channels, residual_channels = infer_backbone_architecture(state)
    use_se, se_reduction = infer_se_config(state)
    use_wdl, value_mode = infer_wdl_head_config(state)
    spatial, _spatial_channels = infer_spatial_value_head_config(state)
    moves_left, moves_left_channels = infer_moves_left_head_config(state)
    return {
        "input_channels": infer_input_channels(state),
        "policy_head": policy_type,
        "policy_attention_channels": attention_channels,
        "side_policy_adapters": infer_side_policy_adapters(state),
        "promotion_policy": infer_promotion_policy(state),
        "stem_channels": stem_channels,
        "residual_channels": list(residual_channels),
        "use_se_blocks": use_se,
        "se_reduction": se_reduction,
        "use_wdl_head": use_wdl,
        "value_head": value_mode,
        "spatial_value_head": spatial,
        "moves_left_head": moves_left,
        "moves_left_head_channels": moves_left_channels,
    }


def _accepted_registry_path(run_root):
    return Path(run_root) / "accepted_data.json"


def _accept_generation_data(state, paths, run_root):
    """Register immutable champion-generated data before candidate training."""
    processed = Path(paths["new_processed"])
    required = [
        "positions.npy", "mcts_values.npy", "game_results.npy",
        "policy_weights.npy", "value_weights.npy", "splits.npz",
        "split_game_ids.json", "generation_audit.json",
    ]
    if (processed / "policies_sparse.npz").exists():
        required.append("policies_sparse.npz")
    else:
        required.append("policies.npy")
    missing = [name for name in required if not (processed / name).exists()]
    if missing:
        raise RuntimeError(f"cannot accept incomplete processed data: {missing}")
    generation_audit = _load_json(processed / "generation_audit.json")
    if (not isinstance(generation_audit, dict)
            or generation_audit.get("verdict") != "PASS"):
        raise RuntimeError(
            "cannot accept processed data without a passing generation audit")
    positions = np.load(processed / "positions.npy", mmap_mode="r")
    with np.load(processed / "splits.npz") as split_file:
        split_rows = {name: int(len(split_file[name]))
                      for name in ("train", "val", "test")}
    acceptance = {
        "generation": int(state["generation"]),
        "accepted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "path": _rel(processed),
        "rows": int(len(positions)),
        "split_rows": split_rows,
        "splits_sha256": _sha256(processed / "splits.npz"),
        "artifact_sha256": {
            name: _sha256(processed / name) for name in required
        },
        "generation_audit": generation_audit,
        "incumbent": state["incumbent"],
        "incumbent_sha256": state["incumbent_sha256"],
        "run_state": state["paths"]["state"],
    }
    registry_path = _accepted_registry_path(run_root)
    registry = _load_json(registry_path, {"schema_version": 1, "entries": []})
    entries = [row for row in registry.get("entries", [])
               if int(row.get("generation", -1)) != state["generation"]]
    entries.append(acceptance)
    entries.sort(key=lambda row: int(row["generation"]))
    registry.update({"schema_version": 1, "entries": entries})
    _atomic_json(registry_path, registry)
    state["data_acceptance"] = acceptance
    return acceptance


def _recent_replay_sources(run_root, before_generation, count):
    registry = _load_json(_accepted_registry_path(run_root), {"entries": []})
    rows = []
    for entry in registry.get("entries", []):
        generation = int(entry.get("generation", 0))
        processed = entry.get("path")
        if generation >= before_generation or not processed:
            continue
        resolved = _absolute(processed)
        if not resolved.exists():
            raise FileNotFoundError(
                f"accepted replay generation {generation} is missing: {resolved}")
        expected_sha = entry.get("splits_sha256")
        if expected_sha and _sha256(resolved / "splits.npz") != expected_sha:
            raise RuntimeError(
                f"accepted replay generation {generation} was modified: {resolved}")
        for name, digest in entry.get("artifact_sha256", {}).items():
            artifact = resolved / name
            if not artifact.exists() or _sha256(artifact) != digest:
                raise RuntimeError(
                    f"accepted replay generation {generation} artifact "
                    f"was modified: {artifact}")
        rows.append((generation, resolved))
    rows.sort(key=lambda row: row[0])
    return rows[-max(0, count):] if count else []


def _run_namespace(run_root):
    resolved = Path(run_root).resolve()
    if resolved == DEFAULT_RUN_ROOT.resolve():
        return "main"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", resolved.name).strip("_")
    return cleaned or "run"


def _paths_for_generation(run_root, generation):
    namespace = _run_namespace(run_root)
    run_dir = Path(run_root) / f"gen_{generation:04d}"
    return {
        "run_dir": run_dir,
        "state": run_dir / "state.json",
        "lock": run_dir / "run.lock",
        "raw": run_dir / "raw",
        "selfplay": run_dir / "raw" / "selfplay",
        "league_black": run_dir / "raw" / "league_black",
        "league_white": run_dir / "raw" / "league_white",
        "reanalysis": run_dir / "raw" / "reanalysis",
        "logs": run_dir / "logs",
        "reports": run_dir / "reports",
        "new_processed": ROOT / "data" / "processed" /
                         f"bootstrap_new_{namespace}_gen_{generation:04d}",
        "replay_processed": ROOT / "data" / "processed" /
                            f"bootstrap_replay_{namespace}_gen_{generation:04d}",
        "candidate_dir": ROOT / "models" / "candidates" /
                         f"bootstrap_{namespace}_gen_{generation:04d}",
        "training_candidate": ROOT / "models" / "candidates" /
                              f"bootstrap_{namespace}_gen_{generation:04d}" /
                              "best_value_net.pt",
        "candidate": ROOT / "models" / "candidates" /
                     f"bootstrap_{namespace}_gen_{generation:04d}" /
                     "arena_selected.pt",
        "training_rejection": ROOT / "models" / "candidates" /
                              f"bootstrap_{namespace}_gen_{generation:04d}" /
                              "selection_rejected.json",
    }


def _binding_book_span(protocol):
    """Return entries consumed by gate.py, using its live protocol constants."""
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from gate import FULL_LEGS, QUICK_LEGS, book_leg_offsets
    spec = FULL_LEGS if protocol == "full" else QUICK_LEGS
    _offsets, needed = book_leg_offsets(spec)
    return int(needed)


def _command_plan(args, generation, incumbent, architecture, paths,
                  replay_sources):
    seed = args.seed + generation * 1009
    generated_commands = [[
        "src/data_generation.py",
        "--engine", args.engine,
        "--num-games", str(args.games),
        "--simulations", str(args.sims),
        "--workers", str(args.workers),
        "--stall-timeout", str(args.worker_stall_timeout),
        "--use-model", str(incumbent),
        "--record-all-plies",
        "--seed", str(seed),
        "--output-dir", str(paths["selfplay"]),
    ]]
    if args.league_games > 0:
        black_games = (args.league_games + 1) // 2
        white_games = args.league_games // 2
        for side, games, output, side_seed in (
                ("black", black_games, paths["league_black"], seed + 100_000),
                ("white", white_games, paths["league_white"], seed + 200_000)):
            if games <= 0:
                continue
            generated_commands.append([
                "src/data_generation.py",
                "--engine", args.engine,
                "--num-games", str(games),
                "--simulations", str(args.sims),
                "--opponent-sims", str(args.sims),
                "--workers", str(args.workers),
                "--stall-timeout", str(args.worker_stall_timeout),
                "--use-model", str(incumbent),
                "--opponent-pool-dir", str(CHAMPIONS_DIR),
                "--opponent-pool-size", str(args.opponent_pool_size),
                "--train-side", side,
                "--record-all-plies",
                "--seed", str(side_seed),
                "--output-dir", str(output),
            ])
    generated_outputs = [str(paths["selfplay"] / "generation_summary.json")]
    if args.league_games > 0:
        if (args.league_games + 1) // 2:
            generated_outputs.append(
                str(paths["league_black"] / "generation_summary.json"))
        if args.league_games // 2:
            generated_outputs.append(
                str(paths["league_white"] / "generation_summary.json"))

    reanalyze = [
        "tools/reanalyze.py",
        "--source-dir", str(paths["raw"]),
        "--model", str(incumbent),
        "--output-dir", str(paths["reanalysis"]),
        "--sample", str(args.reanalysis_sample),
        "--keep", str(args.reanalysis_keep),
        "--black-fraction", str(args.reanalysis_black_fraction),
        "--simulations", str(args.reanalysis_sims),
        "--engine", args.engine,
        "--workers", str(args.workers),
        "--seed", str(seed + 300_000),
        "--stall-timeout", str(args.worker_stall_timeout),
    ]
    process = [
        "src/data_processor.py",
        "--raw-dir", str(paths["raw"]),
        "--output-dir", str(paths["new_processed"]),
        "--min-nonhuman-plies", "0",
        "--max-generation-age", "0",
        "--value-floor", str(args.value_floor),
        "--value-horizon", str(args.value_horizon),
        "--value-discount-mode", "near_mate",
        "--channels", str(architecture["input_channels"]),
        "--seed", str(args.seed),
    ]
    if architecture["promotion_policy"]:
        process.append("--promotion-aware-policy")

    compose = ["tools/compose_processed.py"]
    compose += ["--source", f"anchor={_absolute(args.anchor_data)}"]
    for old_generation, source in replay_sources:
        name = f"gen_{old_generation:04d}"
        compose += ["--source", f"{name}={source}",
                    "--policy-only-multiplier",
                    f"{name}={args.teacher_policy_multiplier}"]
    compose += ["--source", f"gen_{generation:04d}={paths['new_processed']}",
                "--policy-only-multiplier",
                f"gen_{generation:04d}={args.teacher_policy_multiplier}",
                "--output-dir", str(paths["replay_processed"]),
                "--balance-alpha", str(args.replay_balance_alpha),
                "--balance-seed", str(seed + 700_000)]

    moves_left = bool(architecture["moves_left_head"] or args.moves_left_head)
    train = [
        "src/train.py",
        "--data-dir", str(paths["replay_processed"]),
        "--model-dir", str(paths["candidate_dir"]),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--memory-map-data",
        "--lr", str(args.lr),
        "--lr-gamma", str(args.lr_gamma),
        "--policy-loss-weight", "1.0",
        "--black-policy-weight", "1.0",
        "--weight-decay", "0.0001",
        "--grad-clip", "1.0",
        "--warmup-epochs", str(args.warmup_epochs),
        "--warmup-start-factor", "0.1",
        "--ema-decay", str(args.ema_decay),
        "--seed", str(args.seed),
        "--target", "game_result",
        "--value-head", architecture["value_head"],
        "--select-metric", "decisive",
        "--stem-channels", str(architecture["stem_channels"]),
        "--res-channels", ",".join(map(str, architecture["residual_channels"])),
        "--policy-head", architecture["policy_head"],
        "--policy-attention-channels",
        str(architecture["policy_attention_channels"]),
        "--se-reduction", str(architecture["se_reduction"]),
        "--moves-left-head-channels",
        str(architecture["moves_left_head_channels"]),
        "--moves-left-loss-weight", str(args.moves_left_loss_weight),
        "--save-selection-snapshots",
    ]
    train.append("--use-se-blocks" if architecture["use_se_blocks"]
                 else "--no-use-se-blocks")
    train.append("--side-policy-adapters" if architecture["side_policy_adapters"]
                 else "--no-side-policy-adapters")
    train.append("--moves-left-head" if moves_left else "--no-moves-left-head")
    if architecture["promotion_policy"]:
        train.append("--promotion-policy")
    if architecture["spatial_value_head"]:
        train.append("--spatial-value-head")
    if architecture["use_wdl_head"] and architecture["value_head"] == "scalar":
        train.append("--aux-wdl-head")

    audit = [
        "tools/audit_generation_data.py",
        "--raw-dir", str(paths["raw"]),
        "--reanalysis-dir", str(paths["reanalysis"]),
        "--processed-dir", str(paths["new_processed"]),
        "--expected-teachers", str(args.reanalysis_keep),
        "--expected-black-fraction", str(args.reanalysis_black_fraction),
        "--value-floor", str(args.value_floor),
        "--value-horizon", str(args.value_horizon),
    ]

    offline_report = paths["reports"] / "offline_model_diff.json"
    checkpoint_report = paths["reports"] / "checkpoint_screen.json"
    gate_report = paths["reports"] / "binding_gate.json"
    high_fidelity_report = paths["reports"] / "high_fidelity_gate.json"
    self_skew_report = paths["reports"] / "self_skew.json"
    checkpoint_screen = [
        "tools/checkpoint_screen.py",
        "--model-dir", str(paths["candidate_dir"]),
        "--incumbent", str(incumbent),
        "--output-model", str(paths["candidate"]),
        "--report-path", str(checkpoint_report),
        "--games", str(args.checkpoint_screen_games),
        "--probe-games", str(args.checkpoint_probe_games),
        "--sims", str(args.checkpoint_screen_sims),
        "--workers", str(args.workers),
        "--engine", args.engine,
        "--seed", str(seed + 350_000),
        "--stall-timeout", str(args.worker_stall_timeout),
    ]
    offline = [
        "tools/model_diff.py",
        "--candidate", str(paths["candidate"]),
        "--incumbent", str(incumbent),
        "--data-dir", str(paths["replay_processed"]),
        "--split", "test",
        "--max-positions", str(args.offline_positions),
        "--margin", str(args.offline_margin),
        "--report-path", str(offline_report),
    ]
    if args.reject_on_offline_regression:
        offline.append("--enforce")
    binding = [
        "tools/gate.py",
        "--model", str(paths["candidate"]),
        "--bar-model", str(incumbent),
        "--sparring-model", str(_absolute(args.sparring_model)),
        "--protocol", args.gate_protocol,
        "--engine", args.engine,
        "--sims", str(args.arena_sims),
        "--workers", str(args.workers),
        "--seed", str(seed + 400_000),
        "--stall-timeout", str(args.worker_stall_timeout),
        "--report-path", str(gate_report),
    ]
    high_fidelity = [
        "tools/confirm_candidates.py",
        "--bar", str(incumbent),
        "--candidate", f"bootstrap_candidate={paths['candidate']}",
        "--games", str(args.high_fidelity_games),
        "--sims", str(args.high_fidelity_sims),
        "--workers", str(args.workers),
        "--engine", args.engine,
        "--stall-timeout", str(args.worker_stall_timeout),
        "--seed", str(seed + 450_000),
        "--out", str(high_fidelity_report),
    ]
    self_skew = [
        "tools/match.py",
        "--model-a", str(paths["candidate"]),
        "--model-b", str(paths["candidate"]),
        "--games", str(args.self_skew_games),
        "--sims", str(args.arena_sims),
        "--engine", args.engine,
        "--workers", str(args.workers),
        "--seed", str(seed + 500_000),
        "--stall-timeout", str(args.worker_stall_timeout),
        "--report-path", str(self_skew_report),
    ]
    if args.book:
        book = str(_absolute(args.book))
        screen_offset = int(args.book_offset)
        screen_span = (
            args.checkpoint_probe_games + args.checkpoint_screen_games) // 2
        binding_offset = screen_offset + screen_span
        high_fidelity_offset = (
            binding_offset + _binding_book_span(args.gate_protocol))
        self_skew_offset = (
            high_fidelity_offset + args.high_fidelity_games // 2)
        checkpoint_screen += [
            "--book", book, "--book-offset", str(screen_offset)]
        binding += ["--book", book, "--book-offset", str(binding_offset)]
        high_fidelity += [
            "--book", book, "--book-offset", str(high_fidelity_offset),
        ]
        self_skew += [
            "--book", book, "--book-offset", str(self_skew_offset)]
        with open(book, encoding="utf-8") as handle:
            available = len(json.load(handle).get("entries", []))
        needed = self_skew_offset + args.self_skew_games // 2
        if needed > available:
            raise ValueError(
                f"book has {available} entries but this generation reserves "
                f"through {needed} (base offset {screen_offset})")
    return {
        "generate": {"commands": generated_commands,
                     "outputs": generated_outputs},
        "reanalyze": {"commands": [reanalyze],
                      "outputs": [str(paths["reanalysis"] / "reanalysis_summary.json")]},
        "process": {"commands": [process, audit],
                    "outputs": [str(paths["new_processed"] / "splits.npz"),
                                str(paths["new_processed"] /
                                    "generation_audit.json")]},
        "compose": {"commands": [compose],
                    "outputs": [str(paths["replay_processed"] / "replay_manifest.json")]},
        "train": {"commands": [train],
                  "outputs": [str(paths["training_candidate"])]},
        "checkpoint_screen": {"commands": [checkpoint_screen],
                              "outputs": [str(paths["candidate"]),
                                          str(checkpoint_report)]},
        "offline_gate": {"commands": [offline], "outputs": [str(offline_report)]},
        "binding_gate": {"commands": [binding], "outputs": [str(gate_report)]},
        "high_fidelity_gate": {"commands": [high_fidelity],
                               "outputs": [str(high_fidelity_report)]},
        "self_skew": {"commands": [self_skew], "outputs": [str(self_skew_report)]},
        "promote": {"commands": [], "outputs": [str(CHAMPION_POINTER)]},
    }


def _initial_state(args, generation, incumbent, architecture, paths,
                   replay_sources, plan):
    return {
        "schema_version": 1,
        "generation": generation,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "planned",
        "git_commit": _git_commit(),
        "incumbent": _rel(incumbent),
        "incumbent_sha256": _sha256(incumbent),
        "architecture": architecture,
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "paths": {key: _rel(value) for key, value in paths.items()},
        "replay_sources": [{"generation": gen, "path": _rel(path)}
                           for gen, path in replay_sources],
        "plan": plan,
        "phases": {},
    }


def _write_state(state, state_path):
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_json(state_path, state)


def _phase_outputs_exist(phase_plan):
    return all(Path(path).exists() for path in phase_plan.get("outputs", []))


def _validate_generation_summaries(phase_plan, minimum_success_rate=1.0):
    """Reject incomplete game batches before they enter processed replay."""
    rows = []
    for output in phase_plan.get("outputs", []):
        summary = _load_json(output)
        if not isinstance(summary, dict):
            raise RuntimeError(f"invalid generation summary: {output}")
        requested = int(summary.get("num_games_requested", 0))
        saved = int(summary.get("saved_games", -1))
        if requested <= 0 or saved < 0 or saved > requested:
            raise RuntimeError(
                f"invalid requested/saved game counts in {output}: "
                f"requested={requested}, saved={saved}")
        rate = saved / requested
        row = {
            "summary": _rel(output),
            "requested": requested,
            "saved": saved,
            "success_rate": rate,
            "failed": int(summary.get("failed_games", 0)),
            "timed_out": int(summary.get("timed_out_games", 0)),
            "skipped_empty": int(summary.get("skipped_empty", 0)),
        }
        rows.append(row)
        if rate < minimum_success_rate:
            raise RuntimeError(
                f"generation success rate {rate:.3f} is below required "
                f"{minimum_success_rate:.3f}: {output}")
    return rows


def _run_command(command, log_path):
    full = [PY, "-u"] + command
    print("$ " + " ".join(map(str, full)), flush=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(map(str, full)) + "\n")
        log.flush()
        process = subprocess.Popen(
            full, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def _archive_incumbent(incumbent):
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)
    digest = _sha256(incumbent)
    for existing in CHAMPIONS_DIR.glob("*.pt"):
        if _sha256(existing) == digest:
            return existing
    destination = CHAMPIONS_DIR / f"seed_{digest[:12]}.pt"
    if not destination.exists():
        shutil.copy2(incumbent, destination)
    return destination


def _promote(state, paths):
    candidate = Path(paths["candidate"])
    generation = state["generation"]
    digest = _sha256(candidate)
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)
    archived = CHAMPIONS_DIR / f"gen_{generation:04d}_{digest[:12]}.pt"
    if archived.exists() and _sha256(archived) != digest:
        raise RuntimeError(f"refusing to overwrite different champion: {archived}")
    if not archived.exists():
        shutil.copy2(candidate, archived)
    pointer = {
        "schema_version": 1,
        "generation": generation,
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": _rel(archived),
        "checkpoint_sha256": digest,
        "predecessor": state["incumbent"],
        "run_state": state["paths"]["state"],
        "gate_report": _rel(Path(paths["reports"]) / "binding_gate.json"),
        "high_fidelity_report": _rel(
            Path(paths["reports"]) / "high_fidelity_gate.json"),
        "checkpoint_screen_report": _rel(
            Path(paths["reports"]) / "checkpoint_screen.json"),
        "self_skew_report": _rel(Path(paths["reports"]) / "self_skew.json"),
    }
    _atomic_json(CHAMPION_POINTER, pointer)
    return pointer


def _acquire_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        lock = _load_json(path, {})
        if _pid_alive(lock.get("pid")):
            raise RuntimeError(
                f"generation already active under pid {lock.get('pid')}: {path}")
        path.unlink()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  handle)


def _validate_args(args):
    for name in ("games", "sims", "workers", "epochs", "patience",
                 "batch_size", "warmup_epochs", "value_horizon",
                 "reanalysis_sample", "reanalysis_keep", "reanalysis_sims",
                 "offline_positions", "checkpoint_screen_games",
                 "checkpoint_screen_sims", "high_fidelity_games",
                 "high_fidelity_sims", "self_skew_games",
                 "worker_stall_timeout"):
        if getattr(args, name, 1) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "checkpoint_probe_games", 40) <= 0:
        raise ValueError("--checkpoint-probe-games must be positive")
    for name in ("checkpoint_probe_games", "checkpoint_screen_games",
                 "high_fidelity_games", "self_skew_games"):
        if getattr(args, name, 2) % 2:
            raise ValueError(f"--{name.replace('_', '-')} must be even")
    if getattr(args, "book_offset", 0) < 0:
        raise ValueError("--book-offset must be non-negative")
    if args.reanalysis_keep > args.reanalysis_sample:
        raise ValueError("--reanalysis-keep must be <= --reanalysis-sample")
    if getattr(args, "league_games", 0) < 0:
        raise ValueError("--league-games must be non-negative")
    if getattr(args, "opponent_pool_size", 1) <= 0:
        raise ValueError("--opponent-pool-size must be positive")
    if getattr(args, "replay_generations", 1) <= 0:
        raise ValueError("--replay-generations must be positive")
    if not 0 <= args.reanalysis_black_fraction <= 1:
        raise ValueError("--reanalysis-black-fraction must be in [0, 1]")
    if not 0 <= getattr(args, "replay_balance_alpha", 0.0) <= 1:
        raise ValueError("--replay-balance-alpha must be in [0, 1]")
    if not 0 <= getattr(args, "value_floor", 0.5) <= 1:
        raise ValueError("--value-floor must be in [0, 1]")
    if getattr(args, "teacher_policy_multiplier", 1.0) <= 0:
        raise ValueError("--teacher-policy-multiplier must be positive")
    if not 0 < getattr(args, "min_generation_success_rate", 1.0) <= 1:
        raise ValueError("--min-generation-success-rate must be in (0, 1]")
    for name in ("lr", "lr_gamma", "ema_decay", "moves_left_loss_weight",
                 "offline_margin"):
        if getattr(args, name, 0) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if (args.generations > 1 and not args.promote_on_pass
            and not getattr(args, "continue_after_reject", False)):
        raise ValueError(
            "multiple generations require --promote-on-pass or "
            "--continue-after-reject")
    if args.promote_on_pass and args.gate_protocol != "full":
        raise ValueError("promotion requires --gate-protocol=full")


def _assert_resume_config(args, state):
    """Prevent a resumed generation from silently changing its experiment."""
    ignored = {
        "resume", "dry_run", "through_phase", "generations",
        # Operational hardening may be added between a safely stopped phase
        # and its resume without changing the experiment's statistical recipe.
        "worker_stall_timeout", "min_generation_success_rate", "workers",
        "reject_on_offline_regression",
    }
    current = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items() if key not in ignored
    }
    stored = {
        key: value for key, value in state.get("config", {}).items()
        if key not in ignored
    }
    differences = [
        key for key in sorted(set(current) | set(stored))
        if current.get(key) != stored.get(key)
    ]
    if differences:
        details = ", ".join(
            f"{key}={stored.get(key)!r}->{current.get(key)!r}"
            for key in differences[:8])
        raise ValueError(
            "resume configuration differs from state.json; start a new "
            f"generation or restore the original arguments ({details})")


def run_generation(args, generation=None):
    run_root = _absolute(args.run_root)
    generation = generation or _next_generation(run_root)
    paths = _paths_for_generation(run_root, generation)
    existing = _load_json(paths["state"])
    if existing and not args.resume:
        raise RuntimeError(
            f"generation {generation} already exists; pass --resume or choose another")

    if existing:
        _assert_resume_config(args, existing)
        incumbent = _absolute(existing["incumbent"])
        architecture = existing["architecture"]
        replay_sources = [(row["generation"], _absolute(row["path"]))
                          for row in existing.get("replay_sources", [])]
    else:
        incumbent = _resolve_champion(args.incumbent)
        if not incumbent.exists():
            raise FileNotFoundError(f"incumbent not found: {incumbent}")
        architecture = _checkpoint_spec(incumbent)
        replay_sources = _recent_replay_sources(
            run_root, generation, max(0, args.replay_generations - 1))

    anchor_data = _absolute(args.anchor_data)
    sparring = _absolute(args.sparring_model)
    if not anchor_data.exists():
        raise FileNotFoundError(f"anchor processed corpus not found: {anchor_data}")
    if not sparring.exists():
        raise FileNotFoundError(f"sparring model not found: {sparring}")

    plan = _command_plan(
        args, generation, incumbent, architecture, paths, replay_sources)
    if existing:
        # A code-hardened resume may legitimately change operational commands
        # for pending phases. Never rewrite the command record for data that a
        # completed phase already produced.
        old_plan = existing.get("plan", {})
        for phase, phase_state in existing.get("phases", {}).items():
            if phase_state.get("status") == "completed" and phase in old_plan:
                plan[phase] = old_plan[phase]
    state = existing or _initial_state(
        args, generation, incumbent, architecture, paths, replay_sources, plan)

    if args.dry_run:
        print(json.dumps({
            "generation": generation,
            "incumbent": _rel(incumbent),
            "architecture": architecture,
            "replay_sources": state["replay_sources"],
            "paths": state["paths"],
            "plan": plan,
        }, indent=2))
        return None

    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    _acquire_lock(paths["lock"])
    try:
        _archive_incumbent(incumbent)
        state["status"] = "running"
        state["plan"] = plan
        state.setdefault("executions", []).append({
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pid": os.getpid(),
            "git_commit": _git_commit(),
            "resume": bool(existing),
        })
        _write_state(state, paths["state"])
        stop_index = PHASES.index(args.through_phase) if args.through_phase else len(PHASES) - 1
        gate_passed = None
        for index, phase in enumerate(PHASES):
            if index > stop_index:
                break
            phase_plan = plan[phase]
            previous = state["phases"].get(phase, {})
            if (previous.get("status") == "completed"
                    and _phase_outputs_exist(phase_plan)):
                print(f"[{phase}] already completed; resuming after it")
                if phase == "generate":
                    state["generation_quality"] = _validate_generation_summaries(
                        phase_plan, args.min_generation_success_rate)
                    _write_state(state, paths["state"])
                if phase == "process" and not state.get("data_acceptance"):
                    _accept_generation_data(state, paths, run_root)
                    _write_state(state, paths["state"])
                if phase == "binding_gate":
                    report = _load_json(phase_plan["outputs"][0], {})
                    gate_passed = report.get("raw_verdict") == "PASS"
                if phase == "high_fidelity_gate":
                    report = _load_json(phase_plan["outputs"][0], {})
                    results = report.get("results", [])
                    gate_passed = bool(
                        results and results[0].get("passes_both_colors"))
                continue

            started = time.time()
            state["phases"][phase] = {
                "status": "running",
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "commands": phase_plan["commands"],
            }
            _write_state(state, paths["state"])
            print(f"\n=== generation {generation} / {phase} ===", flush=True)
            return_codes = []
            try:
                if phase == "promote":
                    if not gate_passed:
                        state["phases"][phase] = {
                            "status": "skipped", "reason": "binding gate failed"}
                        state["status"] = "rejected"
                        _write_state(state, paths["state"])
                        break
                    if args.promote_on_pass:
                        pointer = _promote(state, paths)
                        state["promotion"] = pointer
                        state["status"] = "promoted"
                    else:
                        state["status"] = "passed_not_promoted"
                        state["phases"][phase] = {
                            "status": "skipped",
                            "reason": "--promote-on-pass not supplied"}
                        _write_state(state, paths["state"])
                        break
                else:
                    log_path = paths["logs"] / f"{phase}.log"
                    for command in phase_plan["commands"]:
                        return_codes.append(_run_command(command, log_path))
                    if (phase == "train" and any(code != 0 for code in return_codes)
                            and Path(paths["training_rejection"]).exists()):
                        state["phases"][phase].update({
                            "status": "completed", "verdict": "REJECT",
                            "rejection_report": _rel(paths["training_rejection"]),
                            "return_codes": return_codes,
                            "elapsed_sec": round(time.time() - started, 1),
                        })
                        state["status"] = "rejected_training"
                        _write_state(state, paths["state"])
                        print("Fixed-incumbent checkpoint selection rejected "
                              "every epoch.")
                        break
                    allowed_failure = phase in ("offline_gate", "binding_gate")
                    if any(code != 0 for code in return_codes) and not allowed_failure:
                        raise RuntimeError(
                            f"{phase} command failed with {return_codes}")
                    if not _phase_outputs_exist(phase_plan):
                        raise RuntimeError(f"{phase} did not produce its declared outputs")
                    if phase == "generate":
                        state["generation_quality"] = _validate_generation_summaries(
                            phase_plan, args.min_generation_success_rate)
                    if phase == "process":
                        _accept_generation_data(state, paths, run_root)
                    if phase == "offline_gate" and any(code != 0 for code in return_codes):
                        state["phases"][phase].update({
                            "status": "completed", "verdict": "FAIL",
                            "return_codes": return_codes,
                            "elapsed_sec": round(time.time() - started, 1),
                        })
                        state["status"] = "rejected_offline"
                        _write_state(state, paths["state"])
                        print("Offline regression gate rejected the candidate.")
                        break
                    if phase == "offline_gate":
                        report = _load_json(phase_plan["outputs"][0], {})
                        warnings = list(report.get("failures", []))
                        if warnings:
                            state["phases"][phase].update({
                                "verdict": "WARN",
                                "warnings": warnings,
                            })
                            print(
                                "Offline comparison raised advisory warnings; "
                                "continuing to the binding game gate.",
                                flush=True,
                            )
                    if phase == "binding_gate":
                        report = _load_json(phase_plan["outputs"][0], {})
                        gate_passed = report.get("raw_verdict") == "PASS"
                        if not gate_passed:
                            state["phases"][phase].update({
                                "status": "completed", "verdict": "FAIL",
                                "return_codes": return_codes,
                                "elapsed_sec": round(time.time() - started, 1),
                            })
                            state["status"] = "rejected"
                            _write_state(state, paths["state"])
                            print("Binding gate rejected the candidate; self-skew skipped.")
                            break
                    if phase == "high_fidelity_gate":
                        report = _load_json(phase_plan["outputs"][0], {})
                        results = report.get("results", [])
                        gate_passed = bool(
                            results and results[0].get("passes_both_colors"))
                        if not gate_passed:
                            state["phases"][phase].update({
                                "status": "completed", "verdict": "FAIL",
                                "return_codes": return_codes,
                                "elapsed_sec": round(time.time() - started, 1),
                            })
                            state["status"] = "rejected_high_fidelity"
                            _write_state(state, paths["state"])
                            print("High-fidelity two-color gate rejected the "
                                  "candidate; self-skew skipped.")
                            break

                if state["phases"].get(phase, {}).get("status") == "running":
                    state["phases"][phase].update({
                        "status": "completed",
                        "return_codes": return_codes,
                        "elapsed_sec": round(time.time() - started, 1),
                    })
                _write_state(state, paths["state"])
            except Exception as exc:
                state["phases"][phase].update({
                    "status": "failed", "error": str(exc),
                    "return_codes": return_codes,
                    "elapsed_sec": round(time.time() - started, 1),
                })
                state["status"] = "failed"
                _write_state(state, paths["state"])
                raise

        if state["status"] == "running":
            state["status"] = "partial" if args.through_phase else "complete"
            _write_state(state, paths["state"])
        return state["status"]
    finally:
        try:
            paths["lock"].unlink()
        except FileNotFoundError:
            pass


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    ap.add_argument("--incumbent", default=None,
                    help="explicit champion; otherwise champion.json then V20")
    ap.add_argument("--anchor-data", default=str(DEFAULT_ANCHOR_DATA),
                    help="immutable processed replay anchor")
    ap.add_argument("--sparring-model", default=str(DEFAULT_SPARRING))
    ap.add_argument("--games", type=int, default=BOOTSTRAP_GAMES)
    ap.add_argument("--league-games", type=int, default=0,
                    help="optional champion-league games in addition to the "
                         "500-game production self-play batch")
    ap.add_argument("--opponent-pool-size", type=int, default=5)
    ap.add_argument("--sims", type=int, default=BOOTSTRAP_SIMS)
    ap.add_argument("--workers", type=int, default=DEFAULT_GAME_WORKERS)
    ap.add_argument("--worker-stall-timeout", type=float, default=600.0,
                    help="fail a generation/reanalysis/match after no progress")
    ap.add_argument("--engine", choices=("python", "native"), default="native")
    ap.add_argument("--reanalysis-sample", type=int,
                    default=BOOTSTRAP_REANALYSIS_SAMPLE)
    ap.add_argument("--reanalysis-keep", type=int,
                    default=BOOTSTRAP_REANALYSIS_KEEP)
    ap.add_argument("--reanalysis-sims", type=int,
                    default=BOOTSTRAP_REANALYSIS_SIMS)
    ap.add_argument("--reanalysis-black-fraction", type=float, default=0.60,
                    help="share of deep-search teachers reserved for Black")
    ap.add_argument("--replay-generations", type=int, default=4,
                    help="maximum generation datasets in replay, including current")
    ap.add_argument("--replay-balance-alpha", type=float, default=0.5,
                    help="smooth train replay across side/outcome/phase strata")
    ap.add_argument("--min-generation-success-rate", type=float, default=1.0,
                    help="minimum saved/requested ratio for every game batch")
    ap.add_argument("--epochs", type=int, default=SCRATCH_EPOCHS)
    ap.add_argument("--patience", type=int, default=SCRATCH_PATIENCE)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=SCRATCH_LR,
                    help="from-scratch learning rate")
    ap.add_argument("--lr-gamma", type=float, default=0.95)
    ap.add_argument("--warmup-epochs", type=int,
                    default=SCRATCH_WARMUP_EPOCHS)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--value-floor", type=float,
                    default=BOOTSTRAP_VALUE_FLOOR)
    ap.add_argument("--value-horizon", type=int,
                    default=BOOTSTRAP_VALUE_HORIZON)
    ap.add_argument("--teacher-policy-multiplier", type=float,
                    default=TEACHER_POLICY_MULTIPLIER,
                    help="effective policy weight for policy-only teachers")
    ap.add_argument("--moves-left-head", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--moves-left-loss-weight", type=float, default=0.01)
    ap.add_argument("--offline-positions", type=int, default=8192)
    ap.add_argument("--offline-margin", type=float, default=0.01)
    ap.add_argument(
        "--reject-on-offline-regression", action="store_true",
        help="restore the legacy hard offline rejection; by default held-out "
             "policy/sign regressions are recorded as warnings and every "
             "successfully trained candidate reaches the binding game gate",
    )
    ap.add_argument("--gate-protocol", choices=("quick", "full"), default="full")
    ap.add_argument("--arena-sims", type=int, default=ITERATE_ARENA_SIMS)
    ap.add_argument("--checkpoint-screen-games", type=int, default=200,
                    help="same-opening games per preserved training checkpoint")
    ap.add_argument("--checkpoint-probe-games", type=int, default=40,
                    help="cheap paired games per checkpoint before finalists")
    ap.add_argument("--checkpoint-screen-sims", type=int, default=400)
    ap.add_argument("--high-fidelity-games", type=int, default=80,
                    help="calibrated final confirmation games before promotion")
    ap.add_argument("--high-fidelity-sims", type=int, default=800)
    ap.add_argument("--self-skew-games", type=int, default=80)
    ap.add_argument("--book", default=None,
                    help="fixed paired opening book shared by every play phase")
    ap.add_argument("--book-offset", type=int, default=0,
                    help="first entry reserved for this generation; later "
                         "phases receive disjoint blocks automatically")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--promote-on-pass", action="store_true",
                    help="archive a passing candidate and advance champion.json")
    ap.add_argument("--continue-after-reject", action="store_true",
                    help="continue accumulating accepted data with the current "
                         "champion after a candidate rejection")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--through-phase", choices=PHASES, default=None,
                    help="stop after this phase, leaving a resumable run")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate inputs and print the complete plan; write nothing")
    return ap


def main():
    args = build_parser().parse_args()
    _validate_args(args)
    run_root = _absolute(args.run_root)
    root_lock = run_root / "run.lock"
    if not args.dry_run:
        _acquire_lock(root_lock)
    try:
        resume_once = args.resume
        for _ in range(args.generations):
            generation = (_latest_generation(run_root) if resume_once
                          else _next_generation(run_root))
            if generation is None:
                raise ValueError("--resume requested but no generation exists")
            status = run_generation(args, generation=generation)
            resume_once = False
            args.resume = False
            can_continue = status == "promoted" or (
                args.continue_after_reject
                and status in ("rejected", "rejected_offline",
                               "rejected_training", "rejected_high_fidelity",
                               "passed_not_promoted"))
            if args.dry_run or not can_continue:
                break
    finally:
        if not args.dry_run:
            try:
                root_lock.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
