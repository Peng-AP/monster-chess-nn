"""Deep continuations from positions where the value estimates disagree (gen51).

Stage 2 of docs/plans/GEN51_STRENGTH_PLAN.md. This extends the root rule that
produced v28 (campaigns/value_calibration/value_calibration.py `select`) from
a separate experiment to generation data:

* Parents are completed normal-start self-play games of the current
  generation, in a seeded random order, one root per parent family.
* Phases rotate black, black, white_first, white_second (50/25/25).
* Candidate roots: primitive plies 4..120 in that phase, excluding records
  whose policy has a single move (forced replies and exact-finisher moves).
* Score = |v_ref - v_player| + |v_player - q_search|: raw network values of
  the reference and the player checkpoints, and the player's recorded search
  value, all side-to-move. The highest score wins; ties go to the earlier ply.
  No move names, no outcomes and no hand-written tactics enter selection.
* Each root receives `continuations` completed games by the player at
  `sims`, with separate seeds, written by stateful_generation's own
  play/receipt machinery. `source_record` links each game to its parent, so
  processing keeps it in the parent's train/validation/test family.

Continuations keep the historical exploration length (no `temperature_plies`):
they are outcome labels for a chosen root, not coverage.
"""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from match_evidence import atomic_json, digest, file_hash, model_identity, runtime_identity
from stateful_generation import run_batch
from worker_lease import worker_lease

PHASE_CYCLE = ("black", "black", "white_first", "white_second")


def phase(record):
    if record["current_player"] == "black":
        return "black"
    return "white_second" if record["half"] else "white_first"


def eligible(record, lo, hi):
    return lo <= len(record["state"]["moves"]) <= hi and len(record["policy"]) > 1


def raw_values(model, records):
    import numpy as np
    import torch
    from encoding import fen_to_tensor
    out = []
    with torch.no_grad():
        for start in range(0, len(records), 256):
            batch = records[start:start + 256]
            x = np.stack([fen_to_tensor(r["fen"], r["current_player"] == "white", bool(r["half"]), 15)
                          for r in batch])
            value, _ = model(torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).cuda())
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite raw value")
            out.extend(value.flatten().cpu().tolist())
    return out


def select_roots(config):
    import torch
    from train import load_model_for_inference
    raw = ROOT / config["raw_dir"]
    parents = sorted((raw / "selfplay").glob("*.jsonl"))
    random.Random(config["seed"]).shuffle(parents)
    player = load_model_for_inference(str(ROOT / config["player"]), "cuda")[0].eval()
    reference = load_model_for_inference(str(ROOT / config["reference"]), "cuda")[0].eval()
    torch.set_num_threads(1)
    roots, skipped = [], 0
    for path in parents:
        if len(roots) == config["roots"]:
            break
        desired = PHASE_CYCLE[len(roots) % len(PHASE_CYCLE)]
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        choices = [(i, r) for i, r in enumerate(rows)
                   if phase(r) == desired and eligible(r, config["min_ply"], config["max_ply"])]
        if not choices:
            skipped += 1
            continue
        records = [r for _, r in choices]
        vp, vr = raw_values(player, records), raw_values(reference, records)
        scored = [dict(line=i + 1, ply=len(r["state"]["moves"]),
                       score=abs(b - a) + abs(a - float(r["mcts_value"])),
                       player_value=a, reference_value=b, search_value=float(r["mcts_value"]))
                  for (i, r), a, b in zip(choices, vp, vr)]
        best = max(scored, key=lambda s: (s["score"], -s["line"]))
        record = rows[best["line"] - 1]
        roots.append(dict(index=len(roots), phase=desired, parent=path.relative_to(raw).as_posix(),
                          state=record["state"], selection=best, candidates=len(scored)))
    if len(roots) != config["roots"]:
        raise ValueError(f"only {len(roots)} of {config['roots']} roots found")
    return dict(roots=roots, parents_without_phase=skipped)


def tasks_for(config, roots):
    return [dict(id=f"disagree/root_{r['index']:04d}_{j}", kind="disagree", model=config["player"],
                 sims=config["sims"], seed=config["seed"] + 100000 + 10 * r["index"] + j,
                 state=r["state"], source_record=dict(path=r["parent"], line=r["selection"]["line"]))
            for r in roots for j in range(config["continuations"])]


def run(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(config=config, runtime=runtime_identity(), implementation=file_hash(__file__),
                    models=[model_identity(ROOT / config[k]) for k in ("player", "reference")])
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("continuation provenance changed; use a new output directory")
    atomic_json(manifest, identity)
    with worker_lease():
        roots_path = out / "roots.json"
        if not roots_path.exists():
            atomic_json(roots_path, dict(select_roots(config), config_sha256=digest(config)))
        selected = json.loads(roots_path.read_text())
        if selected["config_sha256"] != digest(config):
            raise ValueError("roots were selected under another configuration")
        tasks = tasks_for(config, selected["roots"])
        run_batch(tasks, out / "raw", out / "receipts", config["workers"])
    games = sorted((out / "raw").rglob("*.jsonl"))
    if len(games) != len(tasks):
        raise ValueError("continuation games missing")
    by_phase = {p: sum(r["phase"] == p for r in selected["roots"]) for p in dict.fromkeys(PHASE_CYCLE)}
    atomic_json(out / "summary.json", dict(
        complete=True, roots=len(selected["roots"]), games=len(games), by_phase=by_phase,
        rows=sum(len(g.read_text().splitlines()) for g in games),
        roots_sha256=file_hash(roots_path), manifest_sha256=file_hash(manifest)))
    print(f"DISAGREEMENT CONTINUATIONS COMPLETE: {len(games)} games", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ("raw_dir", "player", "reference", "roots", "continuations", "sims", "seed",
                "workers", "min_ply", "max_ply"):
        if key not in config:
            raise ValueError(f"missing config key {key}")
    if not 1 <= config["workers"] <= 8:
        raise ValueError("workers must be 1..8")
    run(config, args.out)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
