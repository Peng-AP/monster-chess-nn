"""Recoverable free-play gate with equal-color, distinct-endpoint scoring.

Endpoint-uniform scores are NOT naturally sampled win rates. Both are saved.
History and continuation collisions are audited, not assumed impossible: tree
reuse and repetition retain information outside an opening FEN. SEs are nominal
descriptive errors, not independent-scenario confidence bounds. Confirmation's
novel subset is conditional on avoiding first-leg keys; it supplies additional
coverage, while the original score floors apply to the full unique confirmation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from match import run_match
from match_evidence import atomic_json, digest, model_identity, read_rows, runtime_identity
from worker_lease import exclusive_workers
from free_gate_stats import (SCORING_VERSION, AGGREGATE_MIN, PER_SIDE_BAND,
                             covered, leg_stats, opening_key, unique_rows, verdict)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dedup(log_path):
    rows = read_rows(log_path)
    return unique_rows(rows), len(rows)


def protocol(args):
    return {"version": SCORING_VERSION, "runtime": runtime_identity(),
            "engine": "native", "sims": args.sims, "workers": args.workers,
            "opening_temp_plies": 16, "opening_temperature": .5,
            "aggregate_min": AGGREGATE_MIN, "per_side_band": PER_SIDE_BAND,
            "target_per_side": args.target_per_side, "par_per_side": args.par_per_side,
            "batch_games": args.batch_games, "seed": args.seed,
            "budget_min": args.budget_min,
            "confirmation": "full_unique_score_plus_novel_endpoint_coverage",
            "uncertainty": "nominal_draw_aware_SE; histories/overlap_may_correlate"}


def cache_key(bar, config):
    return digest({"bar_sha256": bar["sha256"],
                   "protocol": {k: v for k, v in config.items() if k not in
                                ("target_per_side", "par_per_side", "seed",
                                 "budget_min", "batch_games")}})


def leg(name, model_a, model_b, args, out_dir, budget_min, seed, excluded=(),
        target=None):
    out_dir = Path(out_dir)
    state_path = out_dir / f"{name}.json"
    state = load_json(state_path) if state_path.exists() else {
        "name": name, "batches": [], "active_seconds": 0, "complete": False}
    # Completed batches must pass the same manifest/task checks as partial
    # ones. run_match's no-pending-task path validates without starting workers.
    for batch in state["batches"]:
        if batch["complete"]:
            run_match(model_a, model_b, games=batch["games"], sims=args.sims,
                      sims_b=args.sims, workers=args.workers, engine="native",
                      seed=batch["seed"], opening_temp_plies=16,
                      game_log=str(out_dir / batch["log"]), resume=True)
    def read_all():
        return [r for batch in state["batches"]
                for r in read_rows(out_dir / batch["log"], allow_partial_tail=True)]
    rows = read_all()
    target = args.target_per_side if target is None else target
    metric = "novel" if name.endswith("confirm") else "unique"
    previous_seconds = state["active_seconds"]
    start = time.monotonic()

    def save():
        state["active_seconds"] = previous_seconds + time.monotonic() - start
        state["stats"] = leg_stats(rows, excluded)
        atomic_json(state_path, state)

    try:
        while True:
            stats = leg_stats(rows, excluded)
            pending = next((b for b in state["batches"] if not b["complete"]), None)
            if pending is None:
                if covered(stats[metric], target):
                    state["complete"] = True
                    break
                remaining = budget_min * 60 - previous_seconds - (time.monotonic() - start)
                if remaining <= 0:
                    state["stop_reason"] = "budget_exhausted"
                    break
                count = args.batch_games
                if rows and state["active_seconds"] > 0:
                    seconds_per_game = state["active_seconds"] / len(rows)
                    count = min(count, max(2, int(remaining / max(seconds_per_game, .01))))
                    count -= count % 2
                i = len(state["batches"])
                if i >= 10000:
                    state["stop_reason"] = "seed_range_exhausted"
                    break
                pending = {"index": i, "games": count, "seed": seed + i * 100000,
                           "log": f"{name}_b{i:05d}.jsonl", "complete": False}
                state["batches"].append(pending)
                save()  # persist scheduling intent BEFORE launching workers
            run_match(model_a, model_b, games=pending["games"], sims=args.sims,
                      sims_b=args.sims, workers=args.workers, engine="native",
                      seed=pending["seed"], opening_temp_plies=16,
                      game_log=str(out_dir / pending["log"]), resume=True)
            pending["complete"] = True
            rows = read_all()
            save()
            counts = state["stats"][metric]["sides"]
            print(f"{name}: {len(rows)} sampled; {metric} W {counts['white']['n']} "
                  f"B {counts['black']['n']}; {state['active_seconds']/60:.1f}m", flush=True)
    finally:
        rows = read_all()
        save()
    return state["stats"], rows, state


@exclusive_workers
def run_gate(args):
    if args.resume:
        out_dir = Path(args.resume).resolve()
        manifest = load_json(out_dir / "manifest.json")
        expected = {"model": model_identity(args.model), "bar": model_identity(args.bar_model),
                    "protocol": protocol(args)}
        if any(manifest[k] != v for k, v in expected.items()):
            raise ValueError("gate resume provenance/configuration mismatch")
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        out_dir = Path(args.run_dir or ROOT / "benchmarks" / "free_gate" / stamp).resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
        manifest = {"model": model_identity(args.model), "bar": model_identity(args.bar_model),
                    "protocol": protocol(args), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "run_dir": str(out_dir)}
        atomic_json(out_dir / "manifest.json", manifest)
    report_path = out_dir / "report.json"
    if args.report_path:
        destination = Path(args.report_path).resolve()
        if destination.exists() and load_json(destination).get("run_dir") != str(out_dir):
            raise FileExistsError(f"refusing to replace another run's report: {destination}")
    if report_path.exists() and load_json(report_path).get("complete"):
        out = load_json(report_path)
        if not all(Path(p).exists() and model_identity(p)["sha256"] == h
                   for p, h in out["evidence_hashes"].items()):
            raise ValueError("completed gate evidence was modified")
        if args.report_path:
            atomic_json(destination, out)
        return out
    print(f"FREE GATE {SCORING_VERSION}: {out_dir}", flush=True)
    cache = ROOT / "benchmarks" / "free_par_v2" / (cache_key(manifest["bar"], manifest["protocol"]) + ".json")
    par = None
    par_sources = {}
    if cache.exists():
        cached = load_json(cache)
        if (cached.get("key") == cache.stem and covered(cached["stats"]["unique"], args.par_per_side)
                and not cached["stats"]["endpoint_outcome_conflicts"]):
            if cached.get("source_hashes") and all(
                    Path(p).exists() and model_identity(p)["sha256"] == h
                    for p, h in cached["source_hashes"].items()):
                par = cached["stats"]
                par_sources = cached["source_hashes"]
    spent = 0
    # Keep local par accounting even if this run has since populated the cache.
    if (out_dir / "par.json").exists():
        spent = load_json(out_dir / "par.json")["active_seconds"] / 60
    if par is None:
        par, _, par_state = leg("par", args.bar_model, args.bar_model, args, out_dir,
                                args.budget_min * .28, args.seed + 1000000000,
                                target=args.par_per_side)
        spent = par_state["active_seconds"] / 60
        par_sources = {str(out_dir / b["log"]): model_identity(out_dir / b["log"])["sha256"]
                       for b in par_state["batches"]}
        if covered(par["unique"], args.par_per_side) and not par["endpoint_outcome_conflicts"]:
            atomic_json(cache, {"key": cache.stem, "stats": par,
                "source_hashes": {str(out_dir / b["log"]): model_identity(out_dir / b["log"])["sha256"]
                                  for b in par_state["batches"]}})
    legs, first_rows, all_rows = {}, [], []
    out = {**manifest, "instrument": SCORING_VERSION, "bar_free_par": par,
           "par_cache": str(cache), "legs": legs, "complete": False,
           "verdict": "INCONCLUSIVE", "eligible": False, "confirmed": False}
    atomic_json(report_path, out)
    allocation_path = out_dir / "allocation.json"
    allocation = load_json(allocation_path) if allocation_path.exists() else {
        "leg_budget_min": max(0, args.budget_min - spent) / 2}
    atomic_json(allocation_path, allocation)
    for i, name in enumerate(("vs_bar", "vs_bar_confirm")):
        legs[name], rows, _ = leg(name, args.model, args.bar_model, args, out_dir,
            allocation["leg_budget_min"], args.seed + i * 2000000000,
            excluded={opening_key(r) for r in first_rows})
        all_rows.extend(rows)
        if i == 0:
            first_rows = rows
        out.update(verdict(par, legs, args.target_per_side, args.par_per_side))
        atomic_json(report_path, out)
    out["combined_h2h"] = leg_stats(all_rows)
    if out["combined_h2h"]["endpoint_outcome_conflicts"]:
        out.update(verdict="INCONCLUSIVE", raw_verdict="INCONCLUSIVE", eligible=False, confirmed=False)
        out["inconclusive_reasons"].append("conflicting continuation outcomes across legs")
    out["complete"] = True  # campaign ended, not necessarily sufficient evidence
    out["evidence_hashes"] = {**par_sources, **{str(p): model_identity(p)["sha256"]
                              for p in out_dir.glob("*.jsonl")}}
    atomic_json(report_path, out)
    if args.report_path:
        atomic_json(destination, out)
    print(f"VERDICT: {out['verdict']}; report {report_path}", flush=True)
    return out


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bar-model", required=True)
    ap.add_argument("--bar-name", default=None, help="display compatibility only")
    ap.add_argument("--sims", type=int, default=3200)
    ap.add_argument("--target-per-side", type=int, default=100)
    ap.add_argument("--par-per-side", type=int, default=100)
    ap.add_argument("--target-unique", type=int, help="legacy total; divided equally between colors")
    ap.add_argument("--par-unique", type=int, help="legacy total; divided equally between colors")
    ap.add_argument("--budget-min", type=float, default=180)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-games", type=int, default=32)
    ap.add_argument("--seed", type=int, default=6200000)
    ap.add_argument("--report-path")
    ap.add_argument("--run-dir", help="new isolated directory (must not exist)")
    ap.add_argument("--resume", help="existing run directory; identical arguments required")
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if args.target_unique is not None:
        args.target_per_side = math.ceil(args.target_unique / 2)
    if args.par_unique is not None:
        args.par_per_side = math.ceil(args.par_unique / 2)
    if any(v <= 0 for v in (args.sims, args.target_per_side, args.par_per_side,
                            args.budget_min, args.workers, args.batch_games)):
        ap.error("simulations, coverage, budget, workers and batch size must be positive")
    if args.workers > 8 or args.batch_games % 2 or args.batch_games > 1000:
        ap.error("at most eight workers; batch size must be even and <= 1000")
    run_gate(args)


if __name__ == "__main__":
    main()
