"""Bounded sequential candidate validation. Never promotes or starts training."""
import argparse
from pathlib import Path

import gate_free
from gate_free import leg, load_json, run_gate
from match_evidence import atomic_json, model_identity, runtime_identity, file_hash
from worker_lease import exclusive_workers


@exclusive_workers
def run(args):
    directory = Path(args.output).resolve()
    manifest_path = directory / "manifest.json"
    manifest = {"implementation_sha256": file_hash(__file__),
                "candidate": model_identity(args.model), "bar": model_identity(args.bar_model),
                "opponents": {name: model_identity(path) for name, path in
                              (("gen41", args.gen41), ("v24", args.v24))},
                "runtime": runtime_identity(),
                "config": {k: v for k, v in vars(args).items() if k != "resume"}}
    if directory.exists():
        if not args.resume or load_json(manifest_path) != manifest:
            raise ValueError("campaign exists or resume provenance/configuration changed")
    else:
        directory.mkdir(parents=True)
        atomic_json(manifest_path, manifest)
    result_path = directory / "report.json"
    report = {"manifest": manifest, "complete": False, "promotion": "none; results require review"}
    atomic_json(result_path, report)
    gate_dir = directory / "binding"
    gate_args = gate_free.parser().parse_args([
        "--model", args.model, "--bar-model", args.bar_model,
        "--sims", str(args.sims), "--workers", str(args.workers),
        "--seed", str(args.seed), "--batch-games", str(args.batch_games),
        "--target-per-side", str(args.target_per_side),
        "--par-per-side", str(args.target_per_side),
        "--budget-min", str(args.gate_budget_min),
        "--resume" if gate_dir.exists() else "--run-dir", str(gate_dir)])
    report["binding_gate"] = run_gate(gate_args)
    atomic_json(result_path, report)
    for index, (name, opponent) in enumerate((("gen41", args.gen41), ("v24", args.v24),
                                              ("selfplay", args.model))):
        print(f"\nSUPPORTING LEG {name}, budget {args.support_budget_min}m", flush=True)
        stats, rows, _ = leg(name, args.model, opponent, args, directory,
                             args.support_budget_min, args.seed + (index + 3) * 1000000000)
        report[name] = {"stats": stats, "interpretation": "diagnostic; no opponent self-par calibration"}
        if name == "selfplay":
            white_results = [r["result_for_a"] if r["a_is_white"] else -r["result_for_a"] for r in rows]
            report[name]["actual_sampled_outcomes"] = {
                "white_wins": sum(r == 1 for r in white_results),
                "black_wins": sum(r == -1 for r in white_results),
                "draws": sum(abs(r) < 1 for r in white_results)}
        atomic_json(result_path, report)
    report["complete"] = True
    atomic_json(result_path, report)
    print(f"CAMPAIGN COMPLETE: {result_path}; no model promoted", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bar-model", required=True)
    ap.add_argument("--gen41", default="models/candidates/bootstrap_main_gen_0041/screen_nominee.pt")
    ap.add_argument("--v24", default="models/bootstrap_v24/best_value_net.pt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--sims", type=int, default=3200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target-per-side", type=int, default=100)
    ap.add_argument("--batch-games", type=int, default=32)
    ap.add_argument("--gate-budget-min", type=float, default=180)
    ap.add_argument("--support-budget-min", type=float, default=50)
    ap.add_argument("--seed", type=int, default=7200000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if min(args.sims, args.workers, args.target_per_side, args.batch_games,
           args.gate_budget_min, args.support_budget_min) <= 0:
        ap.error("all count/depth/budget arguments must be positive")
    if args.workers > 8 or args.batch_games % 2 or args.batch_games > 1000:
        ap.error("at most eight workers; even batches of at most 1000 games")
    run(args)


if __name__ == "__main__":
    main()
