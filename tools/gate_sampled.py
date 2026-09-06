"""Preregistered fixed-sample gate with durable journals and coverage diagnostics.

This is a new instrument, not a reinterpretation of completed v2 campaigns.
Each leg runs its entire fixed task schedule. A soft deadline can prevent the
next leg starting; it never censors slower games or silently adds more samples.
No cached par, no unseen-endpoint quota, no automatic release promotion.
"""
import argparse
import inspect
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from gate_free import load_json
from free_gate_stats import AGGREGATE_MIN, PER_SIDE_BAND, leg_stats, opening_key
from match import build_tasks, run_match
from match_evidence import atomic_json, file_hash, model_identity, read_rows, runtime_identity, task_id
from sampled_gate_stats import SCORING_VERSION, compare, self_par, verdict
from worker_lease import exclusive_workers

MATCH_SIGNATURE = inspect.signature(run_match)


def protocol(args):
    return {"version": SCORING_VERSION, "runtime": runtime_identity(),
            "implementation": {name: file_hash(ROOT / "tools" / name) for name in
                               ("gate_sampled.py", "sampled_gate_stats.py")},
            "engine": "native", "sims": args.sims, "workers": args.workers,
            "opening_temp_plies": 16, "opening_temperature": .5,
            "aggregate_min": AGGREGATE_MIN, "per_side_band": PER_SIDE_BAND,
            "target_per_side": args.target_per_side, "par_games": args.par_games,
            "seed": args.seed, "budget_min": args.budget_min,
            "confirmation": "fresh_independent_RNG_block; repeats_retained",
            "self_par": "all_actual_color_outcomes; complementary_estimates",
            "uncertainty": "draw-aware nominal SE; not a non-inferiority proof"}


def schedule(args):
    return [dict(name="par", games=args.par_games, seed=args.seed + 100000),
            dict(name="vs_bar", games=2 * args.target_per_side, seed=args.seed),
            dict(name="vs_bar_confirm", games=2 * args.target_per_side,
                 seed=args.seed + 200000)]


def validate_evidence(report):
    hashes = report.get("evidence_hashes", {})
    if not hashes or not all(Path(p).is_file() and file_hash(p) == h for p, h in hashes.items()):
        raise ValueError("completed sampled-gate evidence missing or modified")


def match_settings(model_a, model_b, item, args):
    """Read-only expected journal settings, including every search default."""
    bound = MATCH_SIGNATURE.bind(model_a, model_b, games=item["games"], sims=args.sims,
                                 sims_b=args.sims, workers=args.workers, engine="native",
                                 seed=item["seed"], opening_temp_plies=16)
    bound.apply_defaults()
    settings = dict(bound.arguments)
    for key in ("game_log", "checkpoint_path", "resume", "stall_timeout"):
        settings.pop(key)
    settings.update(model_a=model_identity(model_a), model_b=model_identity(model_b),
                    runtime=runtime_identity())
    return settings


def validate_report(report, model, bar, sims, target_per_side, par_games):
    """Recompute a finished gate from immutable journals without launching games.

    Stored summary arithmetic is not trusted merely because the source-file
    hashes still match. Raises on provenance/schema drift; returns the computed
    verdict for complete compatible evidence.
    """
    if not report.get("complete") or report.get("instrument") != SCORING_VERSION:
        raise ValueError("sampled report is incomplete or uses another instrument")
    args = argparse.Namespace(**report["protocol"])
    if (args.sims, args.target_per_side, args.par_games) != (sims, target_per_side, par_games):
        raise ValueError("sampled report depth/sample counts mismatch")
    if report["protocol"] != protocol(args):
        raise ValueError("sampled report runtime/implementation/protocol changed")
    if report["model"] != model_identity(model) or report["bar"] != model_identity(bar):
        raise ValueError("sampled report model identity mismatch")
    directory = Path(report["run_dir"])
    manifest = load_json(directory / "manifest.json")
    if any(manifest[k] != report[k] for k in ("model", "bar", "protocol", "schedule", "run_dir")):
        raise ValueError("sampled report does not match its run manifest")
    if report["schedule"] != schedule(args):
        raise ValueError("sampled report task schedule changed")
    validate_evidence(report)
    par, first, combined, legs = self_par([]), [], [], {}
    expected_evidence = {str(directory / "manifest.json")}
    for item in report["schedule"]:
        name = item["name"]
        log = directory / (name + ".jsonl")
        meta = log.with_suffix(".jsonl.manifest.json")
        if not log.exists() or not meta.exists():
            # Budget-limited runs can legitimately end before a required leg.
            if report.get("verdict") == "INCONCLUSIVE" and not report.get("eligible"):
                return "INCONCLUSIVE"
            raise ValueError("sampled report is missing a required leg")
        expected_evidence.update((str(log), str(meta)))
        tasks = build_tasks(item["games"], item["seed"], 16)
        expected_tasks = {task_id(t): t for t in tasks}
        settings = match_settings(bar if name == "par" else model, bar, item, args)
        if load_json(meta) != {"schema_version": 1, "settings": settings, "tasks": list(expected_tasks)}:
            raise ValueError("sampled report journal settings/tasks mismatch")
        rows = read_rows(log)
        if len(rows) != len(tasks) or {r["task_id"] for r in rows} != set(expected_tasks):
            raise ValueError("sampled report tasks missing or duplicated")
        for row in rows:
            task = expected_tasks[row["task_id"]]
            if (row["a_is_white"], row["seed"], row["pair"]) != (task[0], task[1], task[4]):
                raise ValueError("sampled report row task metadata changed")
        stats = leg_stats(rows, {opening_key(r) for r in first})
        if name == "par":
            par = self_par(rows)
            if report["bar_free_par"] != {"calibration": par, "diagnostics": stats}:
                raise ValueError("sampled report calibration arithmetic changed")
        else:
            legs[name] = stats
            combined.extend(rows)
            if name == "vs_bar":
                first = rows
    if set(report["evidence_hashes"]) != expected_evidence:
        raise ValueError("sampled report evidence inventory mismatch")
    checked = verdict(par, legs, target_per_side, par_games)
    if (report["legs"] != legs or report["combined_h2h"] != leg_stats(combined)
            or any(report.get(k) != v for k, v in checked.items())
            or report["combined_par_comparisons"] != compare(leg_stats(combined)["sampled"], par)):
        raise ValueError("sampled report score/verdict arithmetic changed")
    return checked["verdict"]


@exclusive_workers
def run_gate(args):
    expected = {"model": model_identity(args.model), "bar": model_identity(args.bar_model),
                "protocol": protocol(args), "schedule": schedule(args)}
    if args.resume:
        directory = Path(args.resume).resolve()
        manifest = load_json(directory / "manifest.json")
        if any(manifest[k] != v for k, v in expected.items()):
            raise ValueError("sampled-gate resume provenance/configuration mismatch")
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        directory = Path(args.run_dir or ROOT / "benchmarks" / "sampled_gate" / stamp).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        manifest = dict(expected, run_dir=str(directory),
                        created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        atomic_json(directory / "manifest.json", manifest)
    report_path = directory / "report.json"
    destination = Path(args.report_path).resolve() if args.report_path else None
    if destination and destination.exists() and load_json(destination).get("run_dir") != str(directory):
        raise FileExistsError(f"refusing to replace another run's report: {destination}")
    if report_path.exists() and load_json(report_path).get("complete"):
        report = load_json(report_path)
        validate_report(report, args.model, args.bar_model, args.sims,
                        args.target_per_side, args.par_games)
        if destination:
            atomic_json(destination, report)
        return report
    state_path = directory / "progress.json"
    state = load_json(state_path) if state_path.exists() else {"active_seconds": 0, "completed": []}
    previous_seconds, started = state["active_seconds"], time.monotonic()
    report = dict(manifest, instrument=SCORING_VERSION, legs={}, complete=False,
                  verdict="INCONCLUSIVE", eligible=False, confirmed=False)
    par, first, combined = self_par([]), [], []

    def save_progress():
        state["active_seconds"] = previous_seconds + time.monotonic() - started
        atomic_json(state_path, state)

    try:
        for item in manifest["schedule"]:
            name = item["name"]
            log = directory / (name + ".jsonl")
            # A scheduled/partial journal must finish, even after a soft deadline.
            scheduled = log.with_suffix(".jsonl.manifest.json").exists()
            if not scheduled and previous_seconds + time.monotonic() - started >= args.budget_min * 60:
                report["stop_reason"] = "budget_exhausted_before_next_fixed_leg"
                break
            print(f"SAMPLED GATE {name}: {item['games']} fixed games", flush=True)
            run_match(args.bar_model if name == "par" else args.model, args.bar_model,
                      games=item["games"], sims=args.sims, sims_b=args.sims,
                      workers=args.workers, engine="native", seed=item["seed"],
                      opening_temp_plies=16, game_log=str(log), resume=True)
            rows = read_rows(log)
            stats = leg_stats(rows, {opening_key(r) for r in first})
            if name == "par":
                par = self_par(rows)
                report["bar_free_par"] = {"calibration": par, "diagnostics": stats}
            else:
                report["legs"][name] = stats
                combined.extend(rows)
                if name == "vs_bar":
                    first = rows
            if name not in state["completed"]:
                state["completed"].append(name)
            save_progress()
            report.update(verdict(par, report["legs"], args.target_per_side, args.par_games))
            atomic_json(report_path, report)
        report.update(verdict(par, report["legs"], args.target_per_side, args.par_games))
        report["combined_h2h"] = leg_stats(combined)
        report["combined_par_comparisons"] = compare(report["combined_h2h"]["sampled"], par)
        report["complete"] = True
        # Pin both data and task manifests. They distinguish independent seed
        # draws from accidentally rerunning the identical task schedule.
        evidence = [directory / "manifest.json", *directory.glob("*.jsonl"),
                    *directory.glob("*.jsonl.manifest.json")]
        report["evidence_hashes"] = {str(p): file_hash(p) for p in evidence}
        atomic_json(report_path, report)
        validate_report(report, args.model, args.bar_model, args.sims,
                        args.target_per_side, args.par_games)
        if destination:
            atomic_json(destination, report)
        print(f"VERDICT: {report['verdict']}; report {report_path}", flush=True)
        return report
    finally:
        save_progress()


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bar-model", required=True)
    ap.add_argument("--sims", type=int, default=3200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target-per-side", type=int, default=200)
    ap.add_argument("--par-games", type=int, default=400)
    ap.add_argument("--budget-min", type=float, default=180)
    ap.add_argument("--seed", type=int, default=40000000)
    ap.add_argument("--report-path")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--run-dir")
    group.add_argument("--resume")
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if min(args.sims, args.workers, args.target_per_side, args.par_games, args.budget_min) <= 0:
        ap.error("counts, simulations and budget must be positive")
    if args.workers > 8 or args.par_games % 2 or max(args.par_games, 2 * args.target_per_side) > 2000:
        ap.error("at most eight workers; even par games; each leg at most 2000 games")
    run_gate(args)


if __name__ == "__main__":
    main()
