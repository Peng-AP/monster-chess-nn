"""Gate v4: the v3 sampled 3,200-sim gate plus a binding 12,800-sim guard.

Declared in docs/plans/GEN51_STRENGTH_PLAN.md section 1 (owner-approved
2026-09-25). The v3 tool and its reports are unchanged; this is a separate,
separately versioned instrument. Why the guard exists: gen50 epoch15 passed
the 3,200 gate while its White scored 26.25% at 12,800.

A candidate is a checkpoint plus its search constants (c_puct, FPU
reduction), so the same instrument also judges search-setting changes. The
bar always plays with the engine defaults from `config.py`.

Legs, all normal-start, temperature 0.5 for 16 plies, captures-only scoring:

    par         bar self-play at 3,200      (v3 calibration)
    vs_bar      candidate vs bar at 3,200   (v3 first leg)
    vs_bar_confirm  fresh RNG block at 3,200 (v3 confirmation)
    deep_par    bar self-play at 12,800     (guard calibration)
    deep_guard  candidate vs bar at 12,800  (guard)

The two par legs may come from a completed `--par-only` run of the same bar,
runtime and implementation (`--par-dir`), so one campaign measures its
incumbent once. Every leg runs to its fixed count regardless of earlier
results; no score-based stopping, no extensions.
"""
import argparse
import inspect
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from config import C_PUCT, FPU_REDUCTION
from free_gate_stats import leg_stats, opening_key
from free_play_audit import audit_game
from match import build_tasks, run_match
from match_evidence import atomic_json, file_hash, model_identity, read_rows, runtime_identity, task_id
from sampled_gate_stats import compare, self_par, verdict as primary_verdict
from worker_lease import exclusive_workers

VERSION = "free_sampled_depth_guard_v4"
PRIMARY_SIMS, GUARD_SIMS = 3200, 12800
PRODUCTION_COUNTS = dict(par=400, per_side=200, deep_par=160, deep=160)
REHEARSAL_SIMS = (8, 16)
REHEARSAL_COUNTS = dict(par=4, per_side=2, deep_par=4, deep=4)
# Guard thresholds (plan section 1). Constants, never CLI arguments.
GUARD_AGGREGATE_MIN = 0.475
GUARD_SIDE_BAND = 0.10
WORKERS = 8
IMPLEMENTATION = ("gate_depth.py", "sampled_gate_stats.py", "free_gate_stats.py",
                  "free_play_audit.py", "match.py")
MATCH_SIGNATURE = inspect.signature(run_match)
PAR_LEGS = ("par", "deep_par")


def default_search():
    return {"c_puct": float(C_PUCT), "fpu_reduction": float(FPU_REDUCTION)}


def search_kwargs(search, side):
    return {f"c_puct_{side}": search["c_puct"], f"fpu_reduction_{side}": search["fpu_reduction"]}


def protocol(rehearsal, seed):
    sims = REHEARSAL_SIMS if rehearsal else (PRIMARY_SIMS, GUARD_SIMS)
    return {"version": VERSION, "runtime": runtime_identity(),
            "implementation": {n: file_hash(ROOT / "tools" / n) for n in IMPLEMENTATION},
            "engine": "native", "workers": WORKERS, "opening_temp_plies": 16,
            "opening_temperature": .5, "primary_sims": sims[0], "guard_sims": sims[1],
            "counts": dict(REHEARSAL_COUNTS if rehearsal else PRODUCTION_COUNTS),
            "guard_aggregate_min": GUARD_AGGREGATE_MIN, "guard_side_band": GUARD_SIDE_BAND,
            "bar_search": default_search(), "seed": seed, "binding": not rehearsal}


def schedule(proto, par_only=False, external_par=False):
    counts, seed = proto["counts"], proto["seed"]
    legs = [dict(name="par", role="par", games=counts["par"], sims=proto["primary_sims"], seed=seed + 100000),
            dict(name="vs_bar", role="h2h", games=2 * counts["per_side"], sims=proto["primary_sims"], seed=seed),
            dict(name="vs_bar_confirm", role="h2h", games=2 * counts["per_side"],
                 sims=proto["primary_sims"], seed=seed + 200000),
            dict(name="deep_par", role="par", games=counts["deep_par"], sims=proto["guard_sims"],
                 seed=seed + 300000),
            dict(name="deep_guard", role="h2h", games=counts["deep"], sims=proto["guard_sims"],
                 seed=seed + 400000)]
    if par_only:
        return [leg for leg in legs if leg["name"] in PAR_LEGS]
    if external_par:
        return [leg for leg in legs if leg["name"] not in PAR_LEGS]
    return legs


def leg_settings(model_a, model_b, leg, search_a):
    """The settings `run_match` pins in each journal manifest, recomputed."""
    bound = MATCH_SIGNATURE.bind(model_a, model_b, games=leg["games"], sims=leg["sims"],
                                 sims_b=leg["sims"], workers=WORKERS, engine="native",
                                 seed=leg["seed"], opening_temp_plies=16,
                                 **search_kwargs(search_a, "a"), **search_kwargs(default_search(), "b"))
    bound.apply_defaults()
    settings = dict(bound.arguments)
    for key in ("game_log", "checkpoint_path", "resume", "stall_timeout"):
        settings.pop(key)
    settings.update(model_a=model_identity(model_a), model_b=model_identity(model_b),
                    runtime=runtime_identity())
    return settings


def leg_players(leg, model, bar, search):
    return (bar, default_search()) if leg["role"] == "par" else (model, search)


def audit_leg(directory, leg, model, bar, search):
    """Journal settings, task inventory and a full legal replay of every game."""
    log = Path(directory) / (leg["name"] + ".jsonl")
    meta = log.with_suffix(".jsonl.manifest.json")
    a, search_a = leg_players(leg, model, bar, search)
    tasks = {task_id(t): t for t in build_tasks(leg["games"], leg["seed"], 16)}
    import json
    if json.loads(meta.read_text()) != {"schema_version": 1, "tasks": list(tasks),
                                        "settings": leg_settings(a, bar, leg, search_a)}:
        raise ValueError(f"{leg['name']}: journal settings/tasks mismatch")
    rows = read_rows(log)
    if len(rows) != len(tasks) or {r["task_id"] for r in rows} != set(tasks):
        raise ValueError(f"{leg['name']}: tasks missing or duplicated")
    for row in rows:
        task = tasks[row["task_id"]]
        if (row["a_is_white"], row["seed"], row["pair"]) != (task[0], task[1], task[4]):
            raise ValueError(f"{leg['name']}: row task metadata changed")
        audit_game(row)
    return rows


def guard_verdict(deep_par, deep_stats, counts):
    missing, failures = [], []
    if deep_par is None or deep_par["n"] != counts["deep_par"]:
        missing.append("deep_par does not have the declared game count")
    if deep_stats is None:
        missing.append("deep_guard is missing")
    elif any(deep_stats["sides"][s]["n"] != counts["deep"] // 2 for s in ("white", "black")):
        missing.append("deep_guard does not have the declared per-color count")
    comparisons = {}
    if not missing:
        comparisons = compare(deep_stats, deep_par)
        if deep_stats["score"] < GUARD_AGGREGATE_MIN - 1e-12:
            failures.append(f"deep_guard aggregate < {GUARD_AGGREGATE_MIN}")
        for color, c in comparisons.items():
            if c["delta_from_par"] < -GUARD_SIDE_BAND - 1e-12:
                failures.append(f"deep_guard {color} below deep par - {GUARD_SIDE_BAND}")
    status = "INCONCLUSIVE" if missing else "FAIL" if failures else "PASS"
    return {"verdict": status, "inconclusive_reasons": missing, "failures": failures,
            "par_comparisons": comparisons}


def overall_verdict(primary, guard):
    if "INCONCLUSIVE" in (primary["verdict"], guard["verdict"]):
        return "INCONCLUSIVE"
    return "PASS" if primary["verdict"] == guard["verdict"] == "PASS" else "FAIL"


def score(rows_by_leg, proto, par_rows):
    """Every verdict number, from rows alone. Shared by run and validation."""
    counts = proto["counts"]
    par = self_par(par_rows["par"])
    deep_par = self_par(par_rows["deep_par"])
    first = {opening_key(r) for r in rows_by_leg.get("vs_bar", [])}
    legs = {n: leg_stats(rows_by_leg[n], first if n != "vs_bar" else set())
            for n in ("vs_bar", "vs_bar_confirm", "deep_guard") if n in rows_by_leg}
    primary = primary_verdict(par, {n: legs[n] for n in ("vs_bar", "vs_bar_confirm") if n in legs},
                              counts["per_side"], counts["par"])
    guard = guard_verdict(deep_par, legs.get("deep_guard", {}).get("sampled"), counts)
    combined = rows_by_leg.get("vs_bar", []) + rows_by_leg.get("vs_bar_confirm", [])
    out = {"par": par, "deep_par": deep_par, "legs": legs, "primary": primary, "guard": guard,
           "verdict": overall_verdict(primary, guard)}
    if combined:
        out["combined_h2h"] = leg_stats(combined)
        out["combined_par_comparisons"] = compare(out["combined_h2h"]["sampled"], par)
    return out


def load_par_source(par_dir, bar, proto):
    """Rows of a completed par-only run, after checking it measured this bar the same way."""
    par_dir = Path(par_dir).resolve()
    report = validate_report(par_dir / "report.json")
    theirs = report["protocol"]
    same = all(theirs[k] == proto[k] for k in ("runtime", "implementation", "primary_sims", "guard_sims",
                                                "counts", "bar_search", "binding"))
    if report["mode"] != "par_only" or report["bar"] != model_identity(bar) or not same:
        raise ValueError("par source measured a different bar, runtime, depth or count")
    return {leg["name"]: read_rows(par_dir / (leg["name"] + ".jsonl")) for leg in report["schedule"]}, \
        {"dir": str(par_dir), "report_sha256": file_hash(par_dir / "report.json")}


def validate_report(path):
    """Recompute a finished report from its immutable journals; raise on any drift."""
    import json
    path = Path(path)
    report = json.loads(path.read_text())
    directory = Path(report["run_dir"])
    if report.get("instrument") != VERSION or not report.get("complete"):
        raise ValueError("not a complete v4 report")
    proto = report["protocol"]
    if proto != protocol(not proto["binding"], proto["seed"]):
        raise ValueError("runtime, implementation or protocol constants changed since this run")
    for p, h in report["evidence_hashes"].items():
        if not Path(p).is_file() or file_hash(p) != h:
            raise ValueError(f"evidence missing or modified: {p}")
    model = report["model"]["path"] if report["model"] else None
    bar = report["bar"]["path"]
    if model_identity(bar) != report["bar"] or (model and model_identity(model) != report["model"]):
        raise ValueError("checkpoint identity changed")
    rows = {leg["name"]: audit_leg(directory, leg, model, bar, report["search"]) for leg in report["schedule"]}
    if report["mode"] == "par_only":
        return report
    source = report["par_source"]
    if source["dir"] == str(directory):
        par_rows = {n: rows[n] for n in PAR_LEGS}
    else:
        par_rows, recorded = load_par_source(source["dir"], bar, report["protocol"])
        if recorded != source:
            raise ValueError("par source changed since the report was written")
    recomputed = score(rows, report["protocol"], par_rows)
    if any(report[k] != v for k, v in recomputed.items()):
        raise ValueError("report arithmetic does not match its journals")
    return report


@exclusive_workers
def run(args):
    directory = Path(args.run_dir).resolve()
    proto = protocol(args.rehearsal, args.seed)
    search = {"c_puct": args.c_puct, "fpu_reduction": args.fpu_reduction}
    plan = schedule(proto, par_only=args.par_only, external_par=bool(args.par_dir))
    head = dict(instrument=VERSION, mode="par_only" if args.par_only else "gate",
                model=None if args.par_only else model_identity(args.model),
                search=default_search() if args.par_only else search,
                bar=model_identity(args.bar_model), protocol=proto, schedule=plan, run_dir=str(directory))
    manifest = directory / "manifest.json"
    if manifest.exists():
        import json
        saved = json.loads(manifest.read_text())
        if {k: saved.get(k) for k in head} != head:
            raise ValueError("run directory holds a different gate configuration")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(manifest, dict(head, created=time.strftime("%Y-%m-%dT%H:%M:%S")))
    report_path = directory / "report.json"
    if report_path.exists():
        return validate_report(report_path)
    par_rows, source = None, {"dir": str(directory)}
    if args.par_dir:
        par_rows, source = load_par_source(args.par_dir, args.bar_model, proto)
    rows = {}
    for leg in plan:
        print(f"GATE V4 {leg['name']}: {leg['games']} games at {leg['sims']} sims", flush=True)
        a, search_a = leg_players(leg, args.model, args.bar_model, search)
        run_match(a, args.bar_model, games=leg["games"], sims=leg["sims"], sims_b=leg["sims"],
                  workers=WORKERS, engine="native", seed=leg["seed"], opening_temp_plies=16,
                  game_log=str(directory / (leg["name"] + ".jsonl")), resume=True,
                  **search_kwargs(search_a, "a"), **search_kwargs(default_search(), "b"))
        rows[leg["name"]] = audit_leg(directory, leg, args.model, args.bar_model, search)
    report = dict(head, complete=True)
    if not args.par_only:
        report["par_source"] = source
        report.update(score(rows, proto, par_rows or {n: rows[n] for n in PAR_LEGS}))
    evidence = [manifest, *directory.glob("*.jsonl"), *directory.glob("*.jsonl.manifest.json")]
    report["evidence_hashes"] = {str(p): file_hash(p) for p in evidence}
    atomic_json(report_path, report)
    validate_report(report_path)
    print(f"GATE V4 {report.get('verdict', 'PAR COMPLETE')}: {report_path}", flush=True)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="candidate checkpoint (omit with --par-only)")
    ap.add_argument("--bar-model", required=True)
    ap.add_argument("--c-puct", type=float, default=C_PUCT, help="candidate's c_puct")
    ap.add_argument("--fpu-reduction", type=float, default=FPU_REDUCTION, help="candidate's FPU reduction")
    ap.add_argument("--seed", type=int, required=True, help="reserve one million seeds per run")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--par-only", action="store_true", help="measure the bar's two self-par legs only")
    ap.add_argument("--par-dir", help="completed --par-only run of the same bar to reuse")
    ap.add_argument("--rehearsal", action="store_true", help="tiny non-binding counts and depths")
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if args.par_only == bool(args.model):
        ap.error("give --model for a gate, or --par-only without --model")
    if args.par_only and args.par_dir:
        ap.error("--par-dir reuses par; it cannot be combined with --par-only")
    run(args)


if __name__ == "__main__":
    main()
