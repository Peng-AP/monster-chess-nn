"""Stage 1 of docs/plans/GEN51_STRENGTH_PLAN.md: re-check v28's search constants.

Same network on both sides; only the candidate's c_puct or FPU reduction moves,
one factor at a time. Fixed chain, no score-based stopping or extensions:

  1. v28 self-par at 3,200 and 12,800 (gate v4 --par-only), measured once.
  2. Screen: four arms x 400 games against the defaults at 3,200.
  3. Nominate at most one arm by the rule fixed below.
  4. Confirm the nominee with the full gate v4 (two fresh 400-game legs at
     3,200 plus the 12,800 guard), reusing step 1's par.

A confirmed nominee is NOT applied automatically: changing `config.py` changes
the runtime identity of every later measurement, so it is an explicit
follow-up step. A full rehearsal at tiny depth runs first, in its own namespace.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from config import C_PUCT, FPU_REDUCTION
from free_gate_stats import leg_stats
from match_evidence import atomic_json, file_hash, read_rows, runtime_identity
from sampled_gate_stats import compare
from start_gpu48 import Campaign, campaign_lock, pin_json, read
import gate_depth

RUN = ROOT / "benchmarks/gen51_program/search_constants_20260925"
MODEL = "models/bootstrap_v28/best_value_net.pt"
MODEL_SHA256 = "b651e7405afe4e5672676c6fb13bb5bc6e1f5ea0071fd5ce4f4d5318e229e35a"
DEFAULT = {"c_puct": 1.5, "fpu_reduction": 0.30}
ARMS = {  # one factor each, plan section 4 Stage 1
    "A_cpuct_1.0": {"c_puct": 1.0, "fpu_reduction": 0.30},
    "B_cpuct_2.0": {"c_puct": 2.0, "fpu_reduction": 0.30},
    "C_fpu_0.20": {"c_puct": 1.5, "fpu_reduction": 0.20},
    "D_fpu_0.40": {"c_puct": 1.5, "fpu_reduction": 0.40},
}
NOMINATION_MIN = 0.525
NOMINATION_SIDE_BAND = 0.05
PINNED = ["tools/gate_depth.py", "tools/search_constants_campaign.py", "tools/match.py",
          "tools/free_gate_stats.py", "tools/sampled_gate_stats.py", "tools/free_play_audit.py",
          "tools/start_gpu48.py", "tests/test_gate_depth.py",
          "tests/test_search_constants_campaign.py", "docs/plans/GEN51_STRENGTH_PLAN.md"]


def layout(smoke):
    base = 2_960_000_000 if smoke else 2_900_000_000
    return dict(sims=8 if smoke else 3200, screen_games=4 if smoke else 400,
                par_seed=base, screen_seed=base + 1_000_000, confirm_seed=base + 5_000_000)


def identity():
    if file_hash(ROOT / MODEL) != MODEL_SHA256:
        raise ValueError("v28 checkpoint changed")
    if {"c_puct": C_PUCT, "fpu_reduction": FPU_REDUCTION} != DEFAULT:
        raise ValueError("engine default search constants are not the ones this campaign tests against")
    return dict(runtime=runtime_identity(), model={MODEL: MODEL_SHA256}, default=DEFAULT, arms=ARMS,
                implementations={p: file_hash(ROOT / p) for p in PINNED},
                protocol=layout(False), rehearsal=layout(True),
                nomination=dict(min_score=NOMINATION_MIN, side_band=NOMINATION_SIDE_BAND))


def gate_outputs(directory, legs):
    out = [directory / "report.json", directory / "manifest.json"]
    for name in legs:
        out += [directory / f"{name}.jsonl", directory / f"{name}.jsonl.manifest.json"]
    return out


def screen_leg(arm_index, spec):
    return dict(name=list(ARMS)[arm_index], role="h2h", games=spec["screen_games"], sims=spec["sims"],
                seed=spec["screen_seed"] + arm_index * 1_000_000)


def nominate(results):
    """At most one arm: overall >= 52.5% and each colour >= default self-par - 5 pp."""
    eligible = [name for name, r in results.items()
                if r["score"] >= NOMINATION_MIN - 1e-12
                and all(r["par_comparisons"][c]["delta_from_par"] >= -NOMINATION_SIDE_BAND - 1e-12
                        for c in ("white", "black"))]
    if not eligible:
        return None, eligible
    worst = lambda n: min(results[n]["par_comparisons"][c]["delta_from_par"] for c in ("white", "black"))
    order = list(results)
    return max(eligible, key=lambda n: (results[n]["score"], worst(n), -order.index(n))), eligible


def work(campaign, smoke):
    spec, root = layout(smoke), campaign.root
    rehearsal = ["--rehearsal"] if smoke else []
    par_dir = root / "par"
    campaign.stage("par", ["tools/gate_depth.py", "--par-only", "--bar-model", MODEL,
                           "--seed", str(spec["par_seed"]), "--run-dir", str(par_dir), *rehearsal],
                   gate_outputs(par_dir, gate_depth.PAR_LEGS))
    par_report = gate_depth.validate_report(par_dir / "report.json")
    par_rows = {n: read_rows(par_dir / f"{n}.jsonl") for n in gate_depth.PAR_LEGS}
    from sampled_gate_stats import self_par
    par = self_par(par_rows["par"])

    results, screen_dir = {}, root / "screen"
    for i, (name, search) in enumerate(ARMS.items()):
        leg = screen_leg(i, spec)
        log = screen_dir / f"{name}.jsonl"
        report = screen_dir / f"{name}.json"
        campaign.stage(f"screen_{name}", ["tools/match.py", "--model-a", MODEL, "--model-b", MODEL,
            "--games", str(leg["games"]), "--sims", str(leg["sims"]), "--sims-b", str(leg["sims"]),
            "--engine", "native", "--workers", str(gate_depth.WORKERS), "--seed", str(leg["seed"]),
            "--opening-temp-plies", "16", "--c-puct-a", str(search["c_puct"]),
            "--fpu-reduction-a", str(search["fpu_reduction"]), "--game-log", str(log),
            "--report-path", str(report), "--resume"], [report, log, log.with_suffix(".jsonl.manifest.json")])
        rows = gate_depth.audit_leg(screen_dir, leg, MODEL, MODEL, search)
        stats = leg_stats(rows)
        results[name] = dict(search=search, score=stats["sampled"]["score"], stats=stats,
                             par_comparisons=compare(stats["sampled"], par),
                             evidence={str(p): file_hash(p) for p in (log, log.with_suffix(".jsonl.manifest.json"))})
    chosen, eligible = nominate(results)
    pin_json(root / "nominee.json", dict(nominee=chosen, eligible=eligible,
             search=ARMS.get(chosen), rule=identity()["nomination"],
             scores={n: r["score"] for n, r in results.items()}))
    print(f"SEARCH NOMINEE {chosen}", flush=True)

    confirmation = None
    if chosen:
        conf_dir = root / "confirm"
        search = ARMS[chosen]
        campaign.stage("confirm", ["tools/gate_depth.py", "--model", MODEL, "--bar-model", MODEL,
            "--c-puct", str(search["c_puct"]), "--fpu-reduction", str(search["fpu_reduction"]),
            "--seed", str(spec["confirm_seed"]), "--run-dir", str(conf_dir), "--par-dir", str(par_dir),
            *rehearsal], gate_outputs(conf_dir, ("vs_bar", "vs_bar_confirm", "deep_guard")))
        report = gate_depth.validate_report(conf_dir / "report.json")
        confirmation = {k: report[k] for k in ("verdict", "primary", "guard", "legs", "combined_h2h",
                                                 "combined_par_comparisons", "search")}
    if identity() != campaign.provenance:
        raise ValueError("inputs changed before publication")
    pin_json(root / "summary.json", dict(
        complete=True, rehearsal=smoke, default=DEFAULT, par=par_report["protocol"]["counts"],
        par_stats={"par": par, "deep_par": self_par(par_rows["deep_par"])},
        screen={n: {k: r[k] for k in ("search", "score", "par_comparisons", "evidence")} | {
                    "sides": r["stats"]["sampled"]["sides"], "endings": r["stats"]["endings"],
                    "unique_score": r["stats"]["unique"]["score"]} for n, r in results.items()},
        nominee=chosen, eligible=eligible, confirmation=confirmation,
        manifest_sha256=file_hash(root / "manifest.json"),
        notes=["Same v28 network on both sides; only the candidate's search constants differ.",
               "A confirmed nominee is not applied to config.py automatically.",
               "Screen results nominate; only the fresh confirmation legs are evidence of a change."]))
    atomic_json(root / "status.json", dict(status="complete"))
    print("SEARCH CONSTANTS CAMPAIGN COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rehearsal-only", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    os.environ["MONSTER_PINNED_INPUT"] = "1"
    provenance = identity()
    with campaign_lock(RUN / "campaign.lock"):
        for smoke in (True, False):
            if not smoke and args.rehearsal_only:
                break
            campaign = Campaign(RUN / ("rehearsal" if smoke else "production"), provenance,
                                identity_fn=identity, label="SEARCH")
            try:
                if smoke:
                    campaign.stage("tests", ["-m", "pytest", "-p", "no:cacheprovider", "tests/test_gate_depth.py",
                                             "tests/test_search_constants_campaign.py", "-q"])
                elif not read(RUN / "rehearsal/summary.json")["complete"]:
                    raise ValueError("rehearsal required")
                work(campaign, smoke)
            except BaseException as exc:
                atomic_json(campaign.root / "status.json", dict(status="failed", error=str(exc)))
                raise


if __name__ == "__main__":
    main()
