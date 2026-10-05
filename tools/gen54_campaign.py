"""Gen54: gen53 teacher plus a teacher mix and a hole-scan opponent pool, one arm.

docs/plans/GEN54_PLAN.md (owner, 2026-10-05: "a mix of teachers as well as the
other proposed changes. 1 arm only"). Derived from tools/gen53_campaign.py:

 1. Teacher par (gen53).
 2. Canonical gen54 iteration through `compose` (gen53 self-play + forks +
    reanalysis, 30-ply exploration).
 3. Deep-value continuations by gen53, relabelled to the ramped game results.
 4. Extra teachers: self-play games by gen52 Arm R and Arm LR, each its own
    increment (policy and value, ramped labels).
 5. Pool: 1,204 gen53-vs-pool games (only gen53's moves carry policy weight);
    opponents chosen by the October 5 hole scan; B2 and v27 held out.
 6. One replay: canonical sources + both deep increments + extra teachers + pool;
    gen51's exact train command.
 7. Selection vs gen53, gate v4 vs gen53, diagnostics (B2, v27, gen49, v28,
    self-play). The v27-position probe and the gate vs the release run as a
    follow-up chain, as for gen53.

Nothing promoted.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import iterate
from match_evidence import atomic_json, file_hash, read_rows, runtime_identity
from sampled_gate_stats import compare, self_par
from start_gpu48 import Campaign, campaign_lock, pin_json, read
import gate_depth
import gen51_campaign as g51
import start_gen49 as prior

RUN = ROOT / "benchmarks/gen54_program/gen54_20261005"
SMOKE_ROOT = ROOT / "iterations/rehearsal_gen54_20261005"
DECISION = ROOT / "docs/plans/gen54_teacher_decision.json"
V28, GEN49 = g51.V28, g51.GEN49
HELD_OUT = [prior.HOLDOUT, prior.RELEASE]  # B2 and v27: never training opponents or teachers
PINNED = ["tools/gen54_campaign.py", "tools/gen51_campaign.py", "tools/disagreement_continuations.py",
          "tools/process_linked_extra.py", "tools/process_linked_extra_ramped.py", "tools/stateful_generation.py",
          "tools/iterate_stateful.py", "tools/reanalyze_coverage.py", "tools/reanalyze_stateful.py",
          "tools/reanalyze.py", "tools/reanalysis_journal.py", "tools/process_families.py",
          "tools/compose_processed.py", "tools/checkpoint_screen.py", "tools/audit_generation_data.py",
          "tools/gate_depth.py", "tools/match.py", "tools/free_gate_stats.py", "tools/sampled_gate_stats.py",
          "tools/free_play_audit.py", "tools/start_gpu48.py", "tools/start_gen49.py", "tools/gate_sampled.py",
          "tests/test_gen54_campaign.py", "tests/test_gate_depth.py",
          "docs/plans/GEN54_PLAN.md", "docs/plans/gen54_teacher_decision.json",
          *[f"tools/recipes/gen54{s}.json" for s in ("", "_rehearsal", "_pool", "_pool_rehearsal", "_teacher_r",
                                                     "_teacher_r_rehearsal", "_teacher_lr", "_teacher_lr_rehearsal")]]


def decision():
    d = read(DECISION)
    for model, sha in [(d["teacher"], d["teacher_sha256"])] + [(t["model"], t["sha256"]) for t in d["extra_teachers"]]:
        if file_hash(ROOT / model) != sha:
            raise ValueError(f"teacher checkpoint changed: {model}")
    return d


def layout(smoke):
    base = 4_255_000_000 if smoke else 4_220_000_000
    return dict(
        generation=1 if smoke else 54, run_root=SMOKE_ROOT if smoke else ROOT / "iterations",
        sims=8 if smoke else 3200, deep_sims=16 if smoke else 12800,
        roots=8 if smoke else 768, continuations=2, continuation_sims=8 if smoke else 6400,
        screen_games=4 if smoke else 200, probe_games=4 if smoke else 40, finalists=1 if smoke else 2,
        deep_probe_games=4 if smoke else 80, diag_games=4 if smoke else 160, self_games=4 if smoke else 200,
        par_seed=base, continuation_seed=base + 1_000_000, screen_seed=base + 2_000_000,
        deep_probe_seed=base + 3_000_000, gate_seed=base + 10_000_000, diag_seed=base + 20_000_000,
        arms_seed=base + 30_000_000)


def recipe_path(kind, smoke):
    return ROOT / "tools/recipes" / f"gen54{kind}{'_rehearsal' if smoke else ''}.json"


def identity():
    d = decision()
    models = {p: file_hash(ROOT / p) for p in [d["teacher"], *(t["model"] for t in d["extra_teachers"]),
                                                *d["pool_opponents"], *HELD_OUT]}
    held = {str(Path(p)) for p in HELD_OUT}
    for smoke in (False, True):
        for kind, model in [("", d["teacher"]), ("_pool", d["teacher"])] + \
                [(f"_teacher_{t['name']}", t["model"]) for t in d["extra_teachers"]]:
            recipe = read(recipe_path(kind, smoke))
            prior.validate_recipe(recipe)
            if recipe["model"] != model or recipe.get("temperature_plies") != 30:
                raise ValueError(f"unexpected gen54 recipe {kind} (smoke={smoke})")
            if {str(Path(p)) for p in recipe["opponents"]} & held:
                raise ValueError("held-out opponents may never be training opponents")
        if read(recipe_path("_pool", smoke))["opponents"] != d["pool_opponents"]:
            raise ValueError("pool recipe opponents differ from the decision")
    if {str(Path(t["model"])) for t in d["extra_teachers"]} & held or str(Path(d["teacher"])) in held:
        raise ValueError("held-out models may never be teachers")
    replay = sorted((e for e in read(ROOT / "iterations/accepted_data.json")["entries"]
                     if int(e["generation"]) < 54), key=lambda e: int(e["generation"]))[-7:]
    if len(replay) != 7 or int(replay[-1]["generation"]) != 53:
        raise ValueError("expected accepted replay through gen53")
    return dict(runtime=runtime_identity(), decision=d, models=models, replay=replay,
                implementations={p: file_hash(ROOT / p) for p in PINNED},
                protocol={k: str(v) for k, v in layout(False).items()},
                rehearsal={k: str(v) for k, v in layout(True).items()})


def iteration_command(smoke):
    command = g51.iteration_command(smoke)
    changes = {"--recipe": str(recipe_path("", smoke).relative_to(ROOT)).replace("\\", "/"),
               "--expected-generation": str(layout(smoke)["generation"]),
               "--incumbent": decision()["teacher"], "--through-phase": "compose"}
    if smoke:
        changes["--run-root"] = str(SMOKE_ROOT)
    for flag, value in changes.items():
        command[command.index(flag) + 1] = value
    return command


def paths_for(smoke):
    spec = layout(smoke)
    paths = iterate._paths_for_generation(spec["run_root"], spec["generation"])
    namespace = iterate._run_namespace(spec["run_root"])
    n, processed = spec["generation"], ROOT / "data/processed"
    stem = f"{namespace}_gen_{n:04d}"
    return paths, dict(
        pool_raw=paths["run_dir"] / "pool_raw", pool=processed / f"bootstrap_extra_{stem}_pool",
        deep=processed / f"bootstrap_extra_{stem}_disagreement",
        teacher=lambda name: (paths["run_dir"] / f"teacher_{name}_raw", processed / f"bootstrap_extra_{stem}_teacher_{name}"),
        a=dict(model_dir=paths["candidate_dir"], replay=processed / f"bootstrap_replay_{stem}_armA"))


def ramped_path(path):
    return Path(path).with_name(Path(path).name + "_ramped")


def teacher_par(campaign, smoke):
    par_dir = campaign.root / "par"
    campaign.stage("par", ["tools/gate_depth.py", "--par-only", "--bar-model", decision()["teacher"],
                           "--seed", str(layout(smoke)["par_seed"]), "--run-dir", str(par_dir),
                           *(["--rehearsal"] if smoke else [])], [par_dir / "report.json"])
    gate_depth.validate_report(par_dir / "report.json")
    return par_dir


def iteration(campaign, smoke):
    paths, _ = paths_for(smoke)
    command = iteration_command(smoke)
    if paths["state"].exists():
        command += ["--resume"]
    receipt = campaign.root / "receipts/iteration.json"
    if receipt.exists():
        command = read(receipt)["command"]
    campaign.stage("iteration", command, [paths["state"]])
    state = read(paths["state"])
    for name in iterate.PHASES[:iterate.PHASES.index("compose") + 1]:
        if state["phases"].get(name, {}).get("status") != "completed":
            raise ValueError(f"iteration phase incomplete: {name}")
    return state


def deep_source(campaign, smoke):
    spec, (paths, extra) = layout(smoke), paths_for(smoke)
    d = decision()
    config = dict(raw_dir=str(paths["raw"].relative_to(ROOT)).replace("\\", "/"), player=d["teacher"],
                  reference=d["reference"], roots=spec["roots"], continuations=spec["continuations"],
                  sims=spec["continuation_sims"], seed=spec["continuation_seed"], workers=8, min_ply=4, max_ply=120)
    path = campaign.root / "continuations_config.json"
    pin_json(path, config)
    out = campaign.root / "continuations"
    campaign.stage("continuations", ["tools/disagreement_continuations.py", "--config", str(path), "--out", str(out)],
                   [out / "summary.json", out / "roots.json"])
    campaign.stage("deep_increment", ["tools/process_linked_extra.py", "--raw-dir", str(out / "raw"),
        "--parent-increment", str(paths["new_processed"]), "--output-dir", str(extra["deep"]),
        "--seed", g51.iteration_seed(smoke), "--value-weight", str(g51.DEEP_VALUE_WEIGHT)],
        [extra["deep"] / "derivation.json"])
    ramped = ramped_path(extra["deep"])
    campaign.stage("deep_increment_ramped", ["tools/process_linked_extra_ramped.py",
        "--original-increment", str(extra["deep"]), "--output-dir", str(ramped)], [ramped / "derivation.json"])
    return ramped


def generated_source(campaign, smoke, label, recipe, raw, out, games):
    """Run a stateful_generation recipe and process its games as their own increment (ramped labels)."""
    summary = campaign.root / f"{label}_generation_summary.json"
    campaign.stage(f"{label}_games", ["tools/stateful_generation.py", "--config", str(recipe.relative_to(ROOT)),
                                      "--raw", str(raw), "--summary", str(summary)], [summary])
    record = read(summary)
    if record["saved_games"] != games or record["failed_games"]:
        raise ValueError(f"{label} games incomplete")
    campaign.stage(f"{label}_increment", ["tools/process_families.py", "--raw-dir", str(raw),
        "--output-dir", str(out), "--seed", g51.iteration_seed(smoke), "--channels", "15",
        "--value-floor", ".5", "--value-horizon", "60"], [out / "split_game_ids.json"])
    return out


def extra_sources(campaign, smoke):
    _, extra = paths_for(smoke)
    n = layout(smoke)["generation"]
    sources = []
    for t in decision()["extra_teachers"]:
        recipe = recipe_path(f"_teacher_{t['name']}", smoke)
        raw, out = extra["teacher"](t["name"])
        sources.append((f"gen_{n:04d}_teacher_{t['name']}",
                        generated_source(campaign, smoke, f"teacher_{t['name']}", recipe, raw, out,
                                         read(recipe)["free_games"])))
    recipe = recipe_path("_pool", smoke)
    sources.append((f"gen_{n:04d}_pool", generated_source(campaign, smoke, "pool", recipe, extra["pool_raw"],
                                                          extra["pool"], read(recipe)["league_games"])))
    return sources


def compose_and_train(campaign, smoke, state, sources):
    _, extra = paths_for(smoke)
    arm_paths = extra["a"]
    compose = list(state["phases"]["compose"]["commands"][0])
    compose[compose.index("--output-dir") + 1] = str(arm_paths["replay"])
    for name, path in sources:
        compose += ["--source", f"{name}={path}"]
    campaign.stage("compose_a", compose, [arm_paths["replay"] / "replay_manifest.json"])
    template = read(g51.arm_paths(smoke)[0]["state"])["phases"]["train"]["commands"][0]
    train = list(template)
    for flag, value in (("--data-dir", arm_paths["replay"]), ("--model-dir", arm_paths["model_dir"])):
        train[train.index(flag) + 1] = str(value)
    if arm_paths["model_dir"].exists() and not (campaign.root / "receipts/train_a.json").exists():
        raise ValueError("unreceipted training output retained; refusing automatic overwrite")
    campaign.stage("train_a", train, [arm_paths["model_dir"] / "best_value_net.pt"])
    return arm_paths["model_dir"]


def select_arm(campaign, smoke, model_dir, deep_par):
    spec, teacher = layout(smoke), decision()["teacher"]
    report_path = campaign.root / "selection" / "a_screen.json"
    campaign.stage("screen_a", ["tools/checkpoint_screen.py", "--model-dir", str(model_dir),
        "--incumbent", teacher, "--output-model", str(model_dir / "screen_pick.pt"), "--report-path", str(report_path),
        "--games", str(spec["screen_games"]), "--probe-games", str(spec["probe_games"]),
        "--sims", str(spec["sims"]), "--probe-sims", str(spec["sims"]), "--finalists", str(spec["finalists"]),
        "--workers", "8", "--engine", "native", "--seed", str(spec["screen_seed"]), "--stall-timeout", "600.0"],
        [report_path, model_dir / "screen_pick.pt"])
    candidates = g51.screen_candidates(read(report_path))
    for i, c in enumerate(candidates):
        audit = g51.match(campaign, f"deep_probe_a_{i}", c["checkpoint"], teacher, spec["deep_probe_games"],
                          spec["deep_probe_seed"] + i * 1_000_000, spec["deep_sims"], directory="selection")
        stats = audit["diagnostics"]["sampled"]
        c["deep"] = dict(score=stats["score"], sides=stats["sides"], par_comparisons=compare(stats, deep_par))
    name, eligible, rule = g51.deep_nominee(candidates)
    chosen = next(c for c in candidates if c["name"] == name)
    source = Path(chosen["checkpoint"]) if Path(chosen["checkpoint"]).is_absolute() else ROOT / chosen["checkpoint"]
    target = model_dir / "arena_selected.pt"
    if target.exists() and file_hash(target) != file_hash(source):
        raise FileExistsError(f"{target} holds a different model; refusing to overwrite")
    if not target.exists():
        shutil.copy2(source, target)
    nominee = dict(arm="a", name=name, rule=rule, eligible=eligible, candidates=candidates,
                   path=str(target), sha256=file_hash(target))
    pin_json(campaign.root / "selection" / "a_nominee.json", nominee)
    return nominee


def evaluate(campaign, smoke, nominee, par_dir):
    spec, teacher = layout(smoke), decision()["teacher"]
    gate_dir = campaign.root / "gates" / "a"
    campaign.stage("gate_a", ["tools/gate_depth.py", "--model", nominee["path"], "--bar-model", teacher,
        "--seed", str(spec["gate_seed"]), "--run-dir", str(gate_dir), "--par-dir", str(par_dir),
        *(["--rehearsal"] if smoke else [])], [gate_dir / "report.json"])
    gate = gate_depth.validate_report(gate_dir / "report.json")
    schedule = [("vs_b2", prior.HOLDOUT, spec["sims"], spec["diag_games"]),
                ("vs_v27", prior.RELEASE, spec["sims"], spec["diag_games"]),
                ("vs_gen49", GEN49, spec["sims"], spec["diag_games"]),
                ("vs_v28", V28, spec["sims"], spec["diag_games"]),
                ("self", nominee["path"], spec["sims"], spec["self_games"]),
                ("deep_self", nominee["path"], spec["deep_sims"], spec["diag_games"])]
    diagnostics = {name: g51.match(campaign, f"a_{name}", nominee["path"], other, games,
                                   spec["diag_seed"] + i * 1_000_000, sims)
                   for i, (name, other, sims, games) in enumerate(schedule)}
    return dict(verdict=gate["verdict"], primary=gate["primary"], guard=gate["guard"], legs=gate["legs"],
                combined_h2h=gate["combined_h2h"], diagnostics=diagnostics, held_out=["vs_b2", "vs_v27"])


def work(campaign, smoke):
    d = decision()
    par_dir = teacher_par(campaign, smoke)
    deep_par = self_par(read_rows(par_dir / "deep_par.jsonl"))
    state = iteration(campaign, smoke)
    base = [] if smoke else [(s["name"], ROOT / s["path"]) for s in d.get("prior_extra_sources", [])]
    base.append((f"gen_{layout(smoke)['generation']:04d}_deepvalue", deep_source(campaign, smoke)))
    model_dir = compose_and_train(campaign, smoke, state, base + extra_sources(campaign, smoke))
    nominee = select_arm(campaign, smoke, model_dir, deep_par)
    result = evaluate(campaign, smoke, nominee, par_dir)
    if identity() != campaign.provenance:
        raise ValueError("inputs changed before publication")
    pin_json(campaign.root / "summary.json", dict(
        complete=True, rehearsal=smoke, decision=d, nominee=nominee, result=result,
        manifest_sha256=file_hash(campaign.root / "manifest.json"),
        notes=["Single arm: canonical gen53-teacher data + Arm R and Arm LR self-play + hole-scan pool.",
               "Nothing promoted.", "B2 and v27 are held out (never teachers or training opponents)."]))
    atomic_json(campaign.root / "status.json", dict(status="complete"))
    print(f"GEN54 {'REHEARSAL' if smoke else 'PRODUCTION'} COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rehearsal-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    os.environ["MONSTER_PINNED_INPUT"] = "1"
    provenance = identity()
    with campaign_lock(RUN / "campaign.lock"):
        for smoke in (True, False):
            if not smoke and args.rehearsal_only:
                break
            campaign = Campaign(RUN / ("rehearsal" if smoke else "production"), provenance,
                                identity_fn=identity, label="GEN54")
            try:
                if smoke:
                    campaign.stage("tests", ["-m", "pytest", "-p", "no:cacheprovider", "tests/test_gen54_campaign.py",
                                             "tests/test_gate_depth.py", "tests/test_stateful_recipe.py", "-q"])
                elif not read(RUN / "rehearsal/summary.json")["complete"]:
                    raise ValueError("rehearsal required")
                work(campaign, smoke)
            except BaseException as exc:
                atomic_json(campaign.root / "status.json", dict(status="failed", error=str(exc)))
                raise


if __name__ == "__main__":
    main()
