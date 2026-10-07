"""Gen55: gen54 teacher plus gen53, Arm R and Arm LR; a White check against v29 in selection; one arm.

docs/plans/GEN55_PLAN.md (owner, 2026-10-07: "Should add one more teacher to
the pool. Then start 55"). Derived from tools/gen54_campaign.py; the changes:

 1. Teacher gen54. Extra teachers: gen53 (its White holds against v29), gen52
    Arm R and Arm LR, 700 self-play games each.
 2. Pool: gen54's hole-scan pool plus gen53, 86 games per opponent per colour;
    only gen54's moves carry policy weight. v29 stays in the pool.
 3. v29's par is measured once, up front, with this campaign's runtime.
 4. Selection: among epochs that pass the deep guard against gen54, keep those
    whose White against v29 at 3,200 is within 5 pp of v29's own White par;
    best screen rank among them; if none, best on that White check.
 5. Evaluation in one chain: gate v4 against gen54 (teacher) AND against v29
    (release, reusing the par from 3), diagnostics (B2, v27, gen49, v28,
    self-play), the v27-position probe and the value audit. The summary
    states promotion eligibility under docs/protocols/PROMOTION_RULE.md.

The engine is repetition-aware (2026-10-07); every par here is measured under
it. Nothing promoted. B2 and v27 are held out.
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

RUN = ROOT / "benchmarks/gen55_program/gen55_20261007"
SMOKE_ROOT = ROOT / "iterations/rehearsal_gen55_20261007"
DECISION = ROOT / "docs/plans/gen55_teacher_decision.json"
V28, GEN49 = g51.V28, g51.GEN49
HELD_OUT = [prior.HOLDOUT, prior.RELEASE]  # B2 and v27: never training opponents or teachers
# PROMOTION_RULE.md: held-out mean no worse than the release's (v29: 92.7%) minus 1 pp.
HELD_OUT_MIN = 0.917
RECIPE_KINDS = ("", "_pool", "_teacher_gen53", "_teacher_r", "_teacher_lr")
PINNED = ["tools/gen55_campaign.py", "tools/gen51_campaign.py", "tools/disagreement_continuations.py",
          "tools/process_linked_extra.py", "tools/process_linked_extra_ramped.py", "tools/stateful_generation.py",
          "tools/iterate_stateful.py", "tools/reanalyze_coverage.py", "tools/reanalyze_stateful.py",
          "tools/reanalyze.py", "tools/reanalysis_journal.py", "tools/process_families.py",
          "tools/compose_processed.py", "tools/checkpoint_screen.py", "tools/audit_generation_data.py",
          "tools/gate_depth.py", "tools/match.py", "tools/free_gate_stats.py", "tools/sampled_gate_stats.py",
          "tools/free_play_audit.py", "tools/start_gpu48.py", "tools/start_gen49.py", "tools/gate_sampled.py",
          "tools/position_probe.py", "tools/value_colour_audit.py",
          "tests/test_gen55_campaign.py", "tests/test_gate_depth.py",
          "docs/plans/GEN55_PLAN.md", "docs/plans/gen55_teacher_decision.json",
          *[f"tools/recipes/gen55{k}{s}.json" for k in RECIPE_KINDS for s in ("", "_rehearsal")]]


def decision():
    d = read(DECISION)
    checks = [(d["teacher"], d["teacher_sha256"]), (d["release"], d["release_sha256"])]
    for model, sha in checks + [(t["model"], t["sha256"]) for t in d["extra_teachers"]]:
        if file_hash(ROOT / model) != sha:
            raise ValueError(f"checkpoint changed: {model}")
    return d


def layout(smoke):
    # Clear of every earlier block: the Elo round robin ends at 3.52e9, the ladder starts at 3.60e9.
    base = 3_565_000_000 if smoke else 3_530_000_000
    return dict(
        generation=1 if smoke else 55, run_root=SMOKE_ROOT if smoke else ROOT / "iterations",
        sims=8 if smoke else 3200, deep_sims=16 if smoke else 12800,
        roots=8 if smoke else 768, continuations=2, continuation_sims=8 if smoke else 6400,
        screen_games=4 if smoke else 200, probe_games=4 if smoke else 40, finalists=1 if smoke else 2,
        deep_probe_games=4 if smoke else 80, diag_games=4 if smoke else 160, self_games=4 if smoke else 200,
        white_check_games=4 if smoke else 160,
        par_seed=base, continuation_seed=base + 1_000_000, screen_seed=base + 2_000_000,
        deep_probe_seed=base + 3_000_000, white_check_seed=base + 4_000_000, release_par_seed=base + 5_000_000,
        gate_seed=base + 10_000_000, release_gate_seed=base + 12_000_000, diag_seed=base + 20_000_000,
        arms_seed=base + 30_000_000)


def recipe_path(kind, smoke):
    return ROOT / "tools/recipes" / f"gen55{kind}{'_rehearsal' if smoke else ''}.json"


def identity():
    d = decision()
    models = {p: file_hash(ROOT / p) for p in [d["teacher"], d["release"], *(t["model"] for t in d["extra_teachers"]),
                                                *d["pool_opponents"], *HELD_OUT]}
    held = {str(Path(p)) for p in HELD_OUT}
    for smoke in (False, True):
        for kind, model in [("", d["teacher"]), ("_pool", d["teacher"])] + \
                [(f"_teacher_{t['name']}", t["model"]) for t in d["extra_teachers"]]:
            recipe = read(recipe_path(kind, smoke))
            prior.validate_recipe(recipe)
            if recipe["model"] != model or recipe.get("temperature_plies") != 30:
                raise ValueError(f"unexpected gen55 recipe {kind} (smoke={smoke})")
            if {str(Path(p)) for p in recipe["opponents"]} & held:
                raise ValueError("held-out opponents may never be training opponents")
        if read(recipe_path("_pool", smoke))["opponents"] != d["pool_opponents"]:
            raise ValueError("pool recipe opponents differ from the decision")
    if {str(Path(t["model"])) for t in d["extra_teachers"]} & held or str(Path(d["teacher"])) in held:
        raise ValueError("held-out models may never be teachers")
    replay = sorted((e for e in read(ROOT / "iterations/accepted_data.json")["entries"]
                     if int(e["generation"]) < 55), key=lambda e: int(e["generation"]))[-7:]
    if len(replay) != 7 or int(replay[-1]["generation"]) != 54:
        raise ValueError("expected accepted replay through gen54")
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


def measure_par(campaign, smoke, name, bar, seed):
    par_dir = campaign.root / name
    campaign.stage(name, ["tools/gate_depth.py", "--par-only", "--bar-model", bar, "--seed", str(seed),
                          "--run-dir", str(par_dir), *(["--rehearsal"] if smoke else [])], [par_dir / "report.json"])
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
    """Run a stateful_generation recipe and process its games as their own increment (ramped labels).

    Each run gets its own directory: stateful_generation keeps its resume
    manifest and per-game receipts next to the summary, and task ids
    (selfplay/game_00000, ...) repeat across recipes, so a shared directory
    would make the second run collide with the first.
    """
    summary = campaign.root / "generation" / label / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
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


def passes_deep_guard(c):
    return (c["deep"]["score"] >= gate_depth.GUARD_AGGREGATE_MIN - 1e-12 and
            all(c["deep"]["par_comparisons"][s]["delta_from_par"] >= -gate_depth.GUARD_SIDE_BAND - 1e-12
                for s in ("white", "black")))


def passes_white_check(c, max_deficit):
    return c["white_check"]["white_delta_from_release_par"] >= -max_deficit - 1e-12


def white_check_nominee(candidates, max_deficit):
    """Pre-declared (GEN55_PLAN.md): best screen rank among epochs passing BOTH the deep guard
    against the teacher and the White check against the release; else best on the White check."""
    eligible = [c for c in candidates if passes_deep_guard(c) and passes_white_check(c, max_deficit)]
    if eligible:
        return eligible[0]["name"], [c["name"] for c in eligible], "best_screen_rank_passing_deep_guard_and_white_check"
    best = max(enumerate(candidates),
               key=lambda item: (item[1]["white_check"]["white_delta_from_release_par"], -item[0]))[1]
    return best["name"], [], "none_passed_both; best_white_check"


def select_arm(campaign, smoke, model_dir, deep_par, release_par):
    spec, d = layout(smoke), decision()
    teacher, release, rule = d["teacher"], d["release"], d["white_check"]
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
        check = g51.match(campaign, f"white_check_a_{i}", c["checkpoint"], release, spec["white_check_games"],
                          spec["white_check_seed"] + i * 1_000_000, spec["sims"], directory="selection")
        stats = check["diagnostics"]["sampled"]
        comparison = compare(stats, release_par)
        c["white_check"] = dict(score=stats["score"], sides=stats["sides"], par_comparisons=comparison,
                                white_delta_from_release_par=comparison["white"]["delta_from_par"])
    name, eligible, why = white_check_nominee(candidates, rule["max_white_deficit_vs_release_par"])
    chosen = next(c for c in candidates if c["name"] == name)
    source = Path(chosen["checkpoint"]) if Path(chosen["checkpoint"]).is_absolute() else ROOT / chosen["checkpoint"]
    target = model_dir / "arena_selected.pt"
    if target.exists() and file_hash(target) != file_hash(source):
        raise FileExistsError(f"{target} holds a different model; refusing to overwrite")
    if not target.exists():
        shutil.copy2(source, target)
    nominee = dict(arm="a", name=name, rule=why, eligible=eligible, candidates=candidates,
                   path=str(target), sha256=file_hash(target))
    pin_json(campaign.root / "selection" / "a_nominee.json", nominee)
    return nominee


def gate(campaign, smoke, name, nominee, bar, seed, par_dir):
    gate_dir = campaign.root / "gates" / name
    campaign.stage(f"gate_{name}", ["tools/gate_depth.py", "--model", nominee["path"], "--bar-model", bar,
        "--seed", str(seed), "--run-dir", str(gate_dir), "--par-dir", str(par_dir),
        *(["--rehearsal"] if smoke else [])], [gate_dir / "report.json"])
    return gate_depth.validate_report(gate_dir / "report.json")


def evaluate(campaign, smoke, nominee, par_dir, release_par_dir):
    spec, d = layout(smoke), decision()
    vs_teacher = gate(campaign, smoke, "a", nominee, d["teacher"], spec["gate_seed"], par_dir)
    vs_release = gate(campaign, smoke, "release", nominee, d["release"], spec["release_gate_seed"], release_par_dir)
    schedule = [("vs_b2", prior.HOLDOUT, spec["sims"], spec["diag_games"]),
                ("vs_v27", prior.RELEASE, spec["sims"], spec["diag_games"]),
                ("vs_gen49", GEN49, spec["sims"], spec["diag_games"]),
                ("vs_v28", V28, spec["sims"], spec["diag_games"]),
                ("self", nominee["path"], spec["sims"], spec["self_games"]),
                ("deep_self", nominee["path"], spec["deep_sims"], spec["diag_games"])]
    diagnostics = {name: g51.match(campaign, f"a_{name}", nominee["path"], other, games,
                                   spec["diag_seed"] + i * 1_000_000, sims)
                   for i, (name, other, sims, games) in enumerate(schedule)}
    probe_dir, audit_dir = campaign.root / "position_probe", campaign.root / "value_audit"
    campaign.stage("position_probe", ["tools/position_probe.py", "--candidate", f"gen55={nominee['path']}",
                                      "--out", str(probe_dir)], [probe_dir / "summary.json"])
    campaign.stage("value_audit", ["tools/value_colour_audit.py", "--models", "v29",
                                   "--model", f"gen54={d['teacher']}", "--model", f"gen55={nominee['path']}",
                                   "--out", str(audit_dir)], [audit_dir / "report.json"])
    held = [diagnostics[n]["diagnostics"]["sampled"]["score"] for n in ("vs_b2", "vs_v27")]
    held_mean = sum(held) / len(held)
    eligible = vs_release["verdict"] == "PASS" and held_mean >= HELD_OUT_MIN - 1e-12
    return dict(verdict=vs_teacher["verdict"], primary=vs_teacher["primary"], guard=vs_teacher["guard"],
                legs=vs_teacher["legs"], combined_h2h=vs_teacher["combined_h2h"],
                release_gate=dict(verdict=vs_release["verdict"], primary=vs_release["primary"],
                                  guard=vs_release["guard"], legs=vs_release["legs"]),
                diagnostics=diagnostics, held_out=["vs_b2", "vs_v27"], held_out_mean=held_mean,
                promotion=dict(eligible=eligible, rule="docs/protocols/PROMOTION_RULE.md",
                               gate_vs_release=vs_release["verdict"], held_out_mean=held_mean,
                               held_out_min=HELD_OUT_MIN, note="eligibility only; the owner decides"))


def work(campaign, smoke):
    d = decision()
    spec = layout(smoke)
    par_dir = measure_par(campaign, smoke, "par", d["teacher"], spec["par_seed"])
    deep_par = self_par(read_rows(par_dir / "deep_par.jsonl"))
    release_par_dir = measure_par(campaign, smoke, "release_par", d["release"], spec["release_par_seed"])
    release_par = self_par(read_rows(release_par_dir / "par.jsonl"))
    state = iteration(campaign, smoke)
    base = [] if smoke else [(s["name"], ROOT / s["path"]) for s in d.get("prior_extra_sources", [])]
    base.append((f"gen_{spec['generation']:04d}_deepvalue", deep_source(campaign, smoke)))
    model_dir = compose_and_train(campaign, smoke, state, base + extra_sources(campaign, smoke))
    nominee = select_arm(campaign, smoke, model_dir, deep_par, release_par)
    result = evaluate(campaign, smoke, nominee, par_dir, release_par_dir)
    if identity() != campaign.provenance:
        raise ValueError("inputs changed before publication")
    pin_json(campaign.root / "summary.json", dict(
        complete=True, rehearsal=smoke, decision=d, nominee=nominee, result=result,
        manifest_sha256=file_hash(campaign.root / "manifest.json"),
        notes=["Single arm: gen54-teacher data + gen53, Arm R and Arm LR self-play + hole-scan pool (+ gen53).",
               "Selection adds a White check against v29; gates against gen54 and v29.",
               "Nothing promoted.", "B2 and v27 are held out (never teachers or training opponents)."]))
    atomic_json(campaign.root / "status.json", dict(status="complete"))
    print(f"GEN55 {'REHEARSAL' if smoke else 'PRODUCTION'} COMPLETE", flush=True)


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
                                identity_fn=identity, label="GEN55")
            try:
                if smoke:
                    campaign.stage("tests", ["-m", "pytest", "-p", "no:cacheprovider", "tests/test_gen55_campaign.py",
                                             "tests/test_gate_depth.py", "tests/test_stateful_recipe.py",
                                             "tests/test_repetition_search.py", "-q"])
                elif not read(RUN / "rehearsal/summary.json")["complete"]:
                    raise ValueError("rehearsal required")
                work(campaign, smoke)
            except BaseException as exc:
                atomic_json(campaign.root / "status.json", dict(status="failed", error=str(exc)))
                raise


if __name__ == "__main__":
    main()
