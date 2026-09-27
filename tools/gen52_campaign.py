"""Gen52: gen51-winner teacher, shared generation, base arm vs model-pool arm, gate v4.

docs/plans/GEN52_PLAN.md (owner decisions 2026-09-26; overnight authority
2026-09-27). Fixed chain; every branch runs regardless of measured results;
an execution error stops safely:

 1. Teacher par (gate_depth --par-only).
 2. Canonical gen52 iteration through `compose` (shared self-play + forks +
    reanalysis, 30-ply exploration).
 3. If the deep-value source is kept (DEEP_VALUE): disagreement continuations
    and their parent-linked value-only increment, as in gen51.
 4. Pool games: 1,200 teacher-vs-pool games (league tasks; only the teacher's
    moves carry policy weight), processed as their own increment.
 5. Arm A replay = canonical sources (+ deep source); Arm B = Arm A + pool.
    Both train with gen51's exact train command, other data/model dirs.
 6. Selection per arm (matched seeds): checkpoint_screen vs the teacher, then
    12,800 probes of the top three epochs (gen51 rule).
 7. Gate v4 per nominee vs the teacher; diagnostics vs v28, gen49 and the
    held-out B2 and v27, and self-play; if both pass, Arm B vs Arm A.

B2 and v27 are never training opponents (owner, 2026-09-26). Nothing promoted.
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

RUN = ROOT / "benchmarks/gen52_program/gen52_20260927"
SMOKE_ROOT = ROOT / "iterations/rehearsal_gen52_20260927"
DECISION = ROOT / "docs/plans/gen52_teacher_decision.json"
V28, GEN49 = g51.V28, g51.GEN49
GEN48 = "models/candidates/bootstrap_main_gen_0048/arena_selected.pt"
V26 = "models/bootstrap_v26/best_value_net.pt"
POOL = [V28, GEN49, GEN48, V26]
HELD_OUT = [prior.HOLDOUT, prior.RELEASE]  # B2 and v27: never training opponents
PINNED = ["tools/gen52_campaign.py", "tools/gen51_campaign.py", "tools/disagreement_continuations.py",
          "tools/process_linked_extra.py", "tools/stateful_generation.py", "tools/iterate_stateful.py",
          "tools/reanalyze_coverage.py", "tools/reanalyze_stateful.py", "tools/reanalyze.py",
          "tools/reanalysis_journal.py", "tools/process_families.py", "tools/compose_processed.py",
          "tools/checkpoint_screen.py", "tools/audit_generation_data.py", "tools/gate_depth.py", "tools/match.py",
          "tools/free_gate_stats.py", "tools/sampled_gate_stats.py", "tools/free_play_audit.py",
          "tools/start_gpu48.py", "tools/start_gen49.py", "tools/gate_sampled.py",
          "tools/recipes/gen52.json", "tools/recipes/gen52_rehearsal.json",
          "tools/recipes/gen52_pool.json", "tools/recipes/gen52_pool_rehearsal.json",
          "tests/test_gen52_campaign.py", "tests/test_gate_depth.py",
          "docs/plans/GEN52_PLAN.md", "docs/plans/gen52_teacher_decision.json"]


def decision():
    d = read(DECISION)
    if file_hash(ROOT / d["teacher"]) != d["teacher_sha256"]:
        raise ValueError("teacher checkpoint changed")
    return d


def layout(smoke):
    base = 3_150_000_000 if smoke else 3_100_000_000
    return dict(
        generation=1 if smoke else 52, run_root=SMOKE_ROOT if smoke else ROOT / "iterations",
        sims=8 if smoke else 3200, deep_sims=16 if smoke else 12800,
        roots=8 if smoke else 768, continuations=2, continuation_sims=8 if smoke else 6400,
        screen_games=4 if smoke else 200, probe_games=4 if smoke else 40, finalists=1 if smoke else 2,
        deep_probe_games=4 if smoke else 80, diag_games=4 if smoke else 160, self_games=4 if smoke else 200,
        arms_games=4 if smoke else 400, arms_deep_games=4 if smoke else 160,
        par_seed=base, continuation_seed=base + 1_000_000, screen_seed=base + 2_000_000,
        deep_probe_seed=base + 3_000_000, gate_seed=base + 10_000_000, diag_seed=base + 20_000_000,
        arms_seed=base + 30_000_000)


def identity():
    d = decision()
    models = {p: file_hash(ROOT / p) for p in [d["teacher"], *POOL, *HELD_OUT]}
    for name in ("gen52.json", "gen52_rehearsal.json", "gen52_pool.json", "gen52_pool_rehearsal.json"):
        recipe = read(ROOT / "tools/recipes" / name)
        prior.validate_recipe(recipe)
        if recipe["model"] != d["teacher"] or recipe.get("temperature_plies") != 30:
            raise ValueError(f"unexpected gen52 recipe {name}")
        if set(recipe["opponents"]) & set(HELD_OUT):
            raise ValueError("held-out opponents may never be training opponents")
    replay = sorted((e for e in read(ROOT / "iterations/accepted_data.json")["entries"]
                     if int(e["generation"]) < 52), key=lambda e: int(e["generation"]))[-7:]
    if len(replay) != 7 or int(replay[-1]["generation"]) != 51:
        raise ValueError("expected accepted replay through gen51")
    return dict(runtime=runtime_identity(), decision=d, models=models, replay=replay,
                implementations={p: file_hash(ROOT / p) for p in PINNED},
                protocol={k: str(v) for k, v in layout(False).items()},
                rehearsal={k: str(v) for k, v in layout(True).items()})


def iteration_command(smoke):
    command = g51.iteration_command(smoke)
    changes = {"--recipe": f"tools/recipes/gen52{'_rehearsal' if smoke else ''}.json",
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
        a=dict(model_dir=paths["candidate_dir"], replay=processed / f"bootstrap_replay_{stem}_armA"),
        b=dict(model_dir=paths["candidate_dir"].with_name(paths["candidate_dir"].name + "_pool"),
               replay=processed / f"bootstrap_replay_{stem}_armB"))


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
    return extra["deep"]


def pool_source(campaign, smoke):
    paths, extra = paths_for(smoke)
    recipe = f"tools/recipes/gen52_pool{'_rehearsal' if smoke else ''}.json"
    summary = campaign.root / "pool_generation_summary.json"
    campaign.stage("pool_games", ["tools/stateful_generation.py", "--config", recipe, "--raw", str(extra["pool_raw"]),
                                  "--summary", str(summary)], [summary])
    record = read(summary)
    if record["saved_games"] != read(ROOT / recipe)["league_games"] or record["failed_games"]:
        raise ValueError("pool games incomplete")
    campaign.stage("pool_increment", ["tools/process_families.py", "--raw-dir", str(extra["pool_raw"]),
        "--output-dir", str(extra["pool"]), "--seed", g51.iteration_seed(smoke), "--channels", "15",
        "--value-floor", ".5", "--value-horizon", "60"], [extra["pool"] / "split_game_ids.json"])
    return extra["pool"]


def compose_and_train(campaign, smoke, state, arm, sources):
    _, extra = paths_for(smoke)
    arm_paths = extra[arm]
    compose = list(state["phases"]["compose"]["commands"][0])
    compose[compose.index("--output-dir") + 1] = str(arm_paths["replay"])
    for name, path in sources:
        compose += ["--source", f"{name}={path}"]
    campaign.stage(f"compose_{arm}", compose, [arm_paths["replay"] / "replay_manifest.json"])
    # gen51's exact training recipe; only the data and model directories differ.
    template = read(g51.arm_paths(smoke)[0]["state"])["phases"]["train"]["commands"][0]
    train = list(template)
    for flag, value in (("--data-dir", arm_paths["replay"]), ("--model-dir", arm_paths["model_dir"])):
        train[train.index(flag) + 1] = str(value)
    if arm_paths["model_dir"].exists() and not (campaign.root / f"receipts/train_{arm}.json").exists():
        raise ValueError(f"unreceipted arm {arm} training output retained; refusing automatic overwrite")
    campaign.stage(f"train_{arm}", train, [arm_paths["model_dir"] / "best_value_net.pt"])
    return arm_paths["model_dir"]


def select_arm(campaign, smoke, arm, model_dir, deep_par):
    spec, teacher = layout(smoke), decision()["teacher"]
    report_path = campaign.root / "selection" / f"{arm}_screen.json"
    campaign.stage(f"screen_{arm}", ["tools/checkpoint_screen.py", "--model-dir", str(model_dir),
        "--incumbent", teacher, "--output-model", str(model_dir / "screen_pick.pt"), "--report-path", str(report_path),
        "--games", str(spec["screen_games"]), "--probe-games", str(spec["probe_games"]),
        "--sims", str(spec["sims"]), "--probe-sims", str(spec["sims"]), "--finalists", str(spec["finalists"]),
        "--workers", "8", "--engine", "native", "--seed", str(spec["screen_seed"]), "--stall-timeout", "600.0"],
        [report_path, model_dir / "screen_pick.pt"])
    candidates = g51.screen_candidates(read(report_path))
    for i, c in enumerate(candidates):
        audit = g51.match(campaign, f"deep_probe_{arm}_{i}", c["checkpoint"], teacher, spec["deep_probe_games"],
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
    nominee = dict(arm=arm, name=name, rule=rule, eligible=eligible, candidates=candidates,
                   path=str(target), sha256=file_hash(target))
    pin_json(campaign.root / "selection" / f"{arm}_nominee.json", nominee)
    return nominee


def evaluate_arm(campaign, smoke, arm, nominee, par_dir):
    spec, teacher = layout(smoke), decision()["teacher"]
    gate_dir = campaign.root / "gates" / arm
    campaign.stage(f"gate_{arm}", ["tools/gate_depth.py", "--model", nominee["path"], "--bar-model", teacher,
        "--seed", str(spec["gate_seed"]), "--run-dir", str(gate_dir), "--par-dir", str(par_dir),
        *(["--rehearsal"] if smoke else [])], [gate_dir / "report.json"])
    gate = gate_depth.validate_report(gate_dir / "report.json")
    schedule = [("vs_b2", prior.HOLDOUT, spec["sims"], spec["diag_games"]),
                ("vs_v27", prior.RELEASE, spec["sims"], spec["diag_games"]),
                ("vs_gen49", GEN49, spec["sims"], spec["diag_games"]),
                ("vs_v28", V28, spec["sims"], spec["diag_games"]),
                ("self", nominee["path"], spec["sims"], spec["self_games"]),
                ("deep_self", nominee["path"], spec["deep_sims"], spec["diag_games"])]
    diagnostics = {name: g51.match(campaign, f"{arm}_{name}", nominee["path"], other, games,
                                   spec["diag_seed"] + i * 1_000_000, sims)
                   for i, (name, other, sims, games) in enumerate(schedule)}
    return dict(verdict=gate["verdict"], primary=gate["primary"], guard=gate["guard"], legs=gate["legs"],
                combined_h2h=gate["combined_h2h"], diagnostics=diagnostics,
                held_out=["vs_b2", "vs_v27"], pool_opponents_not_independent_for_arm_b=["vs_gen49", "vs_v28"])


def work(campaign, smoke):
    spec, d = layout(smoke), decision()
    par_dir = teacher_par(campaign, smoke)
    deep_par = self_par(read_rows(par_dir / "deep_par.jsonl"))
    state = iteration(campaign, smoke)
    # Earlier value-only increments the teacher's own recipe used (e.g. gen51's
    # disagreement increment) roll forward with the replay; smoke uses none.
    base = [] if smoke else [(s["name"], ROOT / s["path"]) for s in d.get("prior_extra_sources", [])]
    if d["deep_value"]:
        base.append((f"gen_{spec['generation']:04d}_deepvalue", deep_source(campaign, smoke)))
    pool = pool_source(campaign, smoke)
    dirs = {"a": compose_and_train(campaign, smoke, state, "a", base),
            "b": compose_and_train(campaign, smoke, state, "b", base + [(f"gen_{spec['generation']:04d}_pool", pool)])}
    nominees = {arm: select_arm(campaign, smoke, arm, dirs[arm], deep_par) for arm in ("a", "b")}
    results = {arm: evaluate_arm(campaign, smoke, arm, nominees[arm], par_dir) for arm in ("a", "b")}
    head_to_head = None
    if all(r["verdict"] == "PASS" for r in results.values()):
        a, b = nominees["b"]["path"], nominees["a"]["path"]
        head_to_head = {
            "pool_vs_base": g51.match(campaign, "pool_vs_base", a, b, spec["arms_games"], spec["arms_seed"], spec["sims"]),
            "pool_vs_base_deep": g51.match(campaign, "pool_vs_base_deep", a, b, spec["arms_deep_games"],
                                           spec["arms_seed"] + 1_000_000, spec["deep_sims"])}
    if identity() != campaign.provenance:
        raise ValueError("inputs changed before publication")
    pin_json(campaign.root / "summary.json", dict(
        complete=True, rehearsal=smoke, decision=d, nominees=nominees, results=results, head_to_head=head_to_head,
        manifest_sha256=file_hash(campaign.root / "manifest.json"),
        notes=["All branches ran regardless of measured results.", "Nothing promoted.",
               "B2 and v27 are held out (never training opponents); pool opponents are not independent for Arm B."]))
    atomic_json(campaign.root / "status.json", dict(status="complete"))
    print(f"GEN52 {'REHEARSAL' if smoke else 'PRODUCTION'} COMPLETE", flush=True)


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
                                identity_fn=identity, label="GEN52")
            try:
                if smoke:
                    campaign.stage("tests", ["-m", "pytest", "-p", "no:cacheprovider", "tests/test_gen52_campaign.py",
                                             "tests/test_gen51_campaign.py", "tests/test_gate_depth.py",
                                             "tests/test_stateful_recipe.py", "-q"])
                elif not read(RUN / "rehearsal/summary.json")["complete"]:
                    raise ValueError("rehearsal required")
                work(campaign, smoke)
            except BaseException as exc:
                atomic_json(campaign.root / "status.json", dict(status="failed", error=str(exc)))
                raise


if __name__ == "__main__":
    main()
