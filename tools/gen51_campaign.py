"""Gen51: v28 teacher, shared generation, control vs deep-value arms, gate v4.

Stages 2-3 of docs/plans/GEN51_STRENGTH_PLAN.md (owner-approved 2026-09-25,
exploration change approved the same day). Fixed chain, every branch runs
regardless of measured results; an execution error stops safely:

 1. Incumbent par: reuse the Stage 1 v28 par if gate v4 accepts it for this
    runtime, else measure it (gate_depth --par-only).
 2. Canonical gen51 iteration through `train` (control arm): gen50 recipe,
    teacher v28, self-play temperature 1.0 for 30 plies.
 3. Disagreement continuations: 768 roots x 2 games at 6,400 (value source).
 4. Extra increment in the parents' splits: strict labels, value weight 4,
    policy weight 0. Deep-value replay = control replay + this source.
 5. Deep-value training: the control's exact train command, other data/model dir.
 6. Selection per arm (matched seeds): checkpoint_screen at 3,200, then 80-game
    12,800 probes of the top three epochs; nomination rule below.
 7. Gate v4 per nominee against v28; diagnostics per nominee; if both pass,
    the two nominees play each other.

Nothing is promoted. Nominees are copied to each arm's `arena_selected.pt`
for the owner's playtest.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import iterate
from checkpoint_screen import rank_key
from free_play_audit import audit_match
from match_evidence import atomic_json, file_hash, read_rows, runtime_identity
from sampled_gate_stats import compare, self_par
from start_gpu48 import Campaign, campaign_lock, pin_json, read
import gate_depth
import start_gen49 as prior

RUN = ROOT / "benchmarks/gen51_program/gen51_20260925"
SMOKE_ROOT = ROOT / "iterations/rehearsal_gen51_20260925"
STAGE1 = ROOT / "benchmarks/gen51_program/search_constants_20260925"
V28 = "models/bootstrap_v28/best_value_net.pt"
E14 = "models/candidates/bootstrap_main_gen_0050/arena_selected.pt"
GEN49 = "models/candidates/bootstrap_main_gen_0049/arena_selected.pt"
MODELS = {V28: "b651e7405afe4e5672676c6fb13bb5bc6e1f5ea0071fd5ce4f4d5318e229e35a",
          E14: "51b5ddb01db51ae9023eaaf8ccbd896b48a805a52b2707633dc1d7e3f8067f25",
          GEN49: "4bcc68a0219acf8c3dc53326d738789e6bd767fccba88fd4f471345567b4a647",
          prior.RELEASE: prior.PINNED_MODELS[prior.RELEASE],
          prior.HOLDOUT: prior.PINNED_MODELS[prior.HOLDOUT]}
DEEP_VALUE_WEIGHT = 4.0
DEEP_PROBE_GAMES = 80
TOP_EPOCHS = 3
PINNED = ["tools/gen51_campaign.py", "tools/disagreement_continuations.py", "tools/process_linked_extra.py",
          "tools/stateful_generation.py", "tools/iterate_stateful.py", "tools/reanalyze_coverage.py",
          "tools/reanalyze_stateful.py", "tools/reanalyze.py", "tools/reanalysis_journal.py",
          "tools/process_families.py", "tools/compose_processed.py", "tools/checkpoint_screen.py",
          "tools/audit_generation_data.py", "tools/gate_depth.py", "tools/match.py", "tools/free_gate_stats.py",
          "tools/sampled_gate_stats.py", "tools/free_play_audit.py", "tools/start_gpu48.py",
          "tools/start_gen49.py", "tools/gate_sampled.py", "tools/recipes/gen51.json",
          "tools/recipes/gen51_rehearsal.json", "tests/test_gen51_campaign.py", "tests/test_gate_depth.py",
          "docs/plans/GEN51_STRENGTH_PLAN.md"]


def layout(smoke):
    base = 2_970_000_000 if smoke else 2_910_000_000
    return dict(
        generation=1 if smoke else 51, run_root=SMOKE_ROOT if smoke else ROOT / "iterations",
        sims=8 if smoke else 3200, deep_sims=16 if smoke else 12800,
        roots=8 if smoke else 768, continuations=2, continuation_sims=8 if smoke else 6400,
        screen_games=4 if smoke else 200, probe_games=4 if smoke else 40, finalists=1 if smoke else 2,
        deep_probe_games=4 if smoke else DEEP_PROBE_GAMES,
        diag_games=4 if smoke else 160, self_games=4 if smoke else 200,
        arms_games=4 if smoke else 400, arms_deep_games=4 if smoke else 160,
        continuation_seed=base, screen_seed=base + 1_000_000, deep_probe_seed=base + 2_000_000,
        gate_seed=base + 10_000_000, diag_seed=base + 20_000_000, arms_seed=base + 30_000_000)


def identity():
    hashes = {p: file_hash(ROOT / p) for p in MODELS}
    if hashes != MODELS:
        raise ValueError("frozen reference model changed")
    replay = sorted((e for e in read(ROOT / "iterations/accepted_data.json")["entries"]
                     if int(e["generation"]) < 51), key=lambda e: int(e["generation"]))[-7:]
    if len(replay) != 7 or int(replay[-1]["generation"]) != 50:
        raise ValueError("expected accepted replay through gen50")
    for name in ("gen51.json", "gen51_rehearsal.json"):
        recipe = read(ROOT / "tools/recipes" / name)
        prior.validate_recipe(recipe)
        if recipe["model"] != V28 or recipe.get("temperature_plies") != 30:
            raise ValueError(f"unexpected gen51 recipe {name}")
    return dict(runtime=runtime_identity(), models=hashes, replay=replay,
                implementations={p: file_hash(ROOT / p) for p in PINNED},
                protocol=layout_record(False), rehearsal=layout_record(True),
                deep_value_weight=DEEP_VALUE_WEIGHT)


def layout_record(smoke):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in layout(smoke).items()}


def iteration_command(smoke):
    command = prior.iteration_command(smoke)
    changes = {"--recipe": f"tools/recipes/gen51{'_rehearsal' if smoke else ''}.json",
               "--expected-generation": str(layout(smoke)["generation"]), "--incumbent": V28,
               "--reanalysis-sims": "8" if smoke else "12800", "--through-phase": "train"}
    if smoke:
        changes["--run-root"] = str(SMOKE_ROOT)
    for flag, value in changes.items():
        command[command.index(flag) + 1] = value
    return command


def iteration_seed(smoke):
    command = iteration_command(smoke)
    return command[command.index("--seed") + 1]


def arm_paths(smoke):
    spec = layout(smoke)
    paths = iterate._paths_for_generation(spec["run_root"], spec["generation"])
    namespace = iterate._run_namespace(spec["run_root"])
    n = spec["generation"]
    processed = ROOT / "data/processed"
    return paths, {
        "control": dict(model_dir=paths["candidate_dir"], replay=paths["replay_processed"]),
        "deep": dict(model_dir=paths["candidate_dir"].with_name(paths["candidate_dir"].name + "_deepvalue"),
                     replay=processed / f"bootstrap_replay_{namespace}_gen_{n:04d}_deepvalue",
                     extra=processed / f"bootstrap_extra_{namespace}_gen_{n:04d}_disagreement"),
    }


def incumbent_par(campaign, smoke):
    """Stage 1's v28 par when gate v4 accepts it for this runtime; else measure."""
    rehearsal = ["--rehearsal"] if smoke else []
    stage1 = STAGE1 / ("rehearsal" if smoke else "production") / "par"
    proto = gate_depth.protocol(smoke, 0)
    try:
        gate_depth.load_par_source(stage1, V28, dict(proto, seed=None))
        pin_json(campaign.root / "par_source.json", dict(dir=str(stage1), reused=True,
                 report_sha256=file_hash(stage1 / "report.json")))
        return stage1
    except (ValueError, FileNotFoundError, KeyError) as exc:
        own = campaign.root / "par"
        pin_json(campaign.root / "par_source.json", dict(dir=str(own), reused=False, reason=str(exc)))
        campaign.stage("par", ["tools/gate_depth.py", "--par-only", "--bar-model", V28, "--seed",
                               str(layout(smoke)["gate_seed"] - 1_000_000), "--run-dir", str(own), *rehearsal],
                       [own / "report.json"])
        return own


def train_control(campaign, smoke):
    paths, _ = arm_paths(smoke)
    command = iteration_command(smoke)
    if paths["state"].exists():
        state = read(paths["state"])
        if state["phases"].get("train", {}).get("status") in ("running", "failed"):
            raise ValueError("interrupted training retained; refusing automatic overwrite")
        command += ["--resume"]
    receipt = campaign.root / "receipts/iteration.json"
    if receipt.exists():
        command = read(receipt)["command"]
    campaign.stage("iteration", command, [paths["state"], paths["training_candidate"]])
    state = read(paths["state"])
    for name in iterate.PHASES[:iterate.PHASES.index("train") + 1]:
        if state["phases"].get(name, {}).get("status") != "completed":
            raise ValueError(f"iteration phase incomplete: {name}")
    return state


def continuations(campaign, smoke):
    spec, (paths, _) = layout(smoke), arm_paths(smoke)
    config = dict(raw_dir=str(paths["raw"].relative_to(ROOT)).replace("\\", "/"), player=V28, reference=E14,
                  roots=spec["roots"], continuations=spec["continuations"], sims=spec["continuation_sims"],
                  seed=spec["continuation_seed"], workers=8, min_ply=4, max_ply=120)
    path = campaign.root / "continuations_config.json"
    pin_json(path, config)
    out = campaign.root / "continuations"
    campaign.stage("continuations", ["tools/disagreement_continuations.py", "--config", str(path), "--out", str(out)],
                   [out / "summary.json", out / "roots.json"])
    summary = read(out / "summary.json")
    if not summary["complete"] or summary["games"] != spec["roots"] * spec["continuations"]:
        raise ValueError("continuations incomplete")
    return out


def train_deep(campaign, smoke, state, continuation_dir):
    paths, arms = arm_paths(smoke)
    deep = arms["deep"]
    campaign.stage("extra_increment", ["tools/process_linked_extra.py", "--raw-dir", str(continuation_dir / "raw"),
        "--parent-increment", str(paths["new_processed"]), "--output-dir", str(deep["extra"]),
        "--seed", iteration_seed(smoke), "--value-weight", str(DEEP_VALUE_WEIGHT)],
        [deep["extra"] / "derivation.json"])
    compose = list(state["phases"]["compose"]["commands"][0])
    compose[compose.index("--output-dir") + 1] = str(deep["replay"])
    compose += ["--source", f"gen_{layout(smoke)['generation']:04d}_deepvalue={deep['extra']}"]
    campaign.stage("compose_deep", compose, [deep["replay"] / "replay_manifest.json"])
    train = list(state["phases"]["train"]["commands"][0])
    for flag, value in (("--data-dir", deep["replay"]), ("--model-dir", deep["model_dir"])):
        train[train.index(flag) + 1] = str(value)
    if deep["model_dir"].exists() and not (campaign.root / "receipts/train_deep.json").exists():
        raise ValueError("unreceipted deep-value training output retained; refusing automatic overwrite")
    campaign.stage("train_deep", train, [deep["model_dir"] / "best_value_net.pt"])


def screen_candidates(report, count=TOP_EPOCHS):
    """Top epochs by the screen's own rank, filling from probe-only results."""
    full = sorted(report["results"], key=rank_key, reverse=True)
    chosen = [dict(name=r["name"], checkpoint=r["checkpoint"], stage="screen", rank=list(rank_key(r)))
              for r in full[:count]]
    names = {c["name"] for c in chosen}
    for r in sorted(report["probe"]["results"], key=rank_key, reverse=True):
        if len(chosen) == count:
            break
        if r["name"] not in names:
            chosen.append(dict(name=r["name"], checkpoint=r["checkpoint"], stage="probe", rank=list(rank_key(r))))
            names.add(r["name"])
    return chosen


def deep_nominee(candidates):
    """Best-ranked candidate passing the guard rule at 12,800; else best deep score."""
    def passes(c):
        return (c["deep"]["score"] >= gate_depth.GUARD_AGGREGATE_MIN - 1e-12 and
                all(c["deep"]["par_comparisons"][s]["delta_from_par"] >= -gate_depth.GUARD_SIDE_BAND - 1e-12
                    for s in ("white", "black")))
    eligible = [c for c in candidates if passes(c)]
    if eligible:
        return eligible[0]["name"], [c["name"] for c in eligible], "best_screen_rank_passing_deep_guard"
    best = max(enumerate(candidates), key=lambda item: (item[1]["deep"]["score"], -item[0]))[1]
    return best["name"], [], "none_passed_deep_guard; best_deep_score"


def match(campaign, name, a, b, games, seed, sims, directory="play"):
    report = campaign.root / directory / f"{name}.json"
    log = report.with_suffix(".jsonl")
    campaign.stage(name, ["tools/match.py", "--model-a", str(a), "--model-b", str(b), "--games", str(games),
        "--sims", str(sims), "--sims-b", str(sims), "--engine", "native", "--workers", "8",
        "--seed", str(seed), "--opening-temp-plies", "16", "--game-log", str(log),
        "--report-path", str(report), "--resume"], [report, log, log.with_suffix(".jsonl.manifest.json")])
    return audit_match(log, a, b, games, seed, sims)


def select_arm(campaign, smoke, arm, model_dir, deep_par):
    spec = layout(smoke)
    report_path = campaign.root / "selection" / f"{arm}_screen.json"
    campaign.stage(f"screen_{arm}", ["tools/checkpoint_screen.py", "--model-dir", str(model_dir),
        "--incumbent", V28, "--output-model", str(model_dir / "screen_pick.pt"), "--report-path", str(report_path),
        "--games", str(spec["screen_games"]), "--probe-games", str(spec["probe_games"]),
        "--sims", str(spec["sims"]), "--probe-sims", str(spec["sims"]), "--finalists", str(spec["finalists"]),
        "--workers", "8", "--engine", "native", "--seed", str(spec["screen_seed"]), "--stall-timeout", "600.0"],
        [report_path, model_dir / "screen_pick.pt"])
    candidates = screen_candidates(read(report_path))
    for i, c in enumerate(candidates):
        audit = match(campaign, f"deep_probe_{arm}_{i}", c["checkpoint"], V28, spec["deep_probe_games"],
                      spec["deep_probe_seed"] + i * 1_000_000, spec["deep_sims"], directory="selection")
        stats = audit["diagnostics"]["sampled"]
        c["deep"] = dict(score=stats["score"], sides=stats["sides"], par_comparisons=compare(stats, deep_par))
    name, eligible, rule = deep_nominee(candidates)
    chosen = next(c for c in candidates if c["name"] == name)
    source = ROOT / chosen["checkpoint"] if not Path(chosen["checkpoint"]).is_absolute() else Path(chosen["checkpoint"])
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
    spec = layout(smoke)
    gate_dir = campaign.root / "gates" / arm
    campaign.stage(f"gate_{arm}", ["tools/gate_depth.py", "--model", nominee["path"], "--bar-model", V28,
        "--seed", str(spec["gate_seed"]), "--run-dir", str(gate_dir), "--par-dir", str(par_dir),
        *(["--rehearsal"] if smoke else [])], [gate_dir / "report.json"])
    gate = gate_depth.validate_report(gate_dir / "report.json")
    diagnostics = {}
    schedule = [("vs_gen49", GEN49, spec["sims"], spec["diag_games"]),
                ("vs_v27", prior.RELEASE, spec["sims"], spec["diag_games"]),
                ("vs_b2", prior.HOLDOUT, spec["sims"], spec["diag_games"]),
                ("self", nominee["path"], spec["sims"], spec["self_games"]),
                ("deep_self", nominee["path"], spec["deep_sims"], spec["diag_games"])]
    for i, (name, other, sims, games) in enumerate(schedule):
        diagnostics[name] = match(campaign, f"{arm}_{name}", nominee["path"], other, games,
                                  spec["diag_seed"] + i * 1_000_000, sims)
    return dict(verdict=gate["verdict"], primary=gate["primary"], guard=gate["guard"], legs=gate["legs"],
                combined_h2h=gate["combined_h2h"], diagnostics=diagnostics)


def work(campaign, smoke, stop_after):
    spec = layout(smoke)
    par_dir = incumbent_par(campaign, smoke)
    deep_par = self_par(read_rows(Path(par_dir) / "deep_par.jsonl"))
    state = train_control(campaign, smoke)
    continuation_dir = continuations(campaign, smoke)
    train_deep(campaign, smoke, state, continuation_dir)
    if stop_after == "training":
        atomic_json(campaign.root / "status.json", dict(status="paused_after_training"))
        print("GEN51 PAUSED AFTER TRAINING (resume with the same command)", flush=True)
        return
    _, arms = arm_paths(smoke)
    nominees = {arm: select_arm(campaign, smoke, arm, arms[arm]["model_dir"], deep_par) for arm in ("control", "deep")}
    results = {arm: evaluate_arm(campaign, smoke, arm, nominees[arm], par_dir) for arm in ("control", "deep")}
    head_to_head = None
    if all(r["verdict"] == "PASS" for r in results.values()):
        a, b = nominees["deep"]["path"], nominees["control"]["path"]
        head_to_head = {
            "deep_vs_control": match(campaign, "deep_vs_control", a, b, spec["arms_games"], spec["arms_seed"], spec["sims"]),
            "deep_vs_control_deep": match(campaign, "deep_vs_control_deep", a, b, spec["arms_deep_games"],
                                          spec["arms_seed"] + 1_000_000, spec["deep_sims"])}
    if identity() != campaign.provenance:
        raise ValueError("inputs changed before publication")
    pin_json(campaign.root / "summary.json", dict(
        complete=True, rehearsal=smoke, par_source=read(campaign.root / "par_source.json"),
        nominees=nominees, results=results, head_to_head=head_to_head,
        continuations=read(continuation_dir / "summary.json"),
        manifest_sha256=file_hash(campaign.root / "manifest.json"),
        notes=["All branches ran regardless of measured results.", "Nothing promoted.",
               "Teacher (v28) and self-play exploration (30 plies) changed together for both arms; "
               "the arm comparison isolates only the deep-value source."]))
    atomic_json(campaign.root / "status.json", dict(status="complete"))
    print(f"GEN51 {'REHEARSAL' if smoke else 'PRODUCTION'} COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rehearsal-only", action="store_true")
    ap.add_argument("--stop-after", choices=("training",), help="pause production after both arms train")
    args = ap.parse_args()
    os.chdir(ROOT)
    os.environ["MONSTER_PINNED_INPUT"] = "1"
    provenance = identity()
    with campaign_lock(RUN / "campaign.lock"):
        for smoke in (True, False):
            if not smoke and args.rehearsal_only:
                break
            campaign = Campaign(RUN / ("rehearsal" if smoke else "production"), provenance,
                                identity_fn=identity, label="GEN51")
            try:
                if smoke:
                    campaign.stage("tests", ["-m", "pytest", "-p", "no:cacheprovider", "tests/test_gen51_campaign.py",
                                             "tests/test_gate_depth.py", "tests/test_stateful_recipe.py", "-q"])
                elif not read(RUN / "rehearsal/summary.json")["complete"]:
                    raise ValueError("rehearsal required")
                work(campaign, smoke, None if smoke else args.stop_after)
            except BaseException as exc:
                atomic_json(campaign.root / "status.json", dict(status="failed", error=str(exc)))
                raise


if __name__ == "__main__":
    main()
