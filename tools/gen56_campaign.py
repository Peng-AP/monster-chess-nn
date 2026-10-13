"""Gen56: train on the positions that decide games; select and measure each colour separately.

docs/plans/GEN56_PLAN.md (owner, 2026-10-12, on the opening census
recommendations: "Yes, work on it"). Derived from tools/gen55_campaign.py; the changes:

 1. Teacher gen55. Half its self-play starts from the census book (70 lines:
    14 key first turns, Black's two best replies to the best 6 first turns,
    the other 44 first turns for breadth), half from the normal start.
 2. Per-colour teachers (owner 2026-10-13, option d): 1,400 book-start games,
    White gen55 or gen54 against Black gen53 or v29, both sides policy targets.
 3. Pool: v29, gen54, gen53, gen52 B/C/L/LR/R; 86 games per opponent per colour.
 4. Selection: deep guard against gen55, plus a per-colour check against gen55
    and v29 on the census key book (14 x 6, 8 sampled half-moves): White at
    least the teacher's own, Black within 2 pp of it; best screen rank; else
    best White.
 5. Evaluation: the per-colour instrument (full census book, both references),
    gate v4 against gen55 and v29, diagnostics, probe, audit; eligibility under
    the current rule and under the proposed per-colour rule (owner decides).

Repetition-aware engine. Nothing promoted. B2 and v27 held out.
"""
import argparse
import json
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

RUN = ROOT / "benchmarks/gen56_program/gen56_20261013"
SMOKE_ROOT = ROOT / "iterations/rehearsal_gen56e_20261013"   # option d; earlier rehearsals superseded (d: model preload bug)
DECISION = ROOT / "docs/plans/gen56_teacher_decision.json"
V28, GEN49 = g51.V28, g51.GEN49
HELD_OUT = [prior.HOLDOUT, prior.RELEASE]  # B2 and v27: never training opponents or teachers
# PROMOTION_RULE.md: held-out mean no worse than the release's (v29: 92.7%) minus 1 pp.
HELD_OUT_MIN = 0.917
RECIPE_KINDS = ("", "_pool", "_paired")
# v29's per-colour scores on the census book (phase 1b), the baseline of the proposed rule.
CENSUS_REPORT = ROOT / "benchmarks/opening_census_20261012/phase1b/report.json"
PINNED = ["tools/gen56_campaign.py", "tools/opening_census.py", "tools/colour_instrument.py", "tools/gen51_campaign.py", "tools/disagreement_continuations.py",
          "tools/process_linked_extra.py", "tools/process_linked_extra_ramped.py", "tools/stateful_generation.py",
          "tools/iterate_stateful.py", "tools/reanalyze_coverage.py", "tools/reanalyze_stateful.py",
          "tools/reanalyze.py", "tools/reanalysis_journal.py", "tools/process_families.py",
          "tools/compose_processed.py", "tools/checkpoint_screen.py", "tools/audit_generation_data.py",
          "tools/gate_depth.py", "tools/match.py", "tools/free_gate_stats.py", "tools/sampled_gate_stats.py",
          "tools/free_play_audit.py", "tools/start_gpu48.py", "tools/start_gen49.py", "tools/gate_sampled.py",
          "tools/position_probe.py", "tools/value_colour_audit.py",
          "tests/test_gen56_campaign.py", "tests/test_gate_depth.py", "tests/test_stateful_recipe.py",
          "docs/plans/GEN56_PLAN.md", "docs/plans/gen56_teacher_decision.json",
          "benchmarks/opening_census_20261012/phase1b/book.json",
          "benchmarks/opening_census_20261012/phase1b/report.json",
          *[f"tools/recipes/gen56{k}{s}.json" for k in RECIPE_KINDS for s in ("", "_rehearsal")]]


def decision():
    d = read(DECISION)
    checks = [(d["teacher"], d["teacher_sha256"]), (d["release"], d["release_sha256"])]
    for model, sha in checks + [(m["model"], m["sha256"]) for m in d["pairing_models"].values()]:
        if file_hash(ROOT / model) != sha:
            raise ValueError(f"checkpoint changed: {model}")
    if file_hash(ROOT / d["colour_book"]) != d["colour_book_sha256"]:
        raise ValueError("census colour book changed")
    return d


def layout(smoke):
    # Clear of every earlier block: between the follow-up gates (4.11e9) and the probe (4.20e9).
    base = 4_155_000_000 if smoke else 4_120_000_000
    return dict(
        generation=1 if smoke else 56, run_root=SMOKE_ROOT if smoke else ROOT / "iterations",
        sims=8 if smoke else 3200, deep_sims=16 if smoke else 12800,
        roots=8 if smoke else 768, continuations=2, continuation_sims=8 if smoke else 6400,
        screen_games=4 if smoke else 200, probe_games=4 if smoke else 40, finalists=1 if smoke else 2,
        deep_probe_games=4 if smoke else 80, diag_games=4 if smoke else 160, self_games=4 if smoke else 200,
        colour_positions=2 if smoke else 14, colour_copies=1 if smoke else 6,
        instrument_copies=1 if smoke else 20,
        # One 1M block per match (each uses seed .. seed+500k+entries):
        # par 0 | continuations 1 | screen 2 | deep probes 3-5 | release par 6 | teacher colour 7-8 |
        # candidate colour 9-14 | gate vs teacher 15 | gate vs release 16 | instrument 17-18 | diagnostics 20-25
        par_seed=base, continuation_seed=base + 1_000_000, screen_seed=base + 2_000_000,
        deep_probe_seed=base + 3_000_000, release_par_seed=base + 6_000_000, colour_seed=base + 7_000_000,
        candidate_colour_seed=base + 9_000_000, gate_seed=base + 15_000_000, release_gate_seed=base + 16_000_000,
        instrument_seed=base + 17_000_000, diag_seed=base + 20_000_000, arms_seed=base + 30_000_000)


def recipe_path(kind, smoke):
    return ROOT / "tools/recipes" / f"gen56{kind}{'_rehearsal' if smoke else ''}.json"


def pairings(d):
    """The decision's per-colour pairings as model paths, in recipe form."""
    m = d["pairing_models"]
    return [dict(white=m[p["white"]]["model"], black=m[p["black"]]["model"]) for p in d["book_pairings"]]


def identity():
    d = decision()
    paired = [p for pair in pairings(d) for p in pair.values()]
    models = {p: file_hash(ROOT / p) for p in [d["teacher"], d["release"], *paired, *d["pool_opponents"], *HELD_OUT]}
    held = {str(Path(p)) for p in HELD_OUT}
    for smoke in (False, True):
        if read(recipe_path("_paired", smoke)).get("book_pairings") != pairings(d):
            raise ValueError("paired recipe pairings differ from the decision")
        for kind, model in [("", d["teacher"]), ("_pool", d["teacher"]), ("_paired", d["teacher"])]:
            recipe = read(recipe_path(kind, smoke))
            prior.validate_recipe(recipe)
            if recipe["model"] != model or recipe.get("temperature_plies") != 30:
                raise ValueError(f"unexpected gen56 recipe {kind} (smoke={smoke})")
            if kind != "_pool" and recipe.get("book_starts") != [dict(moves=b["moves"], weight=b["weight"])
                                                                for b in d["book_starts"]]:
                raise ValueError(f"recipe {kind} book starts differ from the decision")
            if {str(Path(p)) for p in recipe["opponents"]} & held:
                raise ValueError("held-out opponents may never be training opponents")
        if read(recipe_path("_pool", smoke))["opponents"] != d["pool_opponents"]:
            raise ValueError("pool recipe opponents differ from the decision")
    if {str(Path(p)) for p in paired} & held or str(Path(d["teacher"])) in held:
        raise ValueError("held-out models may never be teachers")
    replay = sorted((e for e in read(ROOT / "iterations/accepted_data.json")["entries"]
                     if int(e["generation"]) < 56), key=lambda e: int(e["generation"]))[-7:]
    if len(replay) != 7 or int(replay[-1]["generation"]) != 55:
        raise ValueError("expected accepted replay through gen55")
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
    recipe = recipe_path("_paired", smoke)
    raw, out = extra["teacher"]("paired")
    sources = [(f"gen_{n:04d}_paired", generated_source(campaign, smoke, "paired", recipe, raw, out,
                                                        read(recipe)["book_games"]))]
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


def colour_book(campaign, smoke, name, copies):
    """The census key positions (phase-1b order) as a book of `copies` each."""
    spec, src = layout(smoke), read(ROOT / decision()["colour_book"])
    per = len(src["entries"]) // 14   # phase 1b: 14 positions x 20 copies, position-major
    positions = [src["entries"][i * per] for i in range(spec["colour_positions"])]
    path = campaign.root / "colour" / f"{name}_book.json"
    pin_json(path, dict(schema_version=1, model=f"census key positions x {copies}", plies=0,
                        entries=[e for e in positions for _ in range(copies)]))
    return path, len(positions) * copies


def colour_match(campaign, smoke, name, a, b, book, entries, seed):
    """a vs b on a census book (8 sampled half-moves): a's White and Black scores."""
    report = campaign.root / "colour" / f"{name}.json"
    log = report.with_suffix(".jsonl")
    campaign.stage(name, ["tools/match.py", "--model-a", str(a), "--model-b", str(b), "--games", str(2 * entries),
                          "--sims", str(layout(smoke)["sims"]), "--sims-b", str(layout(smoke)["sims"]),
                          "--engine", "native", "--workers", "8", "--seed", str(seed), "--book", str(book),
                          "--book-temp-plies", "8", "--stall-timeout", "1800", "--game-log", str(log),
                          "--report-path", str(report), "--resume"], [report, log])
    rows = [json.loads(line) for line in open(log, encoding="utf-8") if '"game"' in line]
    if len(rows) != 2 * entries:
        raise ValueError(f"{name}: {len(rows)} games, expected {2 * entries}")
    def side(white):
        rs = [r for r in rows if r["a_is_white"] == white]
        return sum((r["result_for_a"] + 1) / 2 for r in rs) / len(rs)
    return dict(white=round(side(True), 4), black=round(side(False), 4), games=len(rows))


def colour_profile(campaign, smoke, label, model, book, entries, seed):
    """`model` against both references: White and Black score per reference, and their means."""
    d = decision()
    refs = [("gen55", d["teacher"]), ("v29", d["release"])]
    vs = {ref: colour_match(campaign, smoke, f"{label}_vs_{ref}", model, path, book, entries, seed + i * 1_000_000)
          for i, (ref, path) in enumerate(refs)}
    return dict(vs=vs, white=round(sum(v["white"] for v in vs.values()) / 2, 4),
                black=round(sum(v["black"] for v in vs.values()) / 2, 4))


def colour_nominee(candidates, teacher, black_tolerance):
    """Pre-declared (GEN56_PLAN.md): best screen rank among epochs passing the deep guard whose
    White is at least the teacher's and whose Black is within `black_tolerance` of the teacher's;
    else the best White."""
    def passes(c):
        return (passes_deep_guard(c) and c["colour"]["white"] >= teacher["white"] - 1e-12 and
                c["colour"]["black"] >= teacher["black"] - black_tolerance - 1e-12)
    eligible = [c for c in candidates if passes(c)]
    if eligible:
        return eligible[0]["name"], [c["name"] for c in eligible], "best_screen_rank_passing_deep_guard_and_colour_check"
    best = max(enumerate(candidates), key=lambda item: (item[1]["colour"]["white"], -item[0]))[1]
    return best["name"], [], "none_passed_both; best_colour_white"


def select_arm(campaign, smoke, model_dir, deep_par):
    spec, d = layout(smoke), decision()
    teacher = d["teacher"]
    report_path = campaign.root / "selection" / "a_screen.json"
    campaign.stage("screen_a", ["tools/checkpoint_screen.py", "--model-dir", str(model_dir),
        "--incumbent", teacher, "--output-model", str(model_dir / "screen_pick.pt"), "--report-path", str(report_path),
        "--games", str(spec["screen_games"]), "--probe-games", str(spec["probe_games"]),
        "--sims", str(spec["sims"]), "--probe-sims", str(spec["sims"]), "--finalists", str(spec["finalists"]),
        "--workers", "8", "--engine", "native", "--seed", str(spec["screen_seed"]), "--stall-timeout", "600.0"],
        [report_path, model_dir / "screen_pick.pt"])
    candidates = g51.screen_candidates(read(report_path))
    book, entries = colour_book(campaign, smoke, "selection", spec["colour_copies"])
    teacher_colour = colour_profile(campaign, smoke, "teacher", teacher, book, entries, spec["colour_seed"])
    for i, c in enumerate(candidates):
        audit = g51.match(campaign, f"deep_probe_a_{i}", c["checkpoint"], teacher, spec["deep_probe_games"],
                          spec["deep_probe_seed"] + i * 1_000_000, spec["deep_sims"], directory="selection")
        stats = audit["diagnostics"]["sampled"]
        c["deep"] = dict(score=stats["score"], sides=stats["sides"], par_comparisons=compare(stats, deep_par))
        c["colour"] = colour_profile(campaign, smoke, f"candidate_{i}", c["checkpoint"], book, entries,
                                     spec["candidate_colour_seed"] + i * 2_000_000)
    name, eligible, why = colour_nominee(candidates, teacher_colour, d["colour_check"]["black_tolerance"])
    chosen = next(c for c in candidates if c["name"] == name)
    source = Path(chosen["checkpoint"]) if Path(chosen["checkpoint"]).is_absolute() else ROOT / chosen["checkpoint"]
    target = model_dir / "arena_selected.pt"
    if target.exists() and file_hash(target) != file_hash(source):
        raise FileExistsError(f"{target} holds a different model; refusing to overwrite")
    if not target.exists():
        shutil.copy2(source, target)
    nominee = dict(arm="a", name=name, rule=why, eligible=eligible, candidates=candidates,
                   teacher_colour=teacher_colour, path=str(target), sha256=file_hash(target))
    pin_json(campaign.root / "selection" / "a_nominee.json", nominee)
    return nominee


def release_colour_baseline():
    """v29's per-colour scores on the census book, from phase 1b (the same book and sampling)."""
    pos = read(CENSUS_REPORT)["positions"]
    def total(combo, white):
        w = d = l = 0
        for cells in pos.values():
            c = cells[combo]; w += c["w"]; d += c["d"]; l += c["l"]
        n = w + d + l
        return round(((w if white else l) + 0.5 * d) / n, 4)
    # v29 as White vs gen55's Black ("v29>gen55"), vs v29's Black; v29 as Black vs gen55's White, vs v29's White.
    return dict(vs=dict(gen55=dict(white=total("v29>gen55", True), black=total("gen55>v29", False)),
                        v29=dict(white=total("v29>v29", True), black=total("v29>v29", False))))


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
    campaign.stage("position_probe", ["tools/position_probe.py", "--candidate", f"gen56={nominee['path']}",
                                      "--out", str(probe_dir)], [probe_dir / "summary.json"])
    campaign.stage("value_audit", ["tools/value_colour_audit.py", "--models", "v29",
                                   "--model", f"gen55={d['teacher']}", "--model", f"gen56={nominee['path']}",
                                   "--out", str(audit_dir)], [audit_dir / "report.json"])
    held = [diagnostics[n]["diagnostics"]["sampled"]["score"] for n in ("vs_b2", "vs_v27")]
    held_mean = sum(held) / len(held)
    eligible = vs_release["verdict"] == "PASS" and held_mean >= HELD_OUT_MIN - 1e-12
    book, entries = colour_book(campaign, smoke, "instrument", spec["instrument_copies"])
    instrument = colour_profile(campaign, smoke, "instrument", nominee["path"], book, entries, spec["instrument_seed"])
    baseline = release_colour_baseline()
    tolerance = d["proposed_promotion"]["per_colour_tolerance"]
    shortfalls = {f"{ref}_{colour}": round(instrument["vs"][ref][colour] - baseline["vs"][ref][colour], 4)
                  for ref in ("gen55", "v29") for colour in ("white", "black")}
    proposed = (all(v >= -tolerance - 1e-12 for v in shortfalls.values()) and
                held_mean >= d["proposed_promotion"]["held_out_min"] - 1e-12)
    return dict(verdict=vs_teacher["verdict"], primary=vs_teacher["primary"], guard=vs_teacher["guard"],
                legs=vs_teacher["legs"], combined_h2h=vs_teacher["combined_h2h"],
                release_gate=dict(verdict=vs_release["verdict"], primary=vs_release["primary"],
                                  guard=vs_release["guard"], legs=vs_release["legs"]),
                diagnostics=diagnostics, held_out=["vs_b2", "vs_v27"], held_out_mean=held_mean,
                instrument=dict(candidate=instrument, release_baseline=baseline, deltas_vs_release=shortfalls),
                promotion=dict(eligible=eligible, rule="docs/protocols/PROMOTION_RULE.md",
                               gate_vs_release=vs_release["verdict"], held_out_mean=held_mean,
                               held_out_min=HELD_OUT_MIN,
                               proposed_rule=dict(eligible=proposed, tolerance=tolerance, deltas=shortfalls,
                                                  source="docs/plans/GEN56_PLAN.md (owner decides)"),
                               note="eligibility only; the owner decides"))


def work(campaign, smoke):
    d = decision()
    spec = layout(smoke)
    par_dir = measure_par(campaign, smoke, "par", d["teacher"], spec["par_seed"])
    deep_par = self_par(read_rows(par_dir / "deep_par.jsonl"))
    release_par_dir = measure_par(campaign, smoke, "release_par", d["release"], spec["release_par_seed"])
    state = iteration(campaign, smoke)
    base = [] if smoke else [(s["name"], ROOT / s["path"]) for s in d.get("prior_extra_sources", [])]
    base.append((f"gen_{spec['generation']:04d}_deepvalue", deep_source(campaign, smoke)))
    model_dir = compose_and_train(campaign, smoke, state, base + extra_sources(campaign, smoke))
    nominee = select_arm(campaign, smoke, model_dir, deep_par)
    result = evaluate(campaign, smoke, nominee, par_dir, release_par_dir)
    if identity() != campaign.provenance:
        raise ValueError("inputs changed before publication")
    pin_json(campaign.root / "summary.json", dict(
        complete=True, rehearsal=smoke, decision=d, nominee=nominee, result=result,
        manifest_sha256=file_hash(campaign.root / "manifest.json"),
        notes=["Single arm: gen55 self-play, half from census book starts; per-colour teacher book games "
               "(White gen55/gen54 vs Black gen53/v29); pool.",
               "Selection adds a per-colour check against gen55 and v29; the per-colour instrument is the primary measure.",
               "Nothing promoted.", "B2 and v27 are held out (never teachers or training opponents)."]))
    atomic_json(campaign.root / "status.json", dict(status="complete"))
    print(f"GEN56 {'REHEARSAL' if smoke else 'PRODUCTION'} COMPLETE", flush=True)


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
                                identity_fn=identity, label="GEN56")
            try:
                if smoke:
                    campaign.stage("tests", ["-m", "pytest", "-p", "no:cacheprovider", "tests/test_gen56_campaign.py",
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
