"""Gen52 Arm R: Arm B with its deep-value increments on the main game-result labels.

docs/plans/GEN52_RAMP_PLAN.md (owner go-ahead 2026-10-01). Training-only: the
deep-value increments are rebuilt from their raw continuation games with the
ramped labels every other source uses (tools/process_linked_extra_ramped.py),
Arm B's receipted compose and train commands run with only those sources and
the directories swapped, and evaluation reuses gen52's selection/gate code and
seeds plus Arm L's verdict rule. Nothing is promoted.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import gate_depth
import gen51_campaign as g51
import gen52_campaign as g52
import gen52_large_campaign as large
from gen52_poolcap_campaign import deep_share
from match_evidence import atomic_json, file_hash, read_rows, runtime_identity
from sampled_gate_stats import self_par
from start_gpu48 import Campaign, campaign_lock, pin_json, read

RUN = ROOT / "benchmarks/gen52_program/gen52_ramp_20261001"
PINNED = ["tools/gen52_ramp_campaign.py", "tools/gen52_large_campaign.py", "tools/gen52_poolcap_campaign.py",
          "tools/gen52_campaign.py", "tools/gen51_campaign.py", "tools/process_linked_extra_ramped.py",
          "tools/process_linked_extra.py", "tools/compose_processed.py", "tools/checkpoint_screen.py",
          "tools/gate_depth.py", "tools/match.py", "tools/elo_ladder.py", "tools/elo_tournament.py",
          "tools/value_colour_audit.py", "tools/free_gate_stats.py", "tools/sampled_gate_stats.py",
          "tools/free_play_audit.py", "tools/start_gpu48.py", "tests/test_gen52_ramp_campaign.py",
          "docs/plans/GEN52_RAMP_PLAN.md", "docs/plans/gen52_teacher_decision.json"]


def layout(smoke):
    base = 3_850_000_000 if smoke else 3_800_000_000
    spec = g52.layout(smoke)
    return dict(h2h_games=spec["arms_games"], h2h_deep_games=spec["arms_deep_games"],
                sims=spec["sims"], deep_sims=spec["deep_sims"], h2h_seed=base)


def gen52_root(smoke):
    return g52.RUN / ("rehearsal" if smoke else "production")


def arm_r_paths(smoke):
    paths, extra = g52.paths_for(smoke)
    replay_b = extra["b"]["replay"]
    return paths, extra, dict(model_dir=extra["b"]["model_dir"].with_name(paths["candidate_dir"].name + "_ramp"),
                              replay=replay_b.with_name(replay_b.name[:-len("armB")] + "armR"))


def ramped(path):
    path = Path(path)
    return path.with_name(path.name + "_ramped")


def deep_sources(smoke):
    """(name, original increment) for every deep-value source in Arm B's receipted composition."""
    cmd = read(gen52_root(smoke) / "receipts/compose_b.json")["command"]
    out = []
    for i, token in enumerate(cmd):
        if token == "--source":
            name, path = cmd[i + 1].split("=", 1)
            if name.endswith("_deepvalue"):
                out.append((name, Path(path)))
    if not out:
        raise ValueError("Arm B's composition has no deep-value source")
    return out


def compose_command(smoke):
    _, _, arm = arm_r_paths(smoke)
    cmd = list(read(gen52_root(smoke) / "receipts/compose_b.json")["command"])
    swap = {f"{n}={p}": f"{n}={ramped(p)}" for n, p in deep_sources(smoke)}
    cmd = [swap.get(t, t) for t in cmd]
    cmd[cmd.index("--output-dir") + 1] = str(arm["replay"])
    return cmd


def train_command(smoke):
    _, _, arm = arm_r_paths(smoke)
    cmd = list(read(gen52_root(smoke) / "receipts/train_b.json")["command"])
    if cmd[cmd.index("--res-channels") + 1] != large.BASE_TOWER:
        raise ValueError("Arm B's tower is not the declared base tower")
    cmd[cmd.index("--data-dir") + 1] = str(arm["replay"])
    cmd[cmd.index("--model-dir") + 1] = str(arm["model_dir"])
    return cmd


def identity():
    d = g52.decision()
    models = {p: file_hash(ROOT / p) for p in [d["teacher"], *g52.POOL, *g52.HELD_OUT]}
    originals = {str(p): file_hash(p / "derivation.json") for s in (False, True) for _, p in deep_sources(s)}
    return dict(runtime=runtime_identity(), decision=d, models=models, deep_originals=originals,
                implementations={p: file_hash(ROOT / p) for p in PINNED},
                protocol=layout(False), rehearsal=layout(True))


def build_data(campaign, smoke):
    _, _, arm = arm_r_paths(smoke)
    for name, original in deep_sources(smoke):
        campaign.stage(f"relabel_{name}", ["tools/process_linked_extra_ramped.py", "--original-increment",
                                           str(original), "--output-dir", str(ramped(original))],
                       [ramped(original) / "derivation.json"])
    campaign.stage("compose_r", compose_command(smoke), [arm["replay"] / "replay_manifest.json"])
    pin_json(campaign.root / "deep_share.json", deep_share(arm["replay"]))


def train_arm_r(campaign, smoke):
    _, _, arm = arm_r_paths(smoke)
    if arm["model_dir"].exists() and not (campaign.root / "receipts/train_r.json").exists():
        raise ValueError("unreceipted Arm R training output retained; refusing automatic overwrite")
    campaign.stage("train_r", train_command(smoke), [arm["model_dir"] / "best_value_net.pt"])
    return arm["model_dir"]


def work(campaign, smoke):
    spec = layout(smoke)
    par_dir = gen52_root(smoke) / "par"
    gate_depth.validate_report(par_dir / "report.json")
    deep_par = self_par(read_rows(par_dir / "deep_par.jsonl"))
    build_data(campaign, smoke)
    model_dir = train_arm_r(campaign, smoke)
    nominee = g52.select_arm(campaign, smoke, "r", model_dir, deep_par)
    arm_b = read(gen52_root(smoke) / "selection" / "b_nominee.json")
    if file_hash(arm_b["path"]) != arm_b["sha256"]:
        raise ValueError("Arm B nominee changed")
    audit_dir = campaign.root / "audit"
    campaign.stage("audit", ["tools/value_colour_audit.py", "--models", "v29", "--model", f"gen52B={arm_b['path']}",
                             "--model", f"gen52R={nominee['path']}", "--out", str(audit_dir)],
                   [audit_dir / "report.json"])
    result = g52.evaluate_arm(campaign, smoke, "r", nominee, par_dir)
    h2h = {"r_vs_b": g51.match(campaign, "r_vs_b", nominee["path"], arm_b["path"], spec["h2h_games"],
                               spec["h2h_seed"], spec["sims"]),
           "r_vs_b_deep": g51.match(campaign, "r_vs_b_deep", nominee["path"], arm_b["path"],
                                    spec["h2h_deep_games"], spec["h2h_seed"] + 1_000_000, spec["deep_sims"])}
    call = large.verdict(h2h["r_vs_b"], large.held_out_mean(result))
    elo_dir = campaign.root / "elo"
    campaign.stage("elo", ["tools/elo_ladder.py", "--out", str(elo_dir),
                           "--setting", f"gen52R={Path(nominee['path']).relative_to(ROOT).as_posix()}@{spec['sims']}",
                           *(["--smoke"] if smoke else [])], [elo_dir / "ratings.json"])
    if identity() != campaign.provenance:
        raise ValueError("inputs changed before publication")
    audit = read(audit_dir / "report.json")["models"]
    pin_json(campaign.root / "summary.json", dict(
        complete=True, rehearsal=smoke, deep_share=read(campaign.root / "deep_share.json"), nominee=nominee,
        result=result, arm_b_nominee=arm_b, head_to_head=h2h, verdict=call,
        audit={m: {k: audit[m][k] for k in ("all", "strong_games")} for m in audit},
        elo=read(elo_dir / "ratings.json")["standings"], manifest_sha256=file_hash(campaign.root / "manifest.json"),
        notes=["Arm R = gen52 Arm B with deep-value increments on the main ramped game-result labels.",
               "Selection, gate and diagnostic seeds match gen52's arms; verdict rule identical to Arm L's.",
               "Nothing promoted."]))
    atomic_json(campaign.root / "status.json", dict(status="complete"))
    print(f"RAMP {'REHEARSAL' if smoke else 'PRODUCTION'} COMPLETE: {call['call']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
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
                                identity_fn=identity, label="RAMP")
            try:
                if smoke:
                    campaign.stage("tests", ["-m", "pytest", "-p", "no:cacheprovider",
                                             "tests/test_gen52_ramp_campaign.py", "tests/test_gate_depth.py", "-q"])
                elif not read(RUN / "rehearsal/summary.json")["complete"]:
                    raise ValueError("rehearsal required")
                work(campaign, smoke)
            except BaseException as exc:
                atomic_json(campaign.root / "status.json", dict(status="failed", error=str(exc)))
                raise


if __name__ == "__main__":
    main()
