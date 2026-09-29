"""Gen52 Arm C: Arm B's data minus the rolled-forward gen51 deep-value source.

docs/plans/GEN52_POOLCAP_PLAN.md (owner go-ahead 2026-09-29). Training-only:
it reuses gen52's completed iteration, increments, teacher par, selection and
evaluation code and seeds (matched to Arm B), then plays Arm C against Arm B.
Nothing is promoted.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import numpy as np

import gate_depth
import gen51_campaign as g51
import gen52_campaign as g52
from match_evidence import atomic_json, file_hash, read_rows, runtime_identity
from sampled_gate_stats import self_par
from start_gpu48 import Campaign, campaign_lock, pin_json, read

RUN = ROOT / "benchmarks/gen52_program/gen52_poolcap_20260929"
DROPPED = "gen_0051_deepvalue"
PINNED = ["tools/gen52_poolcap_campaign.py", "tools/gen52_campaign.py", "tools/gen51_campaign.py",
          "tools/compose_processed.py", "tools/checkpoint_screen.py", "tools/gate_depth.py", "tools/match.py",
          "tools/free_gate_stats.py", "tools/sampled_gate_stats.py", "tools/free_play_audit.py",
          "tools/start_gpu48.py", "tests/test_gen52_poolcap_campaign.py", "docs/plans/GEN52_POOLCAP_PLAN.md",
          "docs/plans/gen52_teacher_decision.json"]


def layout(smoke):
    base = 3_350_000_000 if smoke else 3_300_000_000
    spec = g52.layout(smoke)
    return dict(h2h_games=spec["arms_games"], h2h_deep_games=spec["arms_deep_games"],
                sims=spec["sims"], deep_sims=spec["deep_sims"], h2h_seed=base)


def gen52_root(smoke):
    return g52.RUN / ("rehearsal" if smoke else "production")


def identity():
    d = g52.decision()
    models = {p: file_hash(ROOT / p) for p in [d["teacher"], *g52.POOL, *g52.HELD_OUT]}
    return dict(runtime=runtime_identity(), decision=d, models=models,
                implementations={p: file_hash(ROOT / p) for p in PINNED},
                protocol=layout(False), rehearsal=layout(True), dropped=DROPPED)


def arm_c_paths(smoke):
    paths, extra = g52.paths_for(smoke)
    n, namespace = g52.layout(smoke)["generation"], __import__("iterate")._run_namespace(g52.layout(smoke)["run_root"])
    return paths, extra, dict(model_dir=paths["candidate_dir"].with_name(paths["candidate_dir"].name + "_poolcap"),
                              replay=ROOT / "data/processed" / f"bootstrap_replay_{namespace}_gen_{n:04d}_armC")


def sources(smoke):
    """Arm B's extra sources, minus the rolled-forward gen51 deep-value increment."""
    _, extra, _ = arm_c_paths(smoke)
    n = g52.layout(smoke)["generation"]
    return [(f"gen_{n:04d}_deepvalue", extra["deep"]), (f"gen_{n:04d}_pool", extra["pool"])]


def deep_share(replay):
    """Share of training value weight carried by deep-value sources, from the composed replay."""
    m = read(Path(replay) / "replay_manifest.json")
    vw = np.load(Path(replay) / "value_weights.npy", mmap_mode="r")
    train = np.load(Path(replay) / "splits.npz")["train"]
    weights = np.asarray(vw)[train]
    total, off, share = float(weights.sum()), 0, {}
    for s in m["sources"]:
        mask = (train >= off) & (train < off + s["rows"])
        if "deepvalue" in s["name"]:
            share[s["name"]] = float(weights[mask].sum()) / total
        off += s["rows"]
    return dict(by_source=share, total=sum(share.values()), sources=[s["name"] for s in m["sources"]])


def train_arm_c(campaign, smoke):
    paths, _, arm = arm_c_paths(smoke)
    state = read(paths["state"])
    compose = list(state["phases"]["compose"]["commands"][0])
    compose[compose.index("--output-dir") + 1] = str(arm["replay"])
    for name, path in sources(smoke):
        compose += ["--source", f"{name}={path}"]
    campaign.stage("compose_c", compose, [arm["replay"] / "replay_manifest.json"])
    share = deep_share(arm["replay"])
    if DROPPED in share["sources"]:
        raise ValueError("the dropped source is still in Arm C's replay")
    pin_json(campaign.root / "deep_share.json", share)
    template = read(g51.arm_paths(smoke)[0]["state"])["phases"]["train"]["commands"][0]
    train = list(template)
    for flag, value in (("--data-dir", arm["replay"]), ("--model-dir", arm["model_dir"])):
        train[train.index(flag) + 1] = str(value)
    if arm["model_dir"].exists() and not (campaign.root / "receipts/train_c.json").exists():
        raise ValueError("unreceipted Arm C training output retained; refusing automatic overwrite")
    campaign.stage("train_c", train, [arm["model_dir"] / "best_value_net.pt"])
    return arm["model_dir"]


def work(campaign, smoke):
    spec = layout(smoke)
    par_dir = gen52_root(smoke) / "par"
    gate_depth.validate_report(par_dir / "report.json")
    deep_par = self_par(read_rows(par_dir / "deep_par.jsonl"))
    model_dir = train_arm_c(campaign, smoke)
    nominee = g52.select_arm(campaign, smoke, "c", model_dir, deep_par)
    result = g52.evaluate_arm(campaign, smoke, "c", nominee, par_dir)
    arm_b = read(gen52_root(smoke) / "selection" / "b_nominee.json")
    if file_hash(arm_b["path"]) != arm_b["sha256"]:
        raise ValueError("Arm B nominee changed")
    h2h = {"c_vs_b": g51.match(campaign, "c_vs_b", nominee["path"], arm_b["path"], spec["h2h_games"],
                               spec["h2h_seed"], spec["sims"]),
           "c_vs_b_deep": g51.match(campaign, "c_vs_b_deep", nominee["path"], arm_b["path"],
                                    spec["h2h_deep_games"], spec["h2h_seed"] + 1_000_000, spec["deep_sims"])}
    if identity() != campaign.provenance:
        raise ValueError("inputs changed before publication")
    pin_json(campaign.root / "summary.json", dict(
        complete=True, rehearsal=smoke, deep_share=read(campaign.root / "deep_share.json"), nominee=nominee,
        result=result, arm_b_nominee=arm_b, head_to_head=h2h, manifest_sha256=file_hash(campaign.root / "manifest.json"),
        notes=["Arm C = gen52 Arm B minus the rolled-forward gen51 deep-value source.",
               "Selection, gate and diagnostic seeds match gen52's arms.", "Nothing promoted."]))
    atomic_json(campaign.root / "status.json", dict(status="complete"))
    print(f"POOLCAP {'REHEARSAL' if smoke else 'PRODUCTION'} COMPLETE", flush=True)


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
                                identity_fn=identity, label="POOLCAP")
            try:
                if smoke:
                    campaign.stage("tests", ["-m", "pytest", "-p", "no:cacheprovider",
                                             "tests/test_gen52_poolcap_campaign.py", "tests/test_gate_depth.py", "-q"])
                elif not read(RUN / "rehearsal/summary.json")["complete"]:
                    raise ValueError("rehearsal required")
                work(campaign, smoke)
            except BaseException as exc:
                atomic_json(campaign.root / "status.json", dict(status="failed", error=str(exc)))
                raise


if __name__ == "__main__":
    main()
