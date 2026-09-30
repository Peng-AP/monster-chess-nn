"""Gen52 Arm L: Arm B's data and recipe with a 2x wider residual tower.

docs/plans/GEN52_LARGE_PLAN.md (owner go-ahead 2026-09-30). Training-only:
it reuses Arm B's composed replay, gen52's teacher par, selection and gate
code and seeds (matched to Arms A-C), then plays Arm L against Arm B and
places it on the joint Elo scale. Nothing is promoted.
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
from match_evidence import atomic_json, file_hash, read_rows, runtime_identity
from sampled_gate_stats import self_par
from start_gpu48 import Campaign, campaign_lock, pin_json, read

RUN = ROOT / "benchmarks/gen52_program/gen52_large_20260930"
BASE_TOWER = "64,64,128,128,128,128,128,128"
WIDE_TOWER = "64,64,192,192,192,192,192,192"
ARM_B_HELD_OUT_MEAN = 0.914   # GEN52_RESULTS.md: B2 87.8%, v27 95.0%
PINNED = ["tools/gen52_large_campaign.py", "tools/gen52_campaign.py", "tools/gen51_campaign.py",
          "tools/checkpoint_screen.py", "tools/gate_depth.py", "tools/match.py", "tools/elo_ladder.py",
          "tools/elo_tournament.py", "tools/free_gate_stats.py", "tools/sampled_gate_stats.py",
          "tools/free_play_audit.py", "tools/start_gpu48.py", "tests/test_gen52_large_campaign.py",
          "docs/plans/GEN52_LARGE_PLAN.md", "docs/plans/gen52_teacher_decision.json"]


def layout(smoke):
    base = 3_750_000_000 if smoke else 3_700_000_000
    spec = g52.layout(smoke)
    return dict(h2h_games=spec["arms_games"], h2h_deep_games=spec["arms_deep_games"],
                sims=spec["sims"], deep_sims=spec["deep_sims"], h2h_seed=base)


def gen52_root(smoke):
    return g52.RUN / ("rehearsal" if smoke else "production")


def arm_l_paths(smoke):
    paths, extra = g52.paths_for(smoke)
    return paths, extra, dict(model_dir=extra["b"]["model_dir"].with_name(paths["candidate_dir"].name + "_large"),
                              replay=extra["b"]["replay"])


def identity():
    d = g52.decision()
    models = {p: file_hash(ROOT / p) for p in [d["teacher"], *g52.POOL, *g52.HELD_OUT]}
    replays = {str(s): file_hash(arm_l_paths(s)[2]["replay"] / "replay_manifest.json") for s in (False, True)}
    return dict(runtime=runtime_identity(), decision=d, models=models, replays=replays,
                implementations={p: file_hash(ROOT / p) for p in PINNED},
                protocol=layout(False), rehearsal=layout(True), tower=WIDE_TOWER)


def train_command(smoke):
    """Arm B's receipted train command; only the model directory and the tower width change."""
    _, _, arm = arm_l_paths(smoke)
    receipt = read(gen52_root(smoke) / "receipts/train_b.json")
    train = list(receipt["command"])
    if train[train.index("--data-dir") + 1] != str(arm["replay"]):
        raise ValueError("Arm B's receipt names a different replay")
    if train[train.index("--res-channels") + 1] != BASE_TOWER:
        raise ValueError("Arm B's tower is not the declared base tower")
    train[train.index("--model-dir") + 1] = str(arm["model_dir"])
    train[train.index("--res-channels") + 1] = WIDE_TOWER
    return train


def train_arm_l(campaign, smoke):
    _, _, arm = arm_l_paths(smoke)
    if arm["model_dir"].exists() and not (campaign.root / "receipts/train_l.json").exists():
        raise ValueError("unreceipted Arm L training output retained; refusing automatic overwrite")
    campaign.stage("train_l", train_command(smoke), [arm["model_dir"] / "best_value_net.pt"])
    return arm["model_dir"]


def score_stats(audit):
    """Sampled score and its per-game SE, plus the endpoint-unique score, from a replay-audited match."""
    d = audit["diagnostics"]
    return dict(n=d["sampled"]["n"], score=d["sampled"]["score"], se=d["sampled"]["se"],
                unique=d["unique"]["score"], unique_n=d["unique"]["n"])


def verdict(h2h, held_out_mean):
    s = score_stats(h2h)
    lo, hi = s["score"] - 1.96 * s["se"], s["score"] + 1.96 * s["se"]
    if lo > 0.5 and s["unique"] > 0.5 and held_out_mean >= ARM_B_HELD_OUT_MEAN - 0.010:
        call = "capacity_helps"
    elif hi < 0.5:
        call = "capacity_harmful"
    else:
        call = "null"
    return dict(call=call, h2h=s, ci95=[lo, hi], held_out_mean=held_out_mean,
                arm_b_held_out_mean=ARM_B_HELD_OUT_MEAN)


def held_out_mean(result):
    scores = [result["diagnostics"][k]["diagnostics"]["sampled"]["score"] for k in ("vs_b2", "vs_v27")]
    return sum(scores) / len(scores)


def work(campaign, smoke):
    spec = layout(smoke)
    par_dir = gen52_root(smoke) / "par"
    gate_depth.validate_report(par_dir / "report.json")
    deep_par = self_par(read_rows(par_dir / "deep_par.jsonl"))
    model_dir = train_arm_l(campaign, smoke)
    nominee = g52.select_arm(campaign, smoke, "l", model_dir, deep_par)
    result = g52.evaluate_arm(campaign, smoke, "l", nominee, par_dir)
    arm_b = read(gen52_root(smoke) / "selection" / "b_nominee.json")
    if file_hash(arm_b["path"]) != arm_b["sha256"]:
        raise ValueError("Arm B nominee changed")
    h2h = {"l_vs_b": g51.match(campaign, "l_vs_b", nominee["path"], arm_b["path"], spec["h2h_games"],
                               spec["h2h_seed"], spec["sims"]),
           "l_vs_b_deep": g51.match(campaign, "l_vs_b_deep", nominee["path"], arm_b["path"],
                                    spec["h2h_deep_games"], spec["h2h_seed"] + 1_000_000, spec["deep_sims"])}
    call = verdict(h2h["l_vs_b"], held_out_mean(result))
    elo_dir = campaign.root / "elo"
    campaign.stage("elo", ["tools/elo_ladder.py", "--out", str(elo_dir),
                           "--setting", f"gen52L={Path(nominee['path']).relative_to(ROOT).as_posix()}@{spec['sims']}",
                           *(["--smoke"] if smoke else [])], [elo_dir / "ratings.json"])
    if identity() != campaign.provenance:
        raise ValueError("inputs changed before publication")
    pin_json(campaign.root / "summary.json", dict(
        complete=True, rehearsal=smoke, tower=WIDE_TOWER, base_tower=BASE_TOWER, nominee=nominee, result=result,
        arm_b_nominee=arm_b, head_to_head=h2h, verdict=call, elo=read(elo_dir / "ratings.json")["standings"],
        manifest_sha256=file_hash(campaign.root / "manifest.json"),
        notes=["Arm L = gen52 Arm B with the residual tower widened 128 -> 192.",
               "Selection, gate and diagnostic seeds match gen52's arms.", "Nothing promoted."]))
    atomic_json(campaign.root / "status.json", dict(status="complete"))
    print(f"LARGE {'REHEARSAL' if smoke else 'PRODUCTION'} COMPLETE: {call['call']}", flush=True)


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
                                identity_fn=identity, label="LARGE")
            try:
                if smoke:
                    campaign.stage("tests", ["-m", "pytest", "-p", "no:cacheprovider",
                                             "tests/test_gen52_large_campaign.py", "tests/test_gate_depth.py", "-q"])
                elif not read(RUN / "rehearsal/summary.json")["complete"]:
                    raise ValueError("rehearsal required")
                work(campaign, smoke)
            except BaseException as exc:
                atomic_json(campaign.root / "status.json", dict(status="failed", error=str(exc)))
                raise


if __name__ == "__main__":
    main()
