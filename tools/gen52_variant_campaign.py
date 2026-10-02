"""Gen52 follow-up variants of Arm L (docs/plans/OVERNIGHT_20261002_PLAN.md step 2).

  --variant lr  Arm L's wide tower on Arm R's relabelled replay (do the two stack?)
  --variant l2  Arm L repeated with training seed 3174 (does the capacity result replicate?)

Each is Arm L's receipted train command with one declared change, then the
same selection, gate, diagnostics and Elo placement as Arms L and R, a primary
head-to-head against its comparator and a secondary one. Nothing is promoted.
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
import gen52_ramp_campaign as ramp
from match_evidence import atomic_json, file_hash, read_rows, runtime_identity
from sampled_gate_stats import self_par
from start_gpu48 import Campaign, campaign_lock, pin_json, read

PINNED = ["tools/gen52_variant_campaign.py", "tools/gen52_large_campaign.py", "tools/gen52_ramp_campaign.py",
          "tools/gen52_campaign.py", "tools/gen51_campaign.py", "tools/checkpoint_screen.py", "tools/gate_depth.py",
          "tools/match.py", "tools/elo_ladder.py", "tools/elo_tournament.py", "tools/free_gate_stats.py",
          "tools/sampled_gate_stats.py", "tools/free_play_audit.py", "tools/start_gpu48.py",
          "tests/test_gen52_variant_campaign.py", "docs/plans/OVERNIGHT_20261002_PLAN.md",
          "docs/plans/gen52_teacher_decision.json"]
# Held-out means (B2, v27) of the comparators: GEN52_RESULTS.md (Arm B) and LARGE_RESULTS.md (Arm L).
HELD_OUT = {"b": 0.914, "l": 0.925}
VARIANTS = {
    "lr": dict(run="gen52_largeramp_20261002", suffix="_large_ramp", primary="l", secondary="r",
               seeds={False: 3_940_000_000, True: 3_960_000_000}),
    "l2": dict(run="gen52_large_s2_20261002", suffix="_large_s2", primary="b", secondary="l",
               seeds={False: 3_950_000_000, True: 3_970_000_000}),
}


def layout(variant, smoke):
    spec = g52.layout(smoke)
    return dict(h2h_games=spec["arms_games"], h2h_deep_games=spec["arms_deep_games"], sims=spec["sims"],
                deep_sims=spec["deep_sims"], h2h_seed=VARIANTS[variant]["seeds"][smoke])


def root(smoke, campaign_dir):
    return campaign_dir / ("rehearsal" if smoke else "production")


def model_dir(variant, smoke):
    paths, _ = g52.paths_for(smoke)
    return paths["candidate_dir"].with_name(paths["candidate_dir"].name + VARIANTS[variant]["suffix"])


def train_command(variant, smoke):
    cmd = list(read(root(smoke, large.RUN) / "receipts/train_l.json")["command"])
    if cmd[cmd.index("--res-channels") + 1] != large.WIDE_TOWER:
        raise ValueError("Arm L's receipt is not the wide tower")
    cmd[cmd.index("--model-dir") + 1] = str(model_dir(variant, smoke))
    if variant == "lr":
        cmd[cmd.index("--data-dir") + 1] = str(ramp.arm_r_paths(smoke)[2]["replay"])
    else:
        seed = cmd.index("--seed") + 1
        cmd[seed] = str(int(cmd[seed]) + 1)
    return cmd


def nominee_of(arm, smoke):
    """Comparator nominees: Arm B from gen52, Arm L and Arm R from their campaigns."""
    path = {"b": root(smoke, g52.RUN) / "selection/b_nominee.json",
            "l": root(smoke, large.RUN) / "selection/l_nominee.json",
            "r": root(smoke, ramp.RUN) / "selection/r_nominee.json"}[arm]
    nominee = read(path)
    if file_hash(nominee["path"]) != nominee["sha256"]:
        raise ValueError(f"Arm {arm.upper()} nominee changed")
    return nominee


def verdict(audit, held_out_mean, comparator_held_out):
    s = large.score_stats(audit)
    lo, hi = s["score"] - 1.96 * s["se"], s["score"] + 1.96 * s["se"]
    if lo > 0.5 and s["unique"] > 0.5 and held_out_mean >= comparator_held_out - 0.010:
        call = "helps"
    elif hi < 0.5:
        call = "hurts"
    else:
        call = "null"
    return dict(call=call, h2h=s, ci95=[lo, hi], held_out_mean=held_out_mean, comparator_held_out=comparator_held_out)


def identity(variant):
    d = g52.decision()
    receipts = {str(s): file_hash(root(s, large.RUN) / "receipts/train_l.json") for s in (False, True)}
    return dict(runtime=runtime_identity(), decision=d, variant=variant, parent_receipts=receipts,
                models={p: file_hash(ROOT / p) for p in [d["teacher"], *g52.POOL, *g52.HELD_OUT]},
                implementations={p: file_hash(ROOT / p) for p in PINNED},
                protocol=layout(variant, False), rehearsal=layout(variant, True))


def work(campaign, variant, smoke):
    v, spec = VARIANTS[variant], layout(variant, smoke)
    par_dir = root(smoke, g52.RUN) / "par"
    gate_depth.validate_report(par_dir / "report.json")
    deep_par = self_par(read_rows(par_dir / "deep_par.jsonl"))
    out_dir = model_dir(variant, smoke)
    if out_dir.exists() and not (campaign.root / f"receipts/train_{variant}.json").exists():
        raise ValueError(f"unreceipted {variant} training output retained; refusing automatic overwrite")
    campaign.stage(f"train_{variant}", train_command(variant, smoke), [out_dir / "best_value_net.pt"])
    nominee = g52.select_arm(campaign, smoke, variant, out_dir, deep_par)
    result = g52.evaluate_arm(campaign, smoke, variant, nominee, par_dir)
    primary, secondary = nominee_of(v["primary"], smoke), nominee_of(v["secondary"], smoke)
    p, s = v["primary"], v["secondary"]
    h2h = {f"{variant}_vs_{p}": g51.match(campaign, f"{variant}_vs_{p}", nominee["path"], primary["path"],
                                          spec["h2h_games"], spec["h2h_seed"], spec["sims"]),
           f"{variant}_vs_{p}_deep": g51.match(campaign, f"{variant}_vs_{p}_deep", nominee["path"], primary["path"],
                                               spec["h2h_deep_games"], spec["h2h_seed"] + 1_000_000, spec["deep_sims"]),
           f"{variant}_vs_{s}": g51.match(campaign, f"{variant}_vs_{s}", nominee["path"], secondary["path"],
                                          spec["h2h_games"], spec["h2h_seed"] + 2_000_000, spec["sims"])}
    call = verdict(h2h[f"{variant}_vs_{p}"], large.held_out_mean(result), HELD_OUT[p])
    elo_dir = campaign.root / "elo"
    campaign.stage("elo", ["tools/elo_ladder.py", "--out", str(elo_dir), "--setting",
                           f"gen52{variant.upper()}={Path(nominee['path']).relative_to(ROOT).as_posix()}@{spec['sims']}",
                           *(["--smoke"] if smoke else [])], [elo_dir / "ratings.json"])
    if identity(variant) != campaign.provenance:
        raise ValueError("inputs changed before publication")
    pin_json(campaign.root / "summary.json", dict(
        complete=True, rehearsal=smoke, variant=variant, nominee=nominee, result=result, head_to_head=h2h,
        primary_comparator=primary, secondary_comparator=secondary, verdict=call,
        elo=read(elo_dir / "ratings.json")["standings"], manifest_sha256=file_hash(campaign.root / "manifest.json"),
        notes=[f"Variant {variant}: see docs/plans/OVERNIGHT_20261002_PLAN.md.", "Nothing promoted."]))
    atomic_json(campaign.root / "status.json", dict(status="complete"))
    print(f"VARIANT {variant.upper()} {'REHEARSAL' if smoke else 'PRODUCTION'} COMPLETE: {call['call']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--rehearsal-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    os.environ["MONSTER_PINNED_INPUT"] = "1"
    run = ROOT / "benchmarks/gen52_program" / VARIANTS[args.variant]["run"]
    provenance = identity(args.variant)
    with campaign_lock(run / "campaign.lock"):
        for smoke in (True, False):
            if not smoke and args.rehearsal_only:
                break
            campaign = Campaign(run / ("rehearsal" if smoke else "production"), provenance,
                                identity_fn=lambda: identity(args.variant), label=f"VARIANT {args.variant.upper()}")
            try:
                if smoke:
                    campaign.stage("tests", ["-m", "pytest", "-p", "no:cacheprovider",
                                             "tests/test_gen52_variant_campaign.py", "tests/test_gate_depth.py", "-q"])
                elif not read(run / "rehearsal/summary.json")["complete"]:
                    raise ValueError("rehearsal required")
                work(campaign, args.variant, smoke)
            except BaseException as exc:
                atomic_json(campaign.root / "status.json", dict(status="failed", error=str(exc)))
                raise


if __name__ == "__main__":
    main()
