"""Owner-authorized release v29, September 29, 2026.

Source is gen51's deep-value arm nominee (selected_epoch_009): scratch-trained
on the gen51 recipe (teacher v28, 30-ply exploration) plus disagreement-root
deep-outcome value data. The owner reported that the models now exceed his
ability to judge strength by playtest, so the release rests on the automated
evidence below, the first under docs/protocols/PROMOTION_RULE.md.
Copies, never moves or overwrites.
"""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from match_evidence import atomic_json, file_hash

VERSION = 29
SOURCE = "models/candidates/bootstrap_main_gen_0051_deepvalue/arena_selected.pt"
SOURCE_SHA256 = "dbf26b9eff47009b25b63929be33bbf534f364599af87b233ed3a4ed313e84e2"
PREDECESSOR_SHA256 = "b651e7405afe4e5672676c6fb13bb5bc6e1f5ea0071fd5ce4f4d5318e229e35a"  # v28
RUN = "benchmarks/gen51_program/gen51_20260925/production"
EVIDENCE = [f"{RUN}/gates/deep/report.json", f"{RUN}/summary.json", f"{RUN}/selection/deep_nominee.json",
            "docs/experiments/gen51/GEN51_RESULTS.md"]


def main():
    source, destination = ROOT / SOURCE, ROOT / f"models/bootstrap_v{VERSION}/best_value_net.pt"
    if file_hash(source) != SOURCE_SHA256:
        raise ValueError(f"checkpoint identity changed: {source}")
    if file_hash(ROOT / "models/bootstrap_v28/best_value_net.pt") != PREDECESSOR_SHA256:
        raise ValueError("v28 identity changed")
    if destination.exists() and file_hash(destination) != SOURCE_SHA256:
        raise ValueError(f"release already occupied: {destination}")
    for path in EVIDENCE:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    gate = json.loads((ROOT / EVIDENCE[0]).read_text())
    if gate["verdict"] != "PASS" or gate["model"]["sha256"] != SOURCE_SHA256 or gate["bar"]["sha256"] != PREDECESSOR_SHA256:
        raise ValueError("gate v4 report does not certify this checkpoint against v28")
    manifest = dict(
        schema_version=1, generation=51, arm="deep-value (selected_epoch_009)",
        checkpoint=SOURCE, checkpoint_sha256=SOURCE_SHA256, checkpoint_bytes=source.stat().st_size,
        predecessor=f"bootstrap_v{VERSION - 1}", predecessor_sha256=PREDECESSOR_SHA256,
        evidence=EVIDENCE, evidence_sha256={p: file_hash(ROOT / p) for p in EVIDENCE},
        promotion_rule="docs/protocols/PROMOTION_RULE.md (first release under it)",
        evidence_summary={
            "gate_v4_vs_v28": "PASS: 74.1% / 75.75% at 3,200 (800 games); 85.0% at 12,800 (160 games); "
                              "colour deltas vs v28 self-par White +36.9 pp, Black +13.0 pp",
            "held_out_mean": {"v29": 0.927, "v28": 0.823,
                              "detail": "B2 99.06% vs 73.75%; v27 86.25% vs 90.94%"},
            "other": "gen49 75.6% (v28: 83.1%); beat the gen51 control nominee 67.8% at 12,800; "
                     "no later candidate (gen52 arms A/B, Arm C) passed gate v4 against it"},
        status=f"promoted_as_v{VERSION}",
        promotion=dict(automatic=False, owner_approved=True,
                       owner_instruction="Promote + adopt an evidence rule (the owner reported the models now "
                                         "exceed his ability to judge strength by playtest).",
                       promoted_at="2026-09-29",
                       promoted_as=str(destination.relative_to(ROOT)).replace("\\", "/"),
                       source_preserved=True, historical_models_overwritten=False),
        caveats=["Weaker than v28 against v27 and gen49 in single 160-game samples; stronger on the held-out mean.",
                 "Head-to-head and older-opponent results are non-transitive; no Elo ladder is implied.",
                 "Not evidence of proximity to perfect play."])
    manifest_path = destination.parent / "promotion_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"manifest already occupied: {manifest_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if file_hash(destination) != SOURCE_SHA256:
        raise ValueError("release copy verification failed")
    atomic_json(manifest_path, manifest)
    pointer_path = ROOT / "models/bootstrap/champion.json"
    backup = ROOT / f"models/bootstrap/champion_before_v{VERSION}_20260929.json"
    if pointer_path.exists() and not backup.exists():
        shutil.copy2(pointer_path, backup)
    atomic_json(pointer_path, dict(schema_version=1, generation=51,
                                  checkpoint=f"models/bootstrap_v{VERSION}/best_value_net.pt",
                                  checkpoint_sha256=SOURCE_SHA256, promoted_at="2026-09-29",
                                  promotion_manifest=f"models/bootstrap_v{VERSION}/promotion_manifest.json"))
    print(f"gen51 deep-value -> {destination.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
