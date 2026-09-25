"""Owner-authorized release v28, September 25, 2026.

Source is gen50 epoch14 with the frozen-policy continuation value head fitted
on September 17 (`docs/experiments/value_calibration/`). Only the six
`value_head.*` tensors differ from gen50 epoch14; the fit artifacts record
bit-identical backbone, policy and buffers. Copies, never moves or overwrites.
"""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from match_evidence import atomic_json, file_hash

VERSION = 28
SOURCE = 'benchmarks/value_calibration_20260917/production/fits/continuation/candidate.pt'
SOURCE_SHA256 = 'b651e7405afe4e5672676c6fb13bb5bc6e1f5ea0071fd5ce4f4d5318e229e35a'
INITIALIZATION = 'models/candidates/bootstrap_main_gen_0050/arena_selected.pt'
INITIALIZATION_SHA256 = '51b5ddb01db51ae9023eaaf8ccbd896b48a805a52b2707633dc1d7e3f8067f25'
EVIDENCE = [
    'benchmarks/value_calibration_20260917/production/confirmation/play/vs_initial/report.json',
    'benchmarks/value_calibration_20260917/production/summary.json',
    'benchmarks/value_calibration_20260917/production/nominee.json',
    'benchmarks/value_calibration_20260917/production/fits/continuation/complete.json',
    'benchmarks/gen50_20260916/production/summary.json',
]


def main():
    source = ROOT / SOURCE
    destination = ROOT / f'models/bootstrap_v{VERSION}/best_value_net.pt'
    if file_hash(source) != SOURCE_SHA256:
        raise ValueError(f'checkpoint identity changed: {source}')
    if file_hash(ROOT / INITIALIZATION) != INITIALIZATION_SHA256:
        raise ValueError('gen50 epoch14 initialization identity changed')
    if destination.exists() and file_hash(destination) != SOURCE_SHA256:
        raise ValueError(f'release already occupied: {destination}')
    for path in EVIDENCE:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    gate = json.loads((ROOT / EVIDENCE[0]).read_text())
    if gate['verdict'] != 'PASS' or not gate['confirmed'] or gate['model']['sha256'] != SOURCE_SHA256:
        raise ValueError('calibration gate report does not certify this checkpoint')
    manifest = dict(
        schema_version=1, generation=50,
        checkpoint=SOURCE, checkpoint_sha256=SOURCE_SHA256,
        checkpoint_bytes=source.stat().st_size,
        initialization=INITIALIZATION, initialization_sha256=INITIALIZATION_SHA256,
        change_from_initialization='value_head.* only (frozen backbone/policy/buffers)',
        predecessor=f'bootstrap_v{VERSION - 1}',
        evidence=EVIDENCE,
        evidence_sha256={p: file_hash(ROOT / p) for p in EVIDENCE},
        status=f'promoted_as_v{VERSION}',
        promotion=dict(automatic=False, owner_approved=True,
                       owner_instruction='We should promote a new v from either 49 or 50; '
                                         'chose gen50 + calibrated value.',
                       promoted_at='2026-09-25',
                       promoted_as=str(destination.relative_to(ROOT)).replace('\\', '/'),
                       source_preserved=True, historical_models_overwritten=False),
        caveats=[
            'Sampled gate vs gen50 epoch14 at 3,200 sims: PASS confirmed, 58.44% over 800; '
            'White +14.0pp / Black +2.9pp vs epoch14 self-par.',
            'At 12,800 sims vs epoch14: 49.375% over 160 games (no deep-search gain shown).',
            'Endpoint-unique H2H score is 51.19% over 157 openings; repertoire is concentrated.',
            'vs gen49 83.1% @3,200 and 56.9% @12,800; vs v27 90.9%; vs B2 73.75% (160 games each).',
            'No transitive Elo claim; not evidence of proximity to perfect play.',
        ])
    manifest_path = destination.parent / 'promotion_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f'manifest already occupied: {manifest_path}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if file_hash(destination) != SOURCE_SHA256:
        raise ValueError('release copy verification failed')
    atomic_json(manifest_path, manifest)
    pointer_path = ROOT / 'models/bootstrap/champion.json'
    backup = ROOT / f'models/bootstrap/champion_before_v{VERSION}_20260925.json'
    if pointer_path.exists() and not backup.exists():
        shutil.copy2(pointer_path, backup)
    atomic_json(pointer_path, dict(schema_version=1, generation=50,
                                  checkpoint=f'models/bootstrap_v{VERSION}/best_value_net.pt',
                                  checkpoint_sha256=SOURCE_SHA256,
                                  promoted_at='2026-09-25',
                                  promotion_manifest=f'models/bootstrap_v{VERSION}/promotion_manifest.json'))
    print(f'gen50+calibrated value -> {destination.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
