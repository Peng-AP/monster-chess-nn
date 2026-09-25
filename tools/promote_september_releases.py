"""Owner-authorized immutable milestone releases, September 7, 2026."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from match_evidence import atomic_json, file_hash

RELEASES = [
    (25, 42, 'screen_nominee.pt', '035324932307273545e2a47545899d09a2dbb5b2965bfbab04aeaf01de86e15a'),
    (26, 45, 'arena_selected.pt', '8ed079aea313ffea79d4f2dd9879d62692d5a49915245dbc5dd86ce8f68a3c86'),
    (27, 46, 'arena_selected.pt', '976294daf7e3d6f0c51c358dd602f11997c7fdf2dc4b255b810b588c253e5459'),
]


def main():
    plan = []
    for version, generation, filename, expected in RELEASES:
        source = ROOT / f'models/candidates/bootstrap_main_gen_{generation:04d}' / filename
        destination = ROOT / f'models/bootstrap_v{version}/best_value_net.pt'
        if file_hash(source) != expected:
            raise ValueError(f'checkpoint identity changed: {source}')
        if destination.exists() and file_hash(destination) != expected:
            raise ValueError(f'release already occupied: {destination}')
        evidence = ([f'benchmarks/tournament/v24_vs_gen42_free_3200.json',
                     f'benchmarks/tournament/v24_vs_gen42_book_3200.json'] if generation == 42 else
                    [f'iterations/gen_{generation:04d}/reports/binding_gate.json',
                     f'iterations/gen_{generation:04d}/reports/self_skew.json'])
        for path in evidence:
            if not (ROOT / path).is_file():
                raise FileNotFoundError(path)
        if generation != 42:
            state = json.loads((ROOT / f'iterations/gen_{generation:04d}/state.json').read_text())
            if state['status'] != 'passed_not_promoted':
                raise ValueError(f'gen{generation} has not passed')
        manifest = dict(schema_version=1, generation=generation,
                        checkpoint=str(source.relative_to(ROOT)).replace('\\', '/'),
                        checkpoint_sha256=expected, checkpoint_bytes=source.stat().st_size,
                        predecessor=f'bootstrap_v{version-1}', evidence=evidence,
                        evidence_sha256={p: file_hash(ROOT / p) for p in evidence},
                        status=f'promoted_as_v{version}',
                        promotion=dict(automatic=False, owner_approved=True,
                                       owner_instruction='Promote 3 releases. Which 3 are up to you.',
                                       promoted_at='2026-09-07',
                                       promoted_as=str(destination.relative_to(ROOT)).replace('\\', '/'),
                                       source_preserved=True, historical_models_overwritten=False),
                        caveats=['Milestone release authorized by owner, not a new gate run.',
                                 'Gen42 uses tournament evidence, not a retroactively fabricated gate PASS.',
                                 'Gen45 binding opponent is gen44; no transitive Elo claim.',
                                 'Gen46 broader follow-up is separate and may still be running.'])
        manifest_path = destination.parent / 'promotion_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f'manifest already occupied: {manifest_path}')
        plan.append((source, destination, manifest_path, manifest))
    for source, destination, manifest_path, manifest in plan:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        if file_hash(destination) != manifest['checkpoint_sha256']:
            raise ValueError('release copy verification failed')
        atomic_json(manifest_path, manifest)
        print(f"gen{manifest['generation']} -> {destination.relative_to(ROOT)}", flush=True)
    pointer_path = ROOT / 'models/bootstrap/champion.json'
    backup = ROOT / 'models/bootstrap/champion_before_v27_20260907.json'
    if pointer_path.exists() and not backup.exists():
        shutil.copy2(pointer_path, backup)
    atomic_json(pointer_path, dict(schema_version=1, generation=46,
                                  checkpoint='models/bootstrap_v27/best_value_net.pt',
                                  checkpoint_sha256=RELEASES[-1][3],
                                  promoted_at='2026-09-07',
                                  promotion_manifest='models/bootstrap_v27/promotion_manifest.json'))


if __name__ == '__main__':
    main()
