"""Recover B2 after reboot: verify completed work, retrain only the deleted hybrid."""
import json
from pathlib import Path
import subprocess
import sys
import os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tools'))
from match_evidence import atomic_json, file_hash, runtime_identity
from b2_campaign import run_stage


def main():
    root = ROOT/'iterations/b2_001'
    manifest = json.loads((root/'manifest.json').read_text())
    if runtime_identity() != manifest['runtime']:
        raise ValueError('Runtime changed since original campaign; inspect before retraining')
    for name in ('prepare15', 'prepare24', 'train_control', 'train_state_cnn'):
        receipt = json.loads((root/'receipts'/f'{name}.json').read_text())
        for path, sha in receipt['artifacts'].items():
            if file_hash(path) != sha:
                raise ValueError(f'Changed completed artifact: {path}')
        print(f'Verified {name}', flush=True)
    directory = ROOT/'models/candidates/b2_001_hybrid'
    if directory.exists():
        raise FileExistsError('Hybrid directory must be absent for this fresh retrain')
    command = json.loads((root/'receipts/train_control.json').read_text())['command']
    for flag, value in (('--model-dir', str(directory)), ('--data-dir', str(root/'processed24')),
                        ('--attention-blocks', '2')):
        command[command.index(flag)+1] = value
    atomic_json(root/'hybrid_retrain_20260909.json', dict(command=command, runtime=runtime_identity(),
        implementation=file_hash(__file__), note='Fresh seed3173 retrain after Windows Update; data and other arms unchanged'))
    run_stage(root, 'train_hybrid', command, [directory])
    paths = {arm: ROOT/f'models/candidates/b2_001_{arm}/best_value_net.pt'
             for arm in ('control','state_cnn','hybrid')}
    atomic_json(root/'training_complete.json', dict(complete=True, rehearsal=False,
                paths={k:str(p) for k,p in paths.items()}, models={k:file_hash(p) for k,p in paths.items()}))
    subprocess.run([sys.executable, 'tools/b2_smoke_models.py', '--root', str(root)],
                   cwd=ROOT, env=dict(os.environ, MONSTER_PINNED_INPUT='1'), check=True)


if __name__ == '__main__':
    main()
