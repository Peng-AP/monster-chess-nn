"""Sequential, fail-closed compression control and timed screens; no promotion."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from match_evidence import atomic_json, file_hash


def main():
    prior = ROOT/'benchmarks/search_first_20260911/exact_frontier_300ms/complete.json'
    if not prior.exists() or json.loads(prior.read_text())['games'] != 16:
        raise RuntimeError('Required exact-frontier 16-game screen did not finish')
    out = ROOT/'benchmarks/search_first_20260911/compression_control'
    out.mkdir(parents=True, exist_ok=False)
    paths = ['native/monster_native.pyd', 'tools/train_search_value.py',
             'tools/distill_search_value.py', 'tools/search_first_match.py', __file__]
    hashes = {p:file_hash(p) for p in paths}
    atomic_json(out/'manifest.json', dict(runtime=hashes, prerequisite=file_hash(prior),
        purpose='same 128/32 architecture, frozen gen47 teacher targets vs original outcome targets'))

    def stage(name, args):
        if any(file_hash(p)!=h for p,h in hashes.items()):
            raise RuntimeError('Source/runtime changed during locked comparison')
        atomic_json(out/'status.json', dict(stage=name, complete=False))
        print(f'STAGE {name}', flush=True)
        subprocess.run([sys.executable, *args], cwd=ROOT, check=True)

    try:
        stage('tests', ['-m','pytest','tests','-q'])
        stage('teacher_labels', ['tools/distill_search_value.py', '--out',str(out/'labels')])
        stage('train_student', ['tools/train_search_value.py', '--targets',str(out/'labels/teacher_values.npy'),
                               '--out','models/candidates/search_first_distilled_001'])
        model_dir = ROOT/'models/candidates/search_first_distilled_001'
        receipt = json.loads((model_dir/'complete.json').read_text())
        value = model_dir/f"epoch_{receipt['best_epoch']:03}.bin"
        if file_hash(value) != receipt['model_sha256']:
            raise RuntimeError('Student binary hash mismatch')
        stage('distilled_300ms', ['tools/search_first_match.py','--value',str(value),
              '--out',str(out/'distilled_300ms'),'--pairs','8','--seconds','.3'])
        stage('outcome_2s', ['tools/search_first_match.py','--value','models/candidates/search_first_001/epoch_002.bin',
              '--out',str(out/'outcome_2s'),'--pairs','4','--seconds','2'])
        stage('distilled_2s', ['tools/search_first_match.py','--value',str(value),
              '--out',str(out/'distilled_2s'),'--pairs','4','--seconds','2'])
        atomic_json(out/'status.json', dict(stage='complete',complete=True))
    except BaseException as exc:
        atomic_json(out/'failure.json', dict(error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
