"""Recover the finalist opening shortfall without changing completed B2 evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from match_evidence import atomic_json, file_hash, runtime_identity


def validate_book(document):
    expected = dict(plies=16, sims=700, temperature=.5, seed=1990000000)
    if any(document.get(k) != value for k,value in expected.items()):
        raise ValueError('Opening generation settings changed')
    entries = document['entries']
    if len(entries) != 100:
        raise ValueError('Need exactly100 positions, not a shortened benchmark')
    keys = {(e['fen'], bool(e['half']), e['turn_count']) for e in entries}
    if len(keys) != 100:
        raise ValueError('Duplicate opening states')


def main():
    root = ROOT/'benchmarks/b2_001_comparison_20260909'
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['runtime'] != runtime_identity() or manifest['implementation'] != file_hash(ROOT/'tools/b2_benchmark.py'):
        raise ValueError('Benchmark runtime or implementation changed')
    for model, sha in manifest['models'].items():
        if file_hash(model) != sha:
            raise ValueError(f'Changed model {model}')
    completed = root/'screen_complete.json'
    if json.loads(completed.read_text()).get('complete') is not True:
        raise ValueError('Initial screen did not complete')
    book = root/'finalist_book.json'
    receipt = book.with_suffix('.receipt.json')
    record_path = root/'opening_recovery_20260909.json'
    record = dict(original_manifest_sha256=file_hash(root/'manifest.json'),
        completed_screen_sha256=file_hash(completed), implementation=file_hash(__file__),
        change='Increase opening attempt budget1.6 ->4.0; same seed/models/depth/temperature/sims/count',
        commands=[], completed=False)
    if record_path.exists():
        saved = json.loads(record_path.read_text())
        for key in ('original_manifest_sha256','completed_screen_sha256','implementation'):
            if saved[key] != record[key]:
                raise ValueError('Recovery provenance changed')
        record = saved
    env = dict(os.environ, MONSTER_PINNED_INPUT='1')
    if not receipt.exists():
        if book.exists():
            raise ValueError('Unreceipted opening book; inspect instead of overwriting')
        command = [sys.executable,'-u','tools/make_book.py']
        for v in (24,25,26):
            command += ['--model',f'models/bootstrap_v{v}/best_value_net.pt']
        command += ['--entries','100','--plies','16','--sims','700','--temperature','0.5',
                    '--seed','1990000000','--workers','8','--oversample','4','--out',str(book)]
        record['commands'].append(command)
        atomic_json(record_path,record)
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        validate_book(json.loads(book.read_text()))
        atomic_json(receipt,dict(sha256=file_hash(book)))
    validate_book(json.loads(book.read_text()))
    if json.loads(receipt.read_text())['sha256'] != file_hash(book):
        raise ValueError('Opening book changed')
    record.update(completed=True,book_sha256=file_hash(book))
    atomic_json(record_path,record)
    # Original driver validates journals; completed matches have no pending
    # tasks and are not played again. Preserve its original manifest verbatim.
    subprocess.run([sys.executable,'-u','tools/b2_benchmark.py'],cwd=ROOT,check=True)


if __name__ == '__main__':
    main()
