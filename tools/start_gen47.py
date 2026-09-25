"""After gen46 diagnostics, rehearse the new data backend, then launch gen47.

No automatic promotion or repair. An execution failure stops the chain. Resume
uses existing states and receipts, never silently replaces a failed experiment.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = 'iterations/rehearsal_stateful_gen47_20260907'
MODEL = 'models/bootstrap_v27/best_value_net.pt'


def commands():
    smoke = ['tools/iterate_stateful.py', '--recipe', 'tools/recipes/gen47_rehearsal.json',
             '--expected-generation', '1', '--', '--run-root', SMOKE_ROOT,
             '--incumbent', MODEL, '--games', '28', '--book-seed-games', '0',
             '--seed', '6173', '--sims', '8', '--reanalysis-sample', '20',
             '--reanalysis-keep', '10', '--reanalysis-sims', '8', '--replay-generations', '1',
             '--epochs', '1', '--patience', '1', '--batch-size', '64', '--warmup-epochs', '1',
             '--offline-positions', '128', '--sampled-gate-target-per-side', '2',
             '--sampled-gate-par-games', '4', '--sampled-gate-budget-min', '10',
             '--arena-sims', '8', '--checkpoint-screen-games', '4', '--checkpoint-probe-games', '4',
             '--checkpoint-screen-sims', '8', '--checkpoint-probe-sims', '8',
             '--checkpoint-screen-finalists', '1', '--self-skew-games', '4']
    production = ['tools/iterate_stateful.py', '--recipe', 'tools/recipes/gen47.json',
                  '--expected-generation', '47', '--', '--incumbent', MODEL,
                  '--games', '5600', '--book-seed-games', '0', '--seed', '3173',
                  '--data-seed-base', '1000000000', '--reanalysis-sample', '80000',
                  '--reanalysis-keep', '40000', '--self-skew-games', '200']
    return smoke, production


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    smoke, production = commands()
    if args.dry_run:
        for cmd in (smoke, production):
            subprocess.run([sys.executable, *cmd, '--dry-run'], cwd=ROOT, check=True)
        return
    predecessor = json.loads((ROOT / 'benchmarks/generalization/gen46_20260907/state.json').read_text())
    if predecessor.get('status') != 'complete':
        raise ValueError('gen46 transfer checks did not finish successfully')
    print('GEN47 PREFLIGHT: full test suite with the worker lease now free', flush=True)
    subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], cwd=ROOT, check=True)
    smoke_state = ROOT / SMOKE_ROOT / 'gen_0001/state.json'
    if smoke_state.exists():
        smoke.append('--resume')
    print('GEN47 PREFLIGHT: real 28-game end-to-end rehearsal', flush=True)
    subprocess.run([sys.executable, '-u', *smoke], cwd=ROOT, check=True)
    state = json.loads(smoke_state.read_text())
    if state['status'] not in ('passed_not_promoted', 'rejected', 'inconclusive'):
        raise ValueError(f"rehearsal did not reach a measured verdict: {state['status']}")
    if state['phases']['binding_gate']['status'] != 'completed':
        raise ValueError('rehearsal did not finish play testing')
    print('GEN47 PREFLIGHT PASSED; launching the production iteration', flush=True)
    if (ROOT / 'iterations/gen_0047/state.json').exists():
        production.append('--resume')
    subprocess.run([sys.executable, '-u', *production], cwd=ROOT, check=True)
    followup = [
        'tools/generation_followup.py', '--candidate', 'models/candidates/bootstrap_main_gen_0047/arena_selected.pt',
        '--baseline', MODEL, '--opponent', 'models/bootstrap_v24/best_value_net.pt',
        '--opponent', 'models/bootstrap_v26/best_value_net.pt',
        '--book-teacher', 'models/bootstrap_v24/best_value_net.pt',
        '--book-teacher', 'models/bootstrap_v25/best_value_net.pt',
        '--book-teacher', 'models/candidates/bootstrap_main_gen_0044/screen_nominee_v3.pt',
        '--iteration-state', 'iterations/gen_0047/state.json',
        '--run-dir', 'benchmarks/generalization/gen47_20260907', '--seed', '170000000',
        '--book-entries', '120', '--book-sims', '700', '--sims', '3200', '--free-games', '200', '--workers', '8']
    subprocess.run([sys.executable, '-u', *followup], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
