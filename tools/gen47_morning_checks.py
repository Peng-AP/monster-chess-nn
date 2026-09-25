"""Bounded additional gen47 diagnostics until 11am Eastern; never trains/promotes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from match_evidence import atomic_json, file_hash, model_identity, runtime_identity

CANDIDATE = 'models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
BASELINE = 'models/bootstrap_v27/best_value_net.pt'
OPPONENTS = ['models/bootstrap_v25/best_value_net.pt',
             'models/candidates/bootstrap_main_gen_0044/screen_nominee_v3.pt',
             'models/bootstrap_v26/best_value_net.pt']
DIRECTORY = ROOT / 'benchmarks/generalization/gen47_morning_20260908'


def round_command(index, directory=DIRECTORY):
    opponent = OPPONENTS[index % len(OPPONENTS)]
    sims = 3200 if index % 2 == 0 else 6400
    return ['tools/generation_followup.py', '--candidate', CANDIDATE, '--baseline', BASELINE,
            '--opponent', opponent, '--book-teacher', 'models/bootstrap_v24/best_value_net.pt',
            '--book-teacher', 'models/bootstrap_v25/best_value_net.pt',
            '--book-teacher', 'models/candidates/bootstrap_main_gen_0044/screen_nominee_v3.pt',
            '--iteration-state', 'iterations/gen_0047/state.json',
            '--run-dir', str(directory / f'round_{index:02d}'),
            '--seed', str(200000000 + index * 10000000),
            '--book-entries', '60', '--book-sims', '700', '--sims', str(sims),
            '--free-games', '120', '--workers', '8']


def run(args):
    deadline = datetime.fromisoformat(args.until)
    if deadline.tzinfo is None:
        raise ValueError('deadline must include UTC offset')
    if args.dry_run:
        print(json.dumps({'until': args.until, 'initial': '200 self-games at 6400 sims',
                          'first_rounds': [round_command(i) for i in range(6)],
                          'stop': 'do not start another round after deadline; finish started rounds'}, indent=2))
        return
    previous = json.loads((ROOT / 'benchmarks/generalization/gen47_20260907/state.json').read_text())
    if previous.get('status') != 'complete':
        raise ValueError('existing gen47 follow-up did not finish successfully')
    report = json.loads((ROOT / 'iterations/gen_0047/reports/binding_gate.json').read_text())
    if report.get('verdict') != 'PASS' or not report.get('confirmed'):
        raise ValueError('expected confirmed gen47 checkpoint')
    manifest = dict(until=args.until, runtime=runtime_identity(), implementation=file_hash(__file__),
                    models=[model_identity(p) for p in [CANDIDATE, BASELINE, *OPPONENTS,
                                                       'models/bootstrap_v24/best_value_net.pt']],
                    followup_implementation=file_hash(ROOT / 'tools/generation_followup.py'))
    path = DIRECTORY / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else dict(manifest=manifest, stages={}, status='running')
    if state['manifest'] != manifest:
        raise ValueError('morning-check resume provenance changed')
    atomic_json(path, state)

    def stage(name, command, output):
        done = state['stages'].get(name)
        if done and done.get('status') == 'complete':
            if file_hash(output) != done['sha256']:
                raise ValueError('completed diagnostic output changed')
            return
        state['stages'][name] = dict(status='running', command=command)
        atomic_json(path, state)
        print(f'MORNING CHECK {name}: {command}', flush=True)
        subprocess.run([sys.executable, '-u', *command], cwd=ROOT, check=True)
        state['stages'][name] = dict(status='complete', command=command, sha256=file_hash(output))
        atomic_json(path, state)

    try:
        self_report = DIRECTORY / 'self_6400.json'
        if datetime.now(deadline.tzinfo) < deadline or state['stages'].get('self_6400', {}).get('status') == 'running':
            stage('self_6400', ['tools/match.py', '--model-a', CANDIDATE, '--model-b', CANDIDATE,
                               '--games', '200', '--sims', '6400', '--engine', 'native', '--workers', '8',
                               '--seed', '190000000', '--report-path', str(self_report),
                               '--game-log', str(DIRECTORY / 'self_6400.jsonl'), '--resume'], self_report)
        # Hard cap guards against pathological clock changes. Work is additional,
        # nonbinding evaluation, not optional extensions to a promotion gate.
        for index in range(24):
            name = f'round_{index:02d}'
            prior = state['stages'].get(name, {})
            if prior.get('status') != 'running' and datetime.now(deadline.tzinfo) >= deadline:
                break
            stage(name, round_command(index), DIRECTORY / name / 'summary.json')
        state['status'] = 'complete'
        state['finished'] = datetime.now(deadline.tzinfo).isoformat()
        atomic_json(path, state)
        print('Morning diagnostics complete; no training or promotion performed.', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        atomic_json(path, state)
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--until', default='2026-09-08T11:00:00-04:00')
    ap.add_argument('--dry-run', action='store_true')
    run(ap.parse_args())
