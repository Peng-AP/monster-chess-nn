"""ABBA pinned-input validation through the real persistent-engine match driver."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from match import run_match
from match_evidence import atomic_json, file_hash, runtime_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require', required=True, help='Completed deeper-generation parity report')
    parser.add_argument('--games', type=int, default=32)
    args = parser.parse_args()
    prerequisite = json.loads(Path(args.require).read_text())
    if prerequisite.get('exact_record_parity') is not True:
        raise RuntimeError('Prerequisite did not pass; refusing to continue')
    if args.games < 8 or args.games % 2:
        raise ValueError('Need an even number of games >=8')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model = 'models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
    opponent = 'models/bootstrap_v27/best_value_net.pt'
    report = dict(model_sha256=file_hash(model), opponent_sha256=file_hash(opponent),
                  runtime=runtime_identity(), implementation=file_hash(__file__),
                  prerequisite_sha256=file_hash(args.require), runs=[])
    previous = os.environ.get('MONSTER_PINNED_INPUT')
    reference = None
    try:
        for index, mode in enumerate(('0', '1', '1', '0')):
            os.environ['MONSTER_PINNED_INPUT'] = mode
            lines = output / f'pass_{index}.jsonl'
            started = time.perf_counter()
            result = run_match(model, opponent, args.games, 3200, 1820000000,
                               workers=8, engine='native', opening_temp_plies=16,
                               checkpoint_path=str(output / f'pass_{index}.json'),
                               game_log=str(lines), stall_timeout=600)
            elapsed = time.perf_counter() - started
            rows = [json.loads(line) for line in lines.read_text().splitlines() if line.strip()]
            records = {row['task_id']: row for row in rows}
            if len(records) != args.games:
                raise RuntimeError('Missing/duplicate match records')
            if reference is None:
                reference = records
            parity = records == reference
            report['runs'].append(dict(mode=mode, seconds=elapsed, exact_game_record_parity=parity,
                                       match=result, log_sha256=file_hash(lines)))
            atomic_json(output / 'summary.json', report)
            if not parity:
                raise RuntimeError('Persistent-engine game records changed; inspect reports')
        times = {mode: sum(r['seconds'] for r in report['runs'] if r['mode'] == mode)/2
                 for mode in ('0', '1')}
        report.update(complete=True, mean_seconds=times, speedup=times['0']/times['1'])
        atomic_json(output / 'summary.json', report)
        print(json.dumps(dict(mean_seconds=times, speedup=report['speedup']), indent=2))
    finally:
        if previous is None:
            os.environ.pop('MONSTER_PINNED_INPUT', None)
        else:
            os.environ['MONSTER_PINNED_INPUT'] = previous


if __name__ == '__main__':
    main()
