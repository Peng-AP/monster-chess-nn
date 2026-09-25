"""Resume frozen mainline v2, then run a separately pinned diagnostic extension."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
import start_mainline_study as base
from mainline_study import load_result

RUN = ROOT / 'benchmarks/mainline_extension_20260915'


def identity():
    return dict(base=base.identity(), launcher=base.file_hash(__file__),
                plan=base.file_hash(ROOT / 'MAINLINE_EXTENSION_PLAN.md'))


def tasks(smoke):
    cases = [c for c in base.build_cases()
             if c['family'] == 'early' and c['id'].endswith('d7d5')]
    result = []
    for white, black in [('gen49', 'gen49'), ('b2', 'gen49'),
                         ('gen49', 'gen48'), ('gen48', 'gen49')]:
        for case in cases:
            for sample in range(1 if smoke else 32):
                result.append(dict(case_id=case['id'], state=case['state'],
                    case_sha256=base.digest(case), white_model=base.MODELS[white],
                    black_model=base.MODELS[black], white_sims=8 if smoke else 12800,
                    black_sims=8 if smoke else 12800, kind='game',
                    seed=(2280000000 if smoke else 2260000000)+len(result),
                    sample_index=sample, deadline_seconds=3600))
    assert len(result) == (8 if smoke else 256)
    assert len({base.digest(t) for t in result}) == len(result)
    return result


def work(smoke):
    provenance = identity()
    campaign = base.Campaign(RUN / ('rehearsal' if smoke else 'production'),
                            provenance, identity_fn=identity, label='EXTENSION')
    schedule = tasks(smoke)
    config = campaign.root / 'conditional_config.json'
    base.pin(config, dict(tasks=schedule, rehearsal=smoke,
                         protocol='full_history_mainline_followup_v1'))
    out = campaign.root / 'd5_defense'
    campaign.stage('d5_defense', ['tools/mainline_study.py', '--config', str(config),
                   '--output', str(out)], [out/'manifest.json', out/'summary.json'])
    summary = base.read(out/'summary.json')
    if not summary['complete'] or summary['games'] != len(schedule):
        raise ValueError('Incomplete conditional extension')
    for task in schedule:
        load_result(out/'tasks'/(base.digest(task)+'.json'), task)
    normal = {}
    for index, (name, a, b) in enumerate([
            ('gen49_vs_gen48_deeper', 'gen49', 'gen48'), ('b2_deeper_self', 'b2', 'b2')]):
        item = dict(name=name, a=base.MODELS[a], b=base.MODELS[b],
                    sims_a=8 if smoke else 12800, sims_b=8 if smoke else 12800,
                    games=4 if smoke else 160,
                    seed=(2290000000 if smoke else 2270000000)+index*1000000)
        normal[name] = base.normal_match(campaign, item)
    if identity() != provenance:
        raise ValueError('Extension inputs changed')
    base.pin(campaign.root/'summary.json', dict(complete=True, rehearsal=smoke,
        games=len(schedule)+sum(v['settings']['games'] for v in normal.values()),
        conditional_summary_sha256=base.file_hash(out/'summary.json'), normal=normal,
        manifest_sha256=base.file_hash(campaign.root/'manifest.json')))
    base.atomic_json(campaign.root/'status.json', dict(status='complete'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rehearsal-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ['MONSTER_PINNED_INPUT'] = '1'
    if base.identity() != base.read(base.RUN/'production/manifest.json'):
        raise ValueError('Original campaign identity drift; refusing resume')
    with base.campaign_lock(RUN/'campaign.lock'):
        work(True)
        if args.rehearsal_only:
            return
        subprocess.run([sys.executable, '-u', 'tools/start_mainline_study.py'], check=True)
        if not base.read(base.RUN/'production/summary.json')['complete']:
            raise ValueError('Original study must finish before extension')
        work(False)
        print('MAINLINE BASE AND EXTENSION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
