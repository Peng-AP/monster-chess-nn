"""Rehearse, train gen50 from gen49, independently test both search budgets."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import start_gen49 as prior
from start_gpu48 import Campaign, campaign_lock, read, pin_json, selected_state
from match_evidence import atomic_json, file_hash, runtime_identity

TEACHER = 'models/candidates/bootstrap_main_gen_0049/arena_selected.pt'
RUN = ROOT/'benchmarks/gen50_20260916'
SMOKE = ROOT/'iterations/rehearsal_gen50_20260916'
MODELS = {TEACHER: '4bcc68a0219acf8c3dc53326d738789e6bd767fccba88fd4f471345567b4a647',
          prior.HOLDOUT: prior.PINNED_MODELS[prior.HOLDOUT],
          prior.RELEASE: prior.PINNED_MODELS[prior.RELEASE]}


def identity():
    paths = sorted((ROOT/'tools').glob('*.py')) + sorted((ROOT/'tests').glob('*.py'))
    paths += sorted((ROOT/'configs').glob('*.json'))
    paths += [ROOT/p for p in ('run_gen50.py', 'test_gen50.py', 'GEN50_PLAN.md',
              'gen50_recipe.json', 'gen50_rehearsal_recipe.json', 'SAMPLED_GATE_PROTOCOL.md')]
    hashes = {p:file_hash(ROOT/p) for p in MODELS}
    if hashes != MODELS:
        raise ValueError('Frozen reference model changed')
    replay = sorted((e for e in read(ROOT/'iterations/accepted_data.json')['entries']
                     if int(e['generation']) < 50), key=lambda e:int(e['generation']))[-7:]
    if len(replay) != 7 or int(replay[-1]['generation']) != 49:
        raise ValueError('Expected accepted replay through gen49')
    for name in ('gen50_recipe.json', 'gen50_rehearsal_recipe.json'):
        prior.validate_recipe(read(ROOT/name))
    return dict(runtime=runtime_identity(), models=hashes, replay=replay,
                implementation={str(p.relative_to(ROOT)):file_hash(p) for p in paths})


def iteration_command(smoke):
    # Reuse the verified gen49 formula, replacing only explicitly declared knobs.
    command = prior.iteration_command(smoke)
    changes = {'--recipe':'gen50_rehearsal_recipe.json' if smoke else 'gen50_recipe.json',
               '--expected-generation':'1' if smoke else '50', '--incumbent':TEACHER,
               '--reanalysis-sims':'8' if smoke else '12800'}
    if smoke:
        changes['--run-root'] = str(SMOKE)
    for flag, value in changes.items():
        command[command.index(flag)+1] = value
    return command


def train(campaign, smoke):
    paths = prior.iterate._paths_for_generation(SMOKE if smoke else ROOT/'iterations', 1 if smoke else 50)
    command = iteration_command(smoke)
    if paths['state'].exists():
        state = read(paths['state'])
        if state['phases'].get('train', {}).get('status') in ('running', 'failed'):
            raise ValueError('Interrupted training retained; refusing automatic overwrite')
        command += ['--resume']
    receipt = campaign.root/'receipts/iteration.json'
    if receipt.exists():
        command = read(receipt)['command']
    campaign.stage('iteration', command, [paths['state'], paths['candidate'],
                   paths['reports']/'reanalysis_coverage.json'])
    _, candidate = selected_state(paths['state'])
    coverage = read(paths['reports']/'reanalysis_coverage.json')
    if not coverage['complete'] or coverage['sample']['sampled'] != (20 if smoke else 24000):
        raise ValueError('Reanalysis coverage incomplete')
    return candidate


def evaluation_schedule(candidate, smoke):
    rows = [('vs_b2', candidate, prior.HOLDOUT, 3200, 200),
            ('vs_v27', candidate, prior.RELEASE, 3200, 200),
            ('self', candidate, candidate, 3200, 200),
            ('deep_vs_gen49', candidate, TEACHER, 12800, 160),
            ('deep_vs_b2', candidate, prior.HOLDOUT, 12800, 160),
            ('deep_self', candidate, candidate, 12800, 160)]
    return [dict(name=n, a=str(a), b=str(b), sims=8 if smoke else sims,
                 games=4 if smoke else games,
                 seed=(2320000000 if smoke else 2310000000)+i*1000000)
            for i,(n,a,b,sims,games) in enumerate(rows)]


def work(campaign, smoke):
    candidate = train(campaign, smoke)
    nominee = dict(path=str(candidate), sha256=file_hash(candidate))
    pin_json(campaign.root/'nominee.json', nominee)
    spec = prior.layout(smoke)
    spec['seed'] = 2340000000 if smoke else 2330000000
    gate = prior.free_gate(campaign, 'vs_gen49', candidate, TEACHER, spec, True)
    extras = {}
    for item in evaluation_schedule(candidate, smoke):
        if file_hash(candidate) != nominee['sha256']:
            raise ValueError('Candidate changed')
        extras[item['name']] = prior.free_match(campaign, **item)
    if identity() != campaign.provenance or file_hash(candidate) != nominee['sha256']:
        raise ValueError('Inputs changed before publication')
    pin_json(campaign.root/'summary.json', dict(complete=True, rehearsal=smoke,
        nominee=nominee, gate=gate, extras=extras, post_selection_games=36 if smoke else 2280,
        manifest_sha256=file_hash(campaign.root/'manifest.json'),
        notes=['All test branches ran regardless of measured gate result.',
               'B2 is a studied diagnostic opponent, not a blind holdout.',
               'No model promoted. New teacher and deeper targets change together.']))
    atomic_json(campaign.root/'status.json', dict(status='complete'))
    print(f'GEN50 {"REHEARSAL" if smoke else "PRODUCTION"} COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rehearsal-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ['MONSTER_PINNED_INPUT'] = '1'
    provenance = identity()
    with campaign_lock(RUN/'campaign.lock'):
        for smoke in (True, False):
            if not smoke and args.rehearsal_only:
                break
            campaign = Campaign(RUN/('rehearsal' if smoke else 'production'), provenance,
                                identity_fn=identity, label='GEN50')
            try:
                if smoke:
                    campaign.stage('tests', ['-m', 'pytest', 'tests', 'test_gen50.py', '-q'])
                elif not read(RUN/'rehearsal/summary.json')['complete']:
                    raise ValueError('Rehearsal required')
                work(campaign, smoke)
            except BaseException as exc:
                atomic_json(campaign.root/'status.json', dict(status='failed', error=str(exc)))
                raise


if __name__ == '__main__':
    main()
