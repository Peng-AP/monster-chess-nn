"""Bounded mainline-only new increment; independent sampled-free tests first/last.

No book generation, architecture search, automatic promotion, optional stopping,
or silent interrupted-training restart. GEN49_PLAN.md declares the experiment.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import iterate
from free_play_audit import audit_match
from gate_sampled import validate_report
from iterate_stateful import validate_recipe
from match_evidence import atomic_json, file_hash, runtime_identity
from start_gpu48 import Campaign, campaign_lock, pin_json, read, selected_state

PREVIOUS = 'models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
TEACHER = 'models/candidates/bootstrap_main_gen_0048/arena_selected.pt'
HOLDOUT = 'models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt'
RELEASE = 'models/bootstrap_v27/best_value_net.pt'
PINNED_MODELS = {
    PREVIOUS: '810297fe98807eb56cfe4184c3210048fed7dc93110c47dfb18c0564fb3865fb',
    TEACHER: 'a8c074390c93390ac974b1f58076a86aa0525d9a34f7cff66e12442bb7e07722',
    HOLDOUT: 'fc23076a9f5c7f237785f27cb1a665c10588ea8e8916cd743016d19a96999d15',
    RELEASE: '976294daf7e3d6f0c51c358dd602f11997c7fdf2dc4b255b810b588c253e5459',
}
RUN = ROOT / 'benchmarks/gen49_mainline_20260914'
REHEARSAL = RUN / 'rehearsal_v2'
SMOKE = ROOT / 'iterations/rehearsal_gen49_20260914'


def layout(smoke):
    return dict(sims=8 if smoke else 3200, extras=4 if smoke else 200,
                research_per_side=2 if smoke else 100, research_par=4 if smoke else 200,
                final_per_side=2 if smoke else 200, final_par=4 if smoke else 400,
                seed=2190000000 if smoke else 2180000000, budget_min=10000)


def identity():
    paths = sorted((ROOT / 'tools').glob('*.py')) + sorted((ROOT / 'tests').glob('*.py'))
    paths += sorted((ROOT / 'configs').glob('*.json'))
    paths += [ROOT / 'tools/recipes' / n for n in ('gen49.json', 'gen49_rehearsal.json')]
    paths += [ROOT / 'GEN49_PLAN.md', ROOT / 'SAMPLED_GATE_PROTOCOL.md']
    recipe = read(ROOT / 'tools/recipes/gen49.json')
    validate_recipe(recipe)
    if (recipe['model'] != TEACHER or recipe['prefix_models'] or recipe['opponents']
            or [recipe[k] for k in ('free_games', 'fresh_games', 'league_games', 'fork_games')]
            != [2800, 0, 0, 400]):
        raise ValueError('Unexpected mainline recipe')
    hashes = {p: file_hash(ROOT / p) for p in PINNED_MODELS}
    if hashes != PINNED_MODELS:
        raise ValueError('Frozen model changed')
    prior = sorted((e for e in read(ROOT / 'iterations/accepted_data.json')['entries']
                    if int(e['generation']) < 49), key=lambda e: int(e['generation']))[-7:]
    if len(prior) != 7 or int(prior[-1]['generation']) != 48:
        raise ValueError('Expected seven accepted replay sources through gen48')
    return dict(runtime=runtime_identity(), models=hashes, replay=prior,
                implementations={str(p.relative_to(ROOT)): file_hash(p) for p in paths},
                protocol=layout(False), rehearsal=layout(True),
                scope='Normal-start research; no promotion; 3200 new mainline games; fixed architecture')


def iteration_command(smoke):
    recipe = 'tools/recipes/gen49_rehearsal.json' if smoke else 'tools/recipes/gen49.json'
    command = ['tools/iterate_stateful.py', '--recipe', recipe,
        '--expected-generation', str(1 if smoke else 49), '--', '--incumbent', TEACHER,
        '--games', str(28 if smoke else 3200), '--book-seed-games', '0',
        '--anchor-data', 'none', '--workers', '8', '--engine', 'native',
        '--seed', str(6173 if smoke else 3173), '--sims', str(8 if smoke else 1600),
        '--reanalysis-sample', str(20 if smoke else 24000),
        '--reanalysis-keep', str(10 if smoke else 12000),
        '--reanalysis-sims', str(8 if smoke else 6400), '--reanalysis-black-fraction', '.6',
        '--replay-generations', str(1 if smoke else 8),
        '--epochs', str(1 if smoke else 30), '--patience', str(1 if smoke else 10),
        '--batch-size', str(64 if smoke else 256), '--warmup-epochs', str(1 if smoke else 3),
        '--lr', '.002', '--ema-decay', '.999', '--value-floor', '.5', '--value-horizon', '60',
        '--teacher-policy-multiplier', '4', '--checkpoint-probe-games', str(4 if smoke else 40),
        '--checkpoint-screen-games', str(4 if smoke else 200),
        '--checkpoint-probe-sims', str(8 if smoke else 3200),
        '--checkpoint-screen-sims', str(8 if smoke else 3200),
        '--checkpoint-screen-finalists', str(1 if smoke else 2), '--through-phase', 'checkpoint_screen']
    if smoke:
        command += ['--run-root', str(SMOKE), '--offline-positions', '128']
    else:
        command += ['--data-seed-base', '1000000000']
    return command


def train(campaign, smoke):
    paths = iterate._paths_for_generation(SMOKE if smoke else ROOT / 'iterations', 1 if smoke else 49)
    command = iteration_command(smoke)
    if paths['state'].exists():
        state = read(paths['state'])
        if state['phases'].get('train', {}).get('status') in ('running', 'failed'):
            raise ValueError('Interrupted training preserved; automatic restart/overwrite refused')
        command += ['--resume']
    receipt = campaign.root / 'receipts/iteration.json'
    if receipt.exists():
        command = read(receipt)['command']
    campaign.stage('iteration', command, [paths['state'], paths['candidate'],
                                         paths['reports'] / 'reanalysis_coverage.json'])
    _, candidate = selected_state(paths['state'])
    coverage = read(paths['reports'] / 'reanalysis_coverage.json')
    if not coverage['complete'] or coverage['sample']['sampled'] != (20 if smoke else 24000):
        raise ValueError('Full requested reanalysis sample not verified')
    return candidate


def free_match(campaign, name, a, b, games, seed, sims):
    report = campaign.root / 'play' / f'{name}.json'
    log = report.with_suffix('.jsonl')
    campaign.stage(name, ['tools/match.py', '--model-a', str(a), '--model-b', str(b),
        '--games', str(games), '--sims', str(sims), '--sims-b', str(sims), '--engine', 'native', '--workers', '8',
        '--seed', str(seed), '--opening-temp-plies', '16', '--game-log', str(log),
        '--report-path', str(report), '--resume'], [report, log, log.with_suffix('.jsonl.manifest.json')])
    document = read(report)
    if document.get('partial') or document['games'] != games or document.get('book'):
        raise ValueError('Incomplete or wrong-instrument match')
    return audit_match(log, a, b, games, seed, sims)


def free_gate(campaign, name, model, bar, spec, final):
    root = campaign.root / 'play' / name
    target, par = ((spec['final_per_side'], spec['final_par']) if final
                   else (spec['research_per_side'], spec['research_par']))
    seed = spec['seed'] + (4000000 if final else 0)
    command = ['tools/gate_sampled.py', '--model', str(model), '--bar-model', str(bar),
        '--target-per-side', str(target), '--par-games', str(par), '--sims', str(spec['sims']),
        '--workers', '8', '--budget-min', str(spec['budget_min']), '--seed', str(seed),
        '--resume' if (root / 'manifest.json').exists() else '--run-dir', str(root)]
    receipt = campaign.root / 'receipts' / f'{name}.json'
    if receipt.exists():
        command = read(receipt)['command']
    outputs = [root / 'report.json', root / 'manifest.json']
    for leg in ('par', 'vs_bar', 'vs_bar_confirm'):
        outputs += [root / f'{leg}.jsonl', root / f'{leg}.jsonl.manifest.json']
    campaign.stage(name, command, outputs)
    report = read(root / 'report.json')
    validate_report(report, str(model), str(bar), spec['sims'], target, par)
    audits = {item['name']: audit_match(root / (item['name'] + '.jsonl'),
        bar if item['name'] == 'par' else model, bar, item['games'], item['seed'], spec['sims'])
        for item in report['schedule']}
    return dict(report=report, audits=audits, standard_production_counts=target == 200 and par == 400)


def precheck(campaign, smoke):
    spec = layout(smoke)
    gate = free_gate(campaign, 'gen48_vs_gen47', TEACHER, PREVIOUS, spec, False)
    extras = {}
    for index, (name, a, b) in enumerate([
        ('gen48_vs_b2', TEACHER, HOLDOUT), ('gen47_vs_b2', PREVIOUS, HOLDOUT),
        ('gen48_self', TEACHER, TEACHER)], 1):
        extras[name] = free_match(campaign, name, a, b, spec['extras'],
                                  spec['seed'] + index * 1000000, spec['sims'])
    summary = dict(complete=True, rehearsal=smoke, gate=gate, extras=extras,
                   games=spec['research_par'] + 4 * spec['research_per_side'] + 3 * spec['extras'],
                   interpretation='Normal-start sampled research; not the previous book instrument')
    pin_json(campaign.root / 'gen48_free_results.json', summary)
    print('GEN48 NORMAL-START EVIDENCE COMPLETE: ' + str(campaign.root / 'gen48_free_results.json'), flush=True)
    return summary


def postcheck(campaign, candidate, smoke, before):
    spec = layout(smoke)
    nominee = dict(path=str(candidate), sha256=file_hash(candidate))
    pin_json(campaign.root / 'nominee.json', nominee)
    gate = free_gate(campaign, 'gen49_vs_gen48', candidate, TEACHER, spec, True)
    extras = {}
    for index, (name, a, b) in enumerate([
        ('gen49_vs_b2', candidate, HOLDOUT), ('gen49_vs_v27', candidate, RELEASE),
        ('gen49_self', candidate, candidate)], 5):
        if file_hash(candidate) != nominee['sha256']:
            raise ValueError('Selected candidate changed during evaluation')
        extras[name] = free_match(campaign, name, a, b, spec['extras'],
                                  spec['seed'] + index * 1000000, spec['sims'])
    if file_hash(candidate) != nominee['sha256'] or identity() != campaign.provenance:
        raise ValueError('Inputs changed before final publication')
    summary = dict(complete=True, rehearsal=smoke, candidate=str(candidate),
        candidate_sha256=file_hash(candidate), gate=gate, extras=extras,
        gen48_free_results_sha256=file_hash(campaign.root / 'gen48_free_results.json'),
        manifest_sha256=file_hash(campaign.root / 'manifest.json'),
        post_selection_games=spec['final_par'] + 4 * spec['final_per_side'] + 3 * spec['extras'],
        pre_training_games=before['games'],
        timing={p.stem: read(p)['elapsed_sec'] for p in (campaign.root / 'receipts').glob('*.json')},
        notes=['Normal initial position, first 16 half-plies temperature .5, then zero.',
               'Repeated openings retain their sampled frequency; novelty is not the target.',
               'Fixed simulations, not equal wall time. Nominal uncertainty, no perfect-play claim.',
               'Independent RNG blocks are not matched identical starts across model pairings.',
               'Self-color estimates use all actual-color outcomes and are complementary.',
               'Historical replay remains mixed; only the new increment is mainline-only.',
               'All prescribed test branches ran irrespective of measured scores; no promotion.'])
    pin_json(campaign.root / 'summary.json', summary)
    atomic_json(campaign.root / 'status.json', dict(status='complete',
        summary_sha256=file_hash(campaign.root / 'summary.json')))
    print(f'GEN49 {"REHEARSAL" if smoke else "PRODUCTION"} COMPLETE: {campaign.root}/summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rehearsal-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ['MONSTER_PINNED_INPUT'] = '1'
    provenance = identity()
    with campaign_lock(RUN / 'campaign.lock'):
        for smoke in (True, False):
            if not smoke and args.rehearsal_only:
                break
            campaign = Campaign(REHEARSAL if smoke else RUN / 'production', provenance,
                                identity_fn=identity, label='GEN49')
            try:
                if smoke:
                    campaign.stage('tests', ['-m', 'pytest', 'tests', '-q'])
                else:
                    if not read(REHEARSAL / 'summary.json')['complete']:
                        raise ValueError('Full rehearsal did not complete')
                before = precheck(campaign, smoke)
                candidate = train(campaign, smoke)
                postcheck(campaign, candidate, smoke, before)
            except BaseException as exc:
                atomic_json(campaign.root / 'status.json', dict(status='failed', error=str(exc)))
                raise


if __name__ == '__main__':
    main()
