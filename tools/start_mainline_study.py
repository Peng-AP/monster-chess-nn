"""Rehearse -> fixed mainline counterplay/search ladder -> normal play -> stop."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from free_play_audit import audit_game, audit_match
from mainline_study import snapshot, checked_restore, pin, read
from match_evidence import atomic_json, digest, file_hash, read_rows, runtime_identity
from monster_chess import MonsterChessGame
from start_gpu48 import Campaign, campaign_lock

MODELS = {
    'gen48': 'models/candidates/bootstrap_main_gen_0048/arena_selected.pt',
    'gen49': 'models/candidates/bootstrap_main_gen_0049/arena_selected.pt',
    'b2': 'models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt',
}
HASHES = {
    'gen48': 'a8c074390c93390ac974b1f58076a86aa0525d9a34f7cff66e12442bb7e07722',
    'gen49': '4bcc68a0219acf8c3dc53326d738789e6bd767fccba88fd4f471345567b4a647',
    'b2': 'fc23076a9f5c7f237785f27cb1a665c10588ea8e8916cd743016d19a96999d15',
}
SOURCE = ROOT / 'benchmarks/gen49_mainline_20260914/production/play/gen49_vs_b2.jsonl'
RUN = ROOT / 'benchmarks/mainline_counterplay_20260915_v2'
REHEARSAL = RUN / 'rehearsal'
STAGES = ('early_3200', 'early_12800', 'early_51200', 'draw_crossplay', 'root_decisions')


def layout(smoke):
    return dict(base_seed=2240000000 if smoke else 2220000000,
                normal_seed=2250000000 if smoke else 2230000000,
                normal_games=4 if smoke else 160,
                normal_low=8 if smoke else 3200, normal_high=16 if smoke else 12800,
                early_depths=[8, 16, 32] if smoke else [3200, 12800, 51200],
                early_repeats=[1, 1, 1] if smoke else [8, 8, 2],
                draw_depths=[8] if smoke else [3200, 12800, 51200],
                probe_depths=[16] if smoke else [3200, 12800, 51200, 204800])


def identity():
    paths = sorted((ROOT / 'tools').glob('*.py')) + sorted((ROOT / 'tests').glob('*.py'))
    paths += [ROOT / 'MAINLINE_COUNTERPLAY_PLAN.md']
    hashes = {name: file_hash(ROOT / path) for name, path in MODELS.items()}
    if hashes != HASHES:
        raise ValueError('Frozen study model changed')
    previous = read(SOURCE.parent.parent / 'summary.json')
    if previous['extras']['gen49_vs_b2']['evidence'][str(SOURCE)] != file_hash(SOURCE):
        raise ValueError('Preserved source journal no longer matches gen49 evidence')
    return dict(runtime=runtime_identity(), models=hashes,
                sources={str(SOURCE): file_hash(SOURCE)},
                implementation={str(p.relative_to(ROOT)): file_hash(p) for p in paths},
                production=layout(False), rehearsal=layout(True), stages=list(STAGES),
                scope='Conditional diagnostics plus free search scaling; no training or promotion')


def make_state(moves):
    import chess
    game = MonsterChessGame()
    initial = game.fen()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in game.get_search_actions():
            raise ValueError('Illegal case prefix')
        game.apply_search_action(move)
    state = snapshot(game, initial, moves)
    checked_restore(state)
    return state


def build_cases():
    cases = []
    for order in [('e2e4', 'd2d4'), ('d2d4', 'e2e4')]:
        for reply in ('e7e5', 'd7d5'):
            moves = [*order, reply]
            cases.append(dict(id='_'.join(moves), family='early', state=make_state(moves),
                              source=dict(kind='explicit_observed_opening_branch', moves=moves)))
    groups = defaultdict(list)
    for row in read_rows(SOURCE):
        audit_game(row)
        if not row['a_is_white'] and row['game']['termination'] == 'repetition':
            opening = row['opening']
            groups[(opening['fen'], opening['half'], opening['turn_count'])].append(row)
    key = sorted(groups, key=lambda k: (-len(groups[k]), k))[0]
    rows = groups[key]
    if len(rows) != 56:
        raise ValueError('Unexpected dominant B2 source family')
    row = min(rows, key=lambda r: r['seed'])
    if row['plies'] != 74:
        raise ValueError('Unexpected source trajectory length')
    for index in (16, 17, 41, 62, 65):
        moves = [a for t in row['game']['trajectory'][1:index+1] for a in t['action']]
        state = make_state(moves)
        original = row['game']['trajectory'][index]
        if any(state[k] != original[k] for k in ('fen', 'half', 'turn_count')):
            raise ValueError('Extracted root disagrees with original game')
        cases.append(dict(id=f'b2_draw_ply_{index:03d}', family='draw', state=state,
            source=dict(path=str(SOURCE), sha256=file_hash(SOURCE), task_id=row['task_id'],
                        seed=row['seed'], ply=index, matching_endpoint_games=len(rows))))
    return cases


def make_tasks(stage, cases, smoke):
    spec = layout(smoke)
    seed_base = spec['base_seed'] + STAGES.index(stage) * 100000
    tasks = []
    if stage.startswith('early_'):
        stage_index = STAGES.index(stage)
        depths = [spec['early_depths'][stage_index]]
        repeats = spec['early_repeats'][stage_index]
        chosen = [c for c in cases if c['family'] == 'early']
    else:
        depths = spec['draw_depths'] if stage == 'draw_crossplay' else spec['probe_depths']
        repeats = 1
        chosen = [c for c in cases if c['family'] == 'draw'] if stage == 'draw_crossplay' else cases
    probe = stage == 'root_decisions'
    for sims in depths:
        for white_name, white in MODELS.items():
            for black_name, black in MODELS.items():
                if probe and black_name != white_name:
                    continue
                for case in chosen:
                    for repetition in range(repeats):
                        tasks.append(dict(case_id=case['id'], state=case['state'],
                            case_sha256=digest(case), white_model=white, black_model=black,
                            white_sims=sims, black_sims=sims, kind='probe' if probe else 'game',
                            seed=seed_base + len(tasks), sample_index=repetition,
                            deadline_seconds=3600))
    if smoke and probe:
        # Exercise the largest real budget with four simultaneous trees before
        # production, not just the tiny-search plumbing path. Cost evidence only.
        for case in (cases[0], cases[4], cases[5], cases[8]):
            tasks.append(dict(case_id=case['id'], state=case['state'], case_sha256=digest(case),
                white_model=MODELS['gen49'], black_model=MODELS['gen49'],
                white_sims=204800, black_sims=204800, kind='probe',
                seed=seed_base+len(tasks), sample_index=0, deadline_seconds=3600))
    return tasks


def normal_schedule(smoke):
    spec = layout(smoke)
    low, high = spec['normal_low'], spec['normal_high']
    rows = [('gen49_search_scaling', 'gen49', 'gen49', high, low),
            ('gen49_vs_deeper_b2', 'gen49', 'b2', low, high),
            ('deeper_gen49_vs_deeper_b2', 'gen49', 'b2', high, high),
            ('deeper_gen49_self', 'gen49', 'gen49', high, high)]
    return [dict(name=n, a=MODELS[a], b=MODELS[b], sims_a=sa, sims_b=sb,
                 games=spec['normal_games'], seed=spec['normal_seed'] + index*1000000)
            for index, (n, a, b, sa, sb) in enumerate(rows)]


def normal_match(campaign, item):
    report = campaign.root / 'normal' / (item['name'] + '.json')
    log = report.with_suffix('.jsonl')
    campaign.stage(item['name'], ['tools/match.py', '--model-a', item['a'], '--model-b', item['b'],
        '--sims', str(item['sims_a']), '--sims-b', str(item['sims_b']), '--games', str(item['games']),
        '--engine', 'native', '--workers', '8', '--seed', str(item['seed']), '--opening-temp-plies', '16',
        '--stall-timeout', '1800', '--report-path', str(report), '--game-log', str(log), '--resume'],
        [report, log, log.with_suffix('.jsonl.manifest.json')])
    document = read(report)
    if document.get('partial') or document['games'] != item['games'] or document.get('book'):
        raise ValueError('Normal-start block incomplete or wrong instrument')
    audit = audit_match(log, item['a'], item['b'], item['games'], item['seed'], item['sims_a'], item['sims_b'])
    return dict(settings=item, audit=audit)


def work(campaign, smoke):
    cases = build_cases()
    pin(campaign.root / 'cases.json', cases)
    summaries = {}
    for stage in STAGES:
        tasks = make_tasks(stage, cases, smoke)
        config = campaign.root / 'configs' / (stage + '.json')
        pin(config, dict(tasks=tasks, cases_sha256=file_hash(campaign.root/'cases.json'),
                         rehearsal=smoke, protocol='full_history_mainline_diagnostic_v1'))
        out = campaign.root / stage
        campaign.stage(stage, ['tools/mainline_study.py', '--config', str(config), '--output', str(out)],
                       [out/'manifest.json', out/'summary.json'])
        summary = read(out / 'summary.json')
        if (not summary['complete'] or summary['tasks'] != len(tasks)
                or any(not Path(p).exists() or file_hash(p) != h for p, h in summary['evidence'].items())):
            raise ValueError('Conditional suite incomplete or evidence changed')
        summaries[stage] = {k:v for k,v in summary.items() if k != 'evidence'}
    normal = {item['name']:normal_match(campaign, item) for item in normal_schedule(smoke)}
    if identity() != campaign.provenance:
        raise ValueError('Inputs changed before study publication')
    summary = dict(complete=True, rehearsal=smoke, conditional=summaries, normal=normal,
        models=MODELS, model_sha256=HASHES, manifest_sha256=file_hash(campaign.root/'manifest.json'),
        cases_sha256=file_hash(campaign.root/'cases.json'),
        games=sum(v['games'] for v in summaries.values()) + sum(v['settings']['games'] for v in normal.values()),
        root_probes=sum(v['root_probes'] for v in summaries.values()),
        timing={p.stem:read(p)['elapsed_sec'] for p in (campaign.root/'receipts').glob('*.json')},
        notes=['No models trained or promoted; B2 is a studied diagnostic opponent, not a blind holdout.',
               'Conditional roots and commuted prefixes are correlated; no pooled Elo claim.',
               'All original driver repetition history restored; native search still has its existing limitation.',
               'Normal-start game budgets are ceilings with existing early-stop/finisher settings.',
               'Pure-MCTS root probes disable early stop/finisher and only finish the root actor turn.',
               'A search value or played draw is not a game-theoretic proof.',
               'Every declared block ran regardless of earlier outcomes. No score-dependent extensions.'])
    if (summary['games'], summary['root_probes']) != ((169, 31) if smoke else (1423, 108)):
        raise ValueError('Incorrect fixed study totals')
    pin(campaign.root / 'summary.json', summary)
    atomic_json(campaign.root / 'status.json', dict(status='complete',
        summary_sha256=file_hash(campaign.root/'summary.json')))
    print(f'MAINLINE {"REHEARSAL" if smoke else "PRODUCTION"} COMPLETE: {campaign.root}/summary.json', flush=True)


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
                                identity_fn=identity, label='MAINLINE')
            try:
                if smoke:
                    campaign.stage('tests', ['-m', 'pytest', 'tests', '-q'])
                elif not read(REHEARSAL/'summary.json')['complete']:
                    raise ValueError('Complete rehearsal required')
                work(campaign, smoke)
            except BaseException as exc:
                atomic_json(campaign.root/'status.json', dict(status='failed', error=str(exc)))
                raise


if __name__ == '__main__':
    main()
