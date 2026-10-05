import copy
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import start_gen49 as campaign
from benchmark import play_one
from free_play_audit import audit_game, audit_match
from gate_sampled import match_settings
from iterate_stateful import validate_recipe
from match import build_tasks, game_score
from match_evidence import atomic_json, digest, task_id
from stateful_generation import ordinary_batches
from types import SimpleNamespace


@pytest.mark.local_artifacts
def test_mainline_recipe_has_no_external_starts_or_opponents():
    config = campaign.read(ROOT / 'tools/recipes/gen49.json')
    validate_recipe(config)
    tasks = [t for batch in ordinary_batches(config) for t in batch]
    assert len(tasks) == 2800
    assert {t['kind'] for t in tasks} == {'selfplay'}
    assert all(t['model'] == campaign.TEACHER and 'other' not in t for t in tasks)
    assert config['fork_games'] == 400 and config['fork_sims'] == 6400
    assert config['coverage_reanalysis']
    assert len({t['seed'] for t in tasks}) == len(tasks)


@pytest.mark.parametrize('count', ['fresh_games', 'league_games'])
def test_empty_models_rejected_when_pool_enabled(count):
    config = campaign.read(ROOT / 'tools/recipes/gen49.json')
    config[count] = 2
    with pytest.raises(ValueError, match='empty'):
        validate_recipe(config)


def test_scratch_training_recipe_unchanged():
    command = campaign.iteration_command(False)
    args = campaign.iterate.build_parser().parse_args(command[command.index('--') + 1:])
    assert (args.epochs, args.patience, args.seed, args.lr, args.batch_size) == (30, 10, 3173, .002, 256)
    assert args.replay_generations == 8 and args.anchor_data == 'none'
    assert args.through_phase == 'checkpoint_screen' and not args.promote_on_pass
    assert not args.reject_on_offline_regression
    assert (args.reanalysis_sample, args.reanalysis_keep, args.reanalysis_sims) == (24000, 12000, 6400)
    assert args.book_seed_games == 0 and args.games == 3200
    assert '--init' not in command


@pytest.mark.parametrize('smoke,total', [(True, 48), (False, 3000)])
def test_fixed_counts_and_seed_blocks(smoke, total):
    spec = campaign.layout(smoke)
    assert spec['research_par'] + 4 * spec['research_per_side'] + spec['final_par'] + 4 * spec['final_per_side'] + 6 * spec['extras'] == total
    blocks = [dict(games=spec['research_par'], seed=spec['seed'] + 100000),
              dict(games=2 * spec['research_per_side'], seed=spec['seed']),
              dict(games=2 * spec['research_per_side'], seed=spec['seed'] + 200000),
              dict(games=spec['final_par'], seed=spec['seed'] + 4100000),
              dict(games=2 * spec['final_per_side'], seed=spec['seed'] + 4000000),
              dict(games=2 * spec['final_per_side'], seed=spec['seed'] + 4200000)]
    blocks += [dict(games=spec['extras'], seed=spec['seed'] + i * 1000000) for i in (1, 2, 3, 5, 6, 7)]
    seeds = [t[1] for b in blocks for t in build_tasks(b['games'], b['seed'], 16)]
    assert len(seeds) == total and len(set(seeds)) == total
    assert max(seeds) < 2**32


class RandomEngine:
    def __init__(self):
        self.rng = random.Random(45)

    def get_best_action(self, game, temperature):
        move = self.rng.choice(game.get_search_actions())
        return move, None, None


@pytest.fixture(scope='module')
def good_row():
    result, plies, _, opening = play_one(RandomEngine(), RandomEngine(),
        opening_temp_plies=16, return_opening=True)
    game = opening.pop('game')
    return dict(pair=None, entry=None, a_is_white=True, result_for_a=result,
                white_score=game_score(result), plies=plies, opening=opening, game=game)


def test_full_free_trajectory_replays(good_row):
    assert audit_game(good_row) in ('white_capture', 'black_capture', 'turn_cap', 'repetition')


@pytest.mark.parametrize('fault', ['start', 'clock', 'history', 'repetition', 'action', 'outcome', 'digest', 'book'])
def test_free_audit_rejects_corrupted_trajectory(good_row, fault):
    row = copy.deepcopy(good_row)
    if fault == 'start':
        row['game']['trajectory'][0]['fen'] = '7k/8/8/8/8/8/8/K7 w - - 0 1'
    elif fault == 'clock':
        row['game']['trajectory'][1]['turn_count'] += 1
    elif fault in ('history', 'repetition'):
        row['opening'][fault + '_sha256'] = 'changed'
    elif fault == 'action':
        row['game']['trajectory'][1]['action'] = ['a1a8']
    elif fault == 'outcome':
        row['result_for_a'] = .5 if row['result_for_a'] != .5 else 1
    elif fault == 'book':
        row['entry'] = 0
    row['game']['trajectory_sha256'] = 'changed' if fault == 'digest' else digest(row['game']['trajectory'])
    with pytest.raises(ValueError):
        audit_game(row)


def write_journal(tmp_path, good_row):
    path = tmp_path / 'games.jsonl'
    tasks = build_tasks(2, 42, 16)
    rows = []
    for task in tasks:
        row = copy.deepcopy(good_row)
        row.update(a_is_white=task[0], seed=task[1], task_id=task_id(task))
        if not task[0]:
            row['result_for_a'] = -row['result_for_a']
        rows.append(row)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    model = campaign.TEACHER
    settings = match_settings(model, model, dict(games=2, seed=42), SimpleNamespace(sims=8, workers=8))
    atomic_json(path.with_suffix('.jsonl.manifest.json'),
                dict(schema_version=1, settings=settings, tasks=[task_id(t) for t in tasks]))
    return path, rows


@pytest.mark.local_artifacts
def test_journal_audit_and_actual_color_par(tmp_path, good_row):
    path, _ = write_journal(tmp_path, good_row)
    out = audit_match(path, campaign.TEACHER, campaign.TEACHER, 2, 42, 8)
    assert out['replay_audited']
    assert out['diagnostics']['sampled']['score'] == .5
    par = out['actual_color_self_par']
    assert par['n'] == 2
    assert par['sides']['white']['score'] == game_score(good_row['result_for_a'])
    assert par['sides']['white']['score'] + par['sides']['black']['score'] == 1
    assert out['opening_concentration']['unique_actual_endpoints'] == 1


@pytest.mark.local_artifacts
@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'seed', 'settings'])
def test_journal_audit_rejects_wrong_task_evidence(tmp_path, good_row, fault):
    path, rows = write_journal(tmp_path, good_row)
    if fault == 'duplicate':
        rows[1] = rows[0]
    elif fault == 'missing':
        rows.pop()
    elif fault == 'seed':
        rows[0]['seed'] += 1
    else:
        meta = path.with_suffix('.jsonl.manifest.json')
        data = campaign.read(meta)
        data['settings']['sims'] = 16
        atomic_json(meta, data)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError):
        audit_match(path, campaign.TEACHER, campaign.TEACHER, 2, 42, 8)


def test_measured_failure_does_not_skip_other_models(tmp_path, monkeypatch):
    job = SimpleNamespace(root=tmp_path, provenance={})
    candidate = tmp_path / 'candidate.pt'
    candidate.write_bytes(b'model')
    atomic_json(tmp_path / 'manifest.json', {})
    atomic_json(tmp_path / 'gen48_free_results.json', {})
    (tmp_path / 'receipts').mkdir()
    calls = []
    monkeypatch.setattr(campaign, 'free_gate', lambda *a, **k: {'report': {'verdict': 'FAIL'}})
    monkeypatch.setattr(campaign, 'free_match', lambda job, name, *a: calls.append(name) or {})
    monkeypatch.setattr(campaign, 'identity', lambda: {})
    campaign.postcheck(job, candidate, True, {'games': 24})
    assert calls == ['gen49_vs_b2', 'gen49_vs_v27', 'gen49_self']
    assert campaign.read(tmp_path / 'summary.json')['complete']


def test_interrupted_training_is_preserved(tmp_path, monkeypatch):
    paths = {'state': tmp_path / 'state.json'}
    atomic_json(paths['state'], dict(phases={'train': {'status': 'running'}}))
    monkeypatch.setattr(campaign.iterate, '_paths_for_generation', lambda *a: paths)
    with pytest.raises(ValueError, match='preserved'):
        campaign.train(SimpleNamespace(root=tmp_path), False)


def test_cli_match_records_both_simulation_budgets(tmp_path, monkeypatch):
    commands = []
    def stage(name, command, outputs):
        commands.append(command)
        atomic_json(outputs[0], dict(games=4, book=None))
    monkeypatch.setattr(campaign, 'audit_match', lambda *a: {'replay_audited': True})
    result = campaign.free_match(SimpleNamespace(root=tmp_path, stage=stage),
                                  'test', 'a.pt', 'b.pt', 4, 1000, 8)
    cmd = commands[0]
    assert cmd[cmd.index('--sims') + 1] == cmd[cmd.index('--sims-b') + 1] == '8'
    assert '--book' not in cmd and result['replay_audited']
