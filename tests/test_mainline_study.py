import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import mainline_study as study
import start_mainline_study as driver
from benchmark import play_one
from free_play_audit import audit_match
from gate_sampled import match_settings
from match import build_tasks
from match_evidence import atomic_json, digest, task_id
from monster_chess import MonsterChessGame
from repetition import RepetitionTracker

CYCLE = ['e1d1', 'd1e1', 'g8f6', 'e1d1', 'd1e1', 'f6g8']


class LoopEngine:
    def get_best_action(self, game, temperature):
        move = chess.Move.from_uci(CYCLE[len(game.board.move_stack) % len(CYCLE)])
        return move, {move.uci(): 1.}, 0.


def task_at(moves, kind='game'):
    return dict(case_id='unit', state=driver.make_state(moves), kind=kind,
                white_model='model', black_model='model', white_sims=8, black_sims=8,
                seed=123, deadline_seconds=60)


def test_history_half_and_repetition_restored():
    state = driver.make_state(CYCLE + CYCLE[:3])
    game, actual = study.checked_restore(state)
    expected = RepetitionTracker()
    original = MonsterChessGame()
    expected.record(original, 0)
    for i, uci in enumerate(state['moves']):
        original.apply_search_action(chess.Move.from_uci(uci))
        expected.record(original, i+1)
    assert game.board.move_stack == original.board.move_stack
    assert actual.counts == expected.counts
    half, _ = study.checked_restore(driver.make_state(['e2e4']))
    assert half.white_half_pending and half.turn_count == 0


@pytest.mark.parametrize('fault', ['history', 'clock', 'half', 'illegal', 'terminal'])
def test_invalid_prefix_rejected(fault):
    state = driver.make_state(CYCLE)
    if fault == 'history':
        state['initial_fen'] = '7k/8/8/8/8/8/8/K7 w - - 0 1'
    elif fault == 'clock':
        state['turn_count'] += 1
    elif fault == 'half':
        state['half'] = not state['half']
    elif fault == 'illegal':
        state['moves'][0] = 'a1a8'
    else:
        state['moves'] = CYCLE * 3
    with pytest.raises(ValueError):
        study.checked_restore(state)


def test_continuation_adjudicates_existing_repetition(monkeypatch):
    monkeypatch.setattr(study, '_engines', {True: LoopEngine(), False: LoopEngine()})
    task = task_at(CYCLE + CYCLE[:3])
    result = study.run_task(task)
    assert result['ending'] == 'repetition' and result['white_score'] == .5
    assert result['continuation_plies'] == 3
    study.audit_result(task, result)


@pytest.mark.parametrize('prefix,expected_plies', [([], 2), (CYCLE[:1], 1), (CYCLE[:2], 1)])
def test_probe_completes_only_root_actor_turn(monkeypatch, prefix, expected_plies):
    monkeypatch.setattr(study, '_engines', {'probe': LoopEngine()})
    task = task_at(prefix, 'probe')
    result = study.run_task(task)
    assert result['continuation_plies'] == expected_plies
    assert result['ending'] == 'diagnostic_limit'
    assert result['result_white'] is None and result['white_score'] is None
    study.audit_result(task, result)


@pytest.mark.parametrize('fault', ['task', 'digest', 'state', 'action', 'ending', 'result', 'repetition'])
def test_conditional_audit_rejects_bad_evidence(monkeypatch, fault):
    monkeypatch.setattr(study, '_engines', {True: LoopEngine(), False: LoopEngine()})
    task = task_at(CYCLE + CYCLE[:3])
    result = study.run_task(task)
    if fault == 'task':
        result['task_id'] = 'changed'
    elif fault == 'state':
        result['trajectory'][0]['turn_count'] += 1
    elif fault == 'action':
        result['trajectory'][1]['action'] = 'a1a8'
    elif fault == 'ending':
        result['ending'] = 'king_capture'
    elif fault == 'result':
        result['result_white'] = -1
    elif fault == 'repetition':
        result['final_repetition_sha256'] = 'changed'
    result['trajectory_sha256'] = 'changed' if fault == 'digest' else digest(result['trajectory'])
    with pytest.raises(ValueError):
        study.audit_result(task, result)


def test_saved_task_is_validated_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(study, '_engines', {True: LoopEngine(), False: LoopEngine()})
    task = task_at(CYCLE + CYCLE[:3])
    result = study.run_task(task)
    path = tmp_path/'task.json'
    atomic_json(path, dict(result_sha256=digest(result), result=result))
    assert study.load_result(path, task)['ending'] == 'repetition'
    saved = study.read(path)
    saved['result']['result_white'] = 1
    atomic_json(path, saved)
    with pytest.raises(ValueError, match='Changed'):
        study.load_result(path, task)


def fake_cases():
    return [dict(id=f'c{i}', family='early' if i < 4 else 'draw', state=driver.make_state([]))
            for i in range(9)]


@pytest.mark.parametrize('smoke,games,probes', [(False,1423,108), (True,169,31)])
def test_fixed_schedule_all_pairs_depths_and_disjoint_seeds(smoke,games,probes):
    cases = fake_cases()
    tasks = [t for stage in driver.STAGES for t in driver.make_tasks(stage,cases,smoke)]
    normal = driver.normal_schedule(smoke)
    assert sum(t['kind']=='game' for t in tasks)+sum(i['games'] for i in normal) == games
    assert sum(t['kind']=='probe' for t in tasks) == probes
    assert len({digest(t) for t in tasks}) == len(tasks)
    seeds = [t['seed'] for t in tasks]
    seeds += [t[1] for item in normal for t in build_tasks(item['games'],item['seed'],16)]
    assert len(seeds) == len(set(seeds)) and max(seeds) < 2**32
    for stage in driver.STAGES[:4]:
        block = driver.make_tasks(stage,cases,smoke)
        assert len({(t['white_model'],t['black_model']) for t in block}) == 9
    assert all(t['sample_index']==0 for t in driver.make_tasks('draw_crossplay',cases,smoke))


def test_cases_are_actual_legal_complete_history():
    cases = driver.build_cases()
    assert len(cases) == 9
    assert [len(c['state']['moves']) for c in cases] == [3,3,3,3,16,17,41,62,65]
    for case in cases:
        study.checked_restore(case['state'])
    roots = [c for c in cases if c['family']=='draw']
    assert len({c['source']['task_id'] for c in roots}) == 1
    assert all(c['source']['matching_endpoint_games']==56 for c in roots)


def test_probe_initialization_bypasses_finisher_and_early_stop(monkeypatch):
    calls = []
    class Inner:
        allow_early_stop = True
    wrapper = SimpleNamespace(_inner=Inner())
    monkeypatch.setattr(study, '_build_engine', lambda *a,**k: calls.append((a,k)) or (wrapper,'label'))
    study.initialize([('model',8),('model',8)],True)
    assert study._engines['probe'] is wrapper._inner
    assert not wrapper._inner.allow_early_stop
    study.initialize([('model',8),('model',8)],False)
    assert study._engines[True] is wrapper and study._engines[False] is wrapper


def test_same_checkpoint_still_builds_separate_color_searches(monkeypatch):
    made = []
    def build(*a,**kw):
        search = SimpleNamespace(_reuse_tree=None)
        made.append(search)
        return search, 'label'
    monkeypatch.setattr(study, '_build_engine', build)
    study.initialize([('same.pt',3200),('same.pt',3200)],False)
    assert len(made) == 2
    assert study._engines[True] is not study._engines[False]
    study._engines[True]._reuse_tree = 'white subtree'
    assert study._engines[False]._reuse_tree is None
    with pytest.raises(ValueError,match='White and Black'):
        study.initialize([('same.pt',3200)],False)


def test_unequal_search_is_not_equal_agent_selfplay(tmp_path):
    result, plies, _, opening = play_one(LoopEngine(),LoopEngine(),opening_temp_plies=16,return_opening=True)
    game = opening.pop('game')
    tasks = build_tasks(2,47,16)
    rows = [dict(pair=None,entry=None,a_is_white=t[0],seed=t[1],task_id=task_id(t),
                 result_for_a=result,white_score=.5,plies=plies,opening=opening,game=game) for t in tasks]
    path = tmp_path/'games.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    model = driver.MODELS['gen49']
    settings = match_settings(model,model,dict(games=2,seed=47),SimpleNamespace(sims=16,workers=8))
    settings['sims_b'] = 8
    atomic_json(path.with_suffix('.jsonl.manifest.json'),dict(schema_version=1,settings=settings,
        tasks=[task_id(t) for t in tasks]))
    checked = audit_match(path,model,model,2,47,16,8)
    assert checked['replay_audited'] and 'actual_color_self_par' not in checked
    with pytest.raises(ValueError,match='settings'):
        audit_match(path,model,model,2,47,16,16)
