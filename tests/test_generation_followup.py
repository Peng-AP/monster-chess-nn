from contextlib import nullcontext
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import generation_followup as follow
from generation_data_snapshot import snapshot


def arguments(tmp_path):
    return follow.parser().parse_args(['--candidate', 'candidate.pt', '--baseline', 'baseline.pt',
                                     '--opponent', 'opponent.pt', '--book-teacher', 'teacher.pt',
                                     '--run-dir', str(tmp_path / 'follow'), '--book-entries', '4', '--free-games', '4'])


def opt(command, name):
    return command[command.index(name) + 1]


def test_followup_matched_starts_share_book_and_seeds(tmp_path):
    stages = follow.plan(arguments(tmp_path))
    baseline, candidate = stages[1][1], stages[2][1]
    for key in ('--book', '--seed', '--book-offset', '--games', '--sims', '--model-b'):
        assert opt(baseline, key) == opt(candidate, key)
    assert opt(baseline, '--model-a') != opt(candidate, '--model-a')
    assert 'candidate.pt' not in stages[0][1]
    assert 'baseline.pt' not in stages[0][1]
    assert opt(stages[3][1], '--seed') != opt(candidate, '--seed')


def test_followup_chains_and_resumes_without_extra_games(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(follow, 'model_identity', lambda p: {'path': p, 'sha256': 'fixed'})
    monkeypatch.setattr(follow, 'runtime_identity', lambda: {'runtime': 'fixed'})
    monkeypatch.setattr(follow, 'worker_lease', nullcontext)
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        if '--out' in command:
            path = Path(opt(command, '--out'))
            report = {'entries': [{}] * args.book_entries}
        else:
            path = Path(opt(command, '--report-path'))
            report = {'games': int(opt(command, '--games')), 'a_score': .5,
                      'a_as_white': {'score': .6}, 'a_as_black': {'score': .4}}
        follow.atomic_json(path, report)
    monkeypatch.setattr(follow.subprocess, 'run', fake)
    follow.run(args)
    assert len(calls) == 4
    follow.run(args)
    assert len(calls) == 4
    state = json.loads((Path(args.run_dir) / 'state.json').read_text())
    assert state['status'] == 'complete' and not state['binding']
    Path(args.run_dir, 'independent_starts.json').write_text('{}')
    with pytest.raises(ValueError, match='output changed'):
        follow.run(args)


def test_followup_refuses_failed_iteration(tmp_path):
    args = arguments(tmp_path)
    state = tmp_path / 'iteration.json'
    state.write_text('{"status":"failed"}')
    args.iteration_state = str(state)
    with pytest.raises(ValueError, match='not safely completed'):
        follow.run(args)


def test_half_data_snapshot_is_reproducible_and_disjoint(tmp_path):
    for group in ('selfplay', 'bookseed'):
        directory = tmp_path / group
        directory.mkdir()
        for index in range(4):
            (directory / f'game_{index}.jsonl').write_text('{"fen":"fixture"}\n')
    result = snapshot(tmp_path, 3173)
    assert result == snapshot(tmp_path, 3173)
    for group in result['groups'].values():
        selected = {p['path'] for p in group['selected']}
        remaining = {p['path'] for p in group['remaining']}
        assert len(selected) == len(remaining) == 2
        assert not selected & remaining
