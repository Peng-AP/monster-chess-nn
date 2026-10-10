import copy
import json
from pathlib import Path
import sys

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from stateful_generation import snapshot, restore_state, ordinary_batches, fork_tasks, play_task
from process_families import split_families
from monster_chess import MonsterChessGame
from repetition import RepetitionTracker


def test_full_state_restores_half_history_and_clocks():
    game = MonsterChessGame()
    initial = game.fen()
    moves = ['e2e4', 'e4e5', 'g8f6', 'd2d4']
    for move in moves:
        game.apply_search_action(chess.Move.from_uci(move))
    restored, _ = restore_state(snapshot(game, initial, moves))
    assert restored.fen() == game.fen()
    assert restored.white_half_pending
    assert restored.turn_count == game.turn_count == 2
    assert restored.board.move_stack == game.board.move_stack
    assert restored.get_search_actions() == game.get_search_actions()


def test_repetition_prefix_is_not_reset():
    game = MonsterChessGame()
    initial = game.fen()
    moves = ['e1d1', 'd1e1', 'g8f6', 'e1d1', 'd1e1', 'f6g8']
    tracker = RepetitionTracker()
    tracker.record(game, 0)
    for i, move in enumerate(moves):
        game.apply_search_action(chess.Move.from_uci(move))
        tracker.record(game, i+1)
    restored, actual = restore_state(snapshot(game, initial, moves))
    assert actual.counts == tracker.counts
    assert restored.turn_count == 4


def test_state_mismatch_fails():
    game = MonsterChessGame()
    state = snapshot(game, game.fen(), [])
    state['turn_count'] = 7
    with pytest.raises(ValueError, match='reconstruction'):
        restore_state(state)


def game_row(name, parent=None, outcome=1):
    return dict(game_id=name, split_parent=parent, result_bucket=outcome, records=[])


def test_forks_and_teachers_share_root_split_despite_different_outcomes():
    games = [game_row(f'root{i}') for i in range(20)]
    games += [game_row('fork', 'root0', -1), game_row('teacher', 'fork', -1), game_row('teacher2', 'root0')]
    splits = split_families(games, 3173)
    location = {g['game_id']: k for k, group in splits.items() for g in group}
    assert len({location[k] for k in ('root0', 'fork', 'teacher', 'teacher2')}) == 1
    assert len(location) == len(games)
    assert games[-2]['result_bucket'] == -1


@pytest.mark.parametrize('games', [[game_row('a', 'missing')], [game_row('a', 'b'), game_row('b', 'a')]])
def test_invalid_family_rejected(games):
    with pytest.raises(ValueError):
        split_families(games, 3173)


def test_production_recipe_counts_colors_models_and_seeds():
    config = json.loads((ROOT / 'tools/recipes/gen47.json').read_text())
    tasks = [t for batch in ordinary_batches(config) for t in batch]
    assert len(tasks) == 4800
    league = [t for t in tasks if t['kind'] == 'league']
    assert len(league) == 1120
    assert sum(t['train_side'] == 'black' for t in league) == 560
    assert len({t['seed'] for t in tasks}) == len(tasks)
    assert len({t['id'] for t in tasks}) == len(tasks)
    assert len({t['other'] for t in league}) == 3


def test_forks_are_unique_parents_and_preserve_state(tmp_path):
    folder = tmp_path / 'selfplay'
    folder.mkdir()
    for index in range(12):
        rows = [dict(half=0, current_player=side, state=dict(moves=['x']*9, marker=index)) for side in ('white', 'black')]
        (folder / f'game_{index}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    config = dict(seed=123, fork_games=10, model='model', fork_sims=3200)
    tasks = fork_tasks(config, tmp_path)
    assert len(tasks) == 10
    assert len({t['source_record']['path'] for t in tasks}) == 10
    assert tasks == fork_tasks(config, tmp_path)
    assert sum(t['source_record']['line'] == 2 for t in tasks) == 6


def test_older_policy_masked_but_loss_retained(monkeypatch):
    import stateful_generation as generation
    import data_generation
    class FixedEngine:
        def get_best_action(self, game, temperature):
            move = chess.Move.from_uci('e2e1')
            return move, {move.uci(): 1.0}, 1.0
    monkeypatch.setattr(generation, 'engine', lambda *a: FixedEngine())
    monkeypatch.setattr(data_generation, '_finisher_settings', lambda: (False, 4, 100, 8))
    game = MonsterChessGame(fen='7k/8/8/8/8/8/4r3/4K3 b - - 0 1')
    rows = play_task(dict(id='test', kind='league', seed=12, model='new', other='old',
                          train_side='white', sims=8, state=snapshot(game, game.fen(), [])))
    assert rows[0]['policy_weight'] == 0
    assert rows[0]['game_result'] == -1
    assert rows[0]['termination'] == 'king_capture'
    assert rows[0]['plies_to_end'] == 0


def test_nonterminal_no_action_is_not_a_draw(monkeypatch):
    import stateful_generation as generation
    import data_generation
    class EmptyEngine:
        def get_best_action(self, game, temperature):
            return None, {}, 0
    monkeypatch.setattr(generation, 'engine', lambda *a: EmptyEngine())
    monkeypatch.setattr(data_generation, '_finisher_settings', lambda: (False, 4, 100, 8))
    with pytest.raises(ValueError, match='false outcome'):
        play_task(dict(id='test', kind='selfplay', seed=12, model='new', sims=8))


def test_family_processor_real_cli(tmp_path):
    import subprocess
    raw = tmp_path / 'raw'
    raw.mkdir()
    base = dict(fen=MonsterChessGame().fen(), current_player='white', half=0,
                policy={'e2e4': 1.0}, mcts_value=0, game_result=1,
                plies_to_end=0)
    for i in range(12):
        (raw / f'parent{i}.jsonl').write_text(json.dumps(base) + '\n')
    fork = dict(base, game_result=-1, source_record={'path': 'parent0.jsonl', 'line': 1})
    (raw / 'fork.jsonl').write_text(json.dumps(fork) + '\n')
    teacher = dict(fork, source='deep_search_reanalysis', value_weight=0,
                   source_record={'path': 'fork.jsonl', 'line': 1})
    (raw / 'teacher.jsonl').write_text(json.dumps(teacher) + '\n')
    output = tmp_path / 'processed'
    subprocess.run([sys.executable, str(ROOT / 'tools/process_families.py'),
                    '--raw-dir', str(raw), '--output-dir', str(output)], check=True, capture_output=True)
    manifest = json.loads((output / 'split_game_ids.json').read_text())
    locations = {name: split for split in ('train', 'val', 'test') for name in manifest[split]}
    assert locations['parent0.jsonl'] == locations['fork.jsonl'] == locations['teacher.jsonl']


def test_stateful_reanalysis_uses_history_and_clears_reuse(monkeypatch):
    import reanalyze_stateful
    game = MonsterChessGame()
    initial = game.fen()
    moves = ['e2e4', 'e4e5', 'g8f6']
    for uci in moves:
        game.apply_search_action(chess.Move.from_uci(uci))
    class FakeEngine:
        _reuse_tree = 'old'
        _reuse_key = 'old'
        def get_best_action(self, restored, temperature):
            assert self._reuse_tree is None and self._reuse_key is None
            assert restored.turn_count == 2
            assert restored.board.move_stack == game.board.move_stack
            return None, {'d2d4': 1.0}, .2
    monkeypatch.setattr(reanalyze_stateful.reanalyze, '_worker_engine', FakeEngine())
    record = dict(fen=game.fen(), half=0, current_player='white', policy={'d2d4': 1.0},
                  state=snapshot(game, initial, moves), game_result=0, plies_to_end=4)
    result = reanalyze_stateful.reanalyze_one(dict(path='parent.jsonl', line=4, record=record))
    assert result['source_path'] == 'parent.jsonl'
    assert result['plies_to_end'] == 4


def test_exploration_length_is_opt_in_and_leaves_old_task_digests_unchanged():
    import stateful_generation as sg
    from match_evidence import digest
    base = dict(model='m.pt', sims=8, seed=100, free_games=2, fresh_games=0, league_games=0,
                prefix_models=[], opponents=[])
    old = sg.ordinary_batches(base)[0]
    assert all('temperature_plies' not in t for t in old)
    assert digest(old[0]) == digest(dict(id='selfplay/game_00000', kind='selfplay', model='m.pt', sims=8, seed=100))
    explored = sg.ordinary_batches(dict(base, temperature_plies=30))[0]
    assert [t['temperature_plies'] for t in explored] == [30, 30]
    import pytest
    with pytest.raises(ValueError):
        sg.ordinary_batches(dict(base, temperature_plies=-1))
    assert all('late_temperature' not in t for t in explored)
    greedy = sg.ordinary_batches(dict(base, temperature_plies=16, late_temperature=0))[0]
    assert [t['late_temperature'] for t in greedy] == [0.0, 0.0]
    for bad in (-0.1, 2, True, '0'):
        with pytest.raises(ValueError):
            sg.ordinary_batches(dict(base, late_temperature=bad))
    assert all('root_noise' not in t for t in explored)
    quiet = sg.ordinary_batches(dict(base, root_noise=False))[0]
    assert [t['root_noise'] for t in quiet] == [False, False]
    with pytest.raises(ValueError):
        sg.ordinary_batches(dict(base, root_noise=0))
