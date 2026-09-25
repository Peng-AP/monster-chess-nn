import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import b2_campaign


def test_merge_preserves_family_namespaces(tmp_path):
    sources = []
    for index in range(2):
        source = tmp_path / str(index)
        source.mkdir()
        (source/'game.jsonl').write_text(json.dumps({'state': {'turn_count': 2}})+'\n')
        (source/'fork.jsonl').write_text(json.dumps({'source_record': {'path': 'game.jsonl', 'line': 1}})+'\n')
        sources.append(source)
    output = tmp_path/'merged'
    b2_campaign.merge_sources(sources, output)
    b2_campaign.merge_sources(sources, output)
    for index in range(2):
        row = json.loads((output/f'teacher{index}/fork.jsonl').read_text())
        assert row['source_record']['path'] == f'teacher{index}/game.jsonl'
    (sources[0]/'game.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Changed merge'):
        b2_campaign.merge_sources(sources, output)


def test_failed_stage_never_receipted(tmp_path, monkeypatch):
    import subprocess
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ['test'])
    monkeypatch.setattr(b2_campaign.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        b2_campaign.run_stage(tmp_path, 'failed', ['test'], [])
    assert not (tmp_path/'receipts/failed.json').exists()


def test_streamed_conversion_matches_reference(tmp_path):
    import numpy as np
    import b2_prepare
    import data_processor
    import sparse_policy
    from monster_chess import MonsterChessGame
    from stateful_generation import snapshot
    game = MonsterChessGame()
    row = dict(fen=game.fen(), half=0, current_player='white', mcts_value=.2,
               policy={'e1f1':1.0}, game_result=-1, plies_to_end=1,
               state=snapshot(game, game.fen(), []), played_action='e1f1')
    raw = tmp_path/'raw'
    raw.mkdir()
    (raw/'game.jsonl').write_text(json.dumps(row)+'\n')
    evidence = b2_prepare.audit(raw)
    output = tmp_path/'output'
    b2_prepare.convert(raw, output, 24, evidence)
    reference = data_processor._convert_games_to_arrays([dict(records=[row])], False,
        input_channels=24, value_horizon=60, value_floor=.5, value_discount_mode='near_mate',
        include_moves_left=True, include_legal_masks=True, include_capture_results=True)
    names = ('positions', 'mcts_values', 'game_results', 'policies', 'policy_weights',
             'value_weights', 'moves_left', 'moves_left_weights', 'legal_masks_packed', 'capture_results')
    for name, expected in zip(names, reference):
        actual = sparse_policy.load(output).to_dense() if name == 'policies' else np.load(output/f'{name}.npy')
        np.testing.assert_array_equal(actual, expected)
