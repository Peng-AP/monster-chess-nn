"""Audit state-complete sources and create identical unmirrored B2 arm splits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
import data_processor
from process_families import split_families
from stateful_generation import restore_state
from match_evidence import atomic_json, file_hash


def audit(raw):
    paths = sorted(Path(raw).rglob('*.jsonl'))
    if not paths:
        raise ValueError('No source records')
    counts = {}
    games = []
    row_counts = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f'Empty source {path}')
        for row in rows:
            if 'state' not in row:
                raise ValueError(f'Missing complete state: {path}')
            state = row['state']
            if state['fen'] != row['fen'] or int(state['half']) != int(row['half']):
                raise ValueError(f'Inconsistent state: {path}')
            # Validate endpoints by full replay; intermediate records are checked
            # for exact prefix progression below, avoiding quadratic replay cost.
        import chess
        game, tracker = restore_state(rows[0]['state'])
        for index, row in enumerate(rows):
            state = row['state']
            if (game.fen() != row['fen'] or game.turn_count != state['turn_count']
                    or int(game.white_half_pending) != int(row['half'])
                    or game.is_white_turn != (row['current_player'] == 'white')):
                raise ValueError(f'Replayed intermediate state differs: {path}:{index+1}')
            if index + 1 < len(rows):
                current = rows[index+1]
                if current['state']['moves'] != state['moves'] + [row['played_action']]:
                    raise ValueError(f'Non-contiguous action prefix: {path}')
                if game.is_terminal() or tracker.fired_at is not None:
                    raise ValueError(f'Game continues after termination: {path}')
                game.apply_search_action(chess.Move.from_uci(row['played_action']))
                tracker.record(game, len(current['state']['moves']))
        key = (rows[0].get('source', 'unknown'), rows[0]['game_result'])
        counts[str(key)] = counts.get(str(key), 0) + 1
        game_id = path.relative_to(raw).as_posix()
        row_counts[game_id] = len(rows)
        games.append(dict(game_id=game_id, result_bucket=data_processor._result_bucket(rows[-1]['game_result']),
                          split_parent=rows[0].get('source_record', {}).get('path')))
    splits = split_families(games, 3173)
    return dict(sources={str(p.relative_to(raw)): file_hash(p) for p in paths},
                counts=counts, row_counts=row_counts,
                splits={k: [g['game_id'] for g in v] for k, v in splits.items()})


def convert(raw, output, channels, evidence):
    """One game's dense policies in RAM, never the whole multi-million-row corpus."""
    import numpy as np
    import sparse_policy
    output.mkdir(parents=True)
    total = sum(evidence['row_counts'].values())
    names = ('positions', 'mcts_values', 'game_results', 'policies', 'policy_weights',
             'value_weights', 'moves_left', 'moves_left_weights', 'legal_masks_packed', 'capture_results')
    arrays = {}
    for name in names:
        if name == 'policies':
            continue
        shape = ((total,8,8,channels) if name == 'positions' else
                 (total,512) if name == 'legal_masks_packed' else (total,))
        arrays[name] = np.lib.format.open_memmap(output/f'{name}.npy', mode='w+',
            dtype=np.uint8 if name == 'legal_masks_packed' else np.float32, shape=shape)
    policies = sparse_policy.Builder(4096)
    offset = 0
    splits = {}
    for split, paths in evidence['splits'].items():
        start = offset
        for path in paths:
            rows = [json.loads(line) for line in (raw/path).read_text().splitlines() if line.strip()]
            values = data_processor._convert_games_to_arrays([dict(game_id=path, records=rows)],
                False, value_horizon=60, value_floor=.5, value_discount_mode='near_mate',
                input_channels=channels, include_moves_left=True, include_legal_masks=True,
                include_capture_results=True)
            n = len(rows)
            for name, block in zip(names, values):
                if name == 'policies':
                    policies.add_dense(block)
                else:
                    arrays[name][offset:offset+n] = block
            offset += n
        splits[split] = np.arange(start,offset,dtype=np.int64)
    for array in arrays.values():
        array.flush()
    policies.save(str(output))
    np.savez(output/'splits.npz', **splits)
    atomic_json(output/'split_game_ids.json', dict(**evidence['splits'], augment=False,
        input_channels=channels, total_positions=total, value_discount_mode='near_mate',
        value_floor=.5, value_horizon=60, policy_size=4096, promotion_aware_policy=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--channels', required=True, type=int, choices=(15, 24))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    evidence = audit(Path(args.raw))
    convert(Path(args.raw), output, args.channels, evidence)
    atomic_json(output / 'b2_audit.json', evidence)


if __name__ == '__main__':
    main()
