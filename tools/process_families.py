"""Use the unchanged tensor processor with transitive source-family splitting.

Alternate completed continuations may legitimately have different outcomes.
Group by the original source game, stratify using that root's outcome, and keep
every member's original outcome and weights when converting to tensors.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import data_processor

_legacy_split = data_processor._split_games_by_result


def split_families(games, seed):
    by_id = {g['game_id']: g for g in games}
    if len(by_id) != len(games):
        raise ValueError('duplicate source game ID')
    def root(game):
        seen = set()
        while game.get('split_parent'):
            if game['game_id'] in seen:
                raise ValueError('source-family cycle')
            seen.add(game['game_id'])
            parent = game['split_parent']
            if parent not in by_id:
                raise ValueError(f'missing source parent: {parent}')
            game = by_id[parent]
        return game['game_id']
    families = {}
    for game in games:
        families.setdefault(root(game), []).append(game)
    representatives = [dict(by_id[k], split_parent=None) for k in sorted(families)]
    assignments = _legacy_split(representatives, seed)
    return {split: [member for representative in roots for member in families[representative['game_id']]]
            for split, roots in assignments.items()}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=3173)
    ap.add_argument('--channels', type=int, default=15)
    ap.add_argument('--value-floor', type=float, default=.5)
    ap.add_argument('--value-horizon', type=int, default=60)
    args = ap.parse_args()
    data_processor._split_games_by_result = split_families
    data_processor.process_raw_data(raw_dir=args.raw_dir, output_dir=args.output_dir,
                                seed=args.seed, input_channels=args.channels,
                                value_floor=args.value_floor, value_horizon=args.value_horizon,
                                value_discount_mode='near_mate', min_nonhuman_plies=0,
                                max_generation_age=0)
