"""Preserve a reproducible game-level half-data split; never edits training data."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from match_evidence import atomic_json, model_identity


def snapshot(raw, seed):
    raw = Path(raw).resolve()
    groups = {}
    for name in ('selfplay', 'bookseed'):
        files = sorted((raw / name).glob('*.jsonl'))
        if not files or len(files) % 2:
            raise ValueError(f'{name} needs a nonempty even number of complete games')
        files.sort(key=lambda p: hashlib.sha256(f'{seed}:{name}/{p.name}'.encode()).hexdigest())
        identities = [model_identity(p) for p in files]
        groups[name] = {'selected': identities[:len(files)//2], 'remaining': identities[len(files)//2:]}
    return {'seed': seed, 'raw_directory': str(raw), 'groups': groups,
            'interpretation': 'ordinary source-game partition only, not a trained control; any future teacher subset must preserve source-game ancestry and split membership'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--seed', type=int, default=3173)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('snapshot already exists; do not overwrite')
    atomic_json(args.output, snapshot(args.raw_dir, args.seed))
    print(f'Saved source-game half-data manifest: {args.output}', flush=True)


if __name__ == '__main__':
    main()
