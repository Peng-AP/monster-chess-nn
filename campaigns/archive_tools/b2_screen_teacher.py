"""Preregistered broad gen47 checkpoint selection for B2 data teachers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from match_evidence import atomic_json, file_hash, runtime_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require', required=True, help='Successful B2 model smoke receipt')
    args = parser.parse_args()
    if json.loads(Path(args.require).read_text()).get('complete') is not True:
        raise ValueError('B2 rehearsal did not pass')
    root = Path(args.output)
    folder = Path('models/candidates/bootstrap_main_gen_0047')
    candidates = []
    seen = set()
    for filename in ('selected_epoch_006.pt', 'selected_epoch_011.pt', 'selected_epoch_017.pt', 'best_value_net.pt'):
        path = folder / filename
        sha = file_hash(path)
        if sha not in seen:
            candidates.append(str(path))
            seen.add(sha)
    opponents = [f'models/bootstrap_v{v}/best_value_net.pt' for v in (24, 25, 26, 27)]
    manifest = dict(candidates={p: file_hash(p) for p in candidates},
                    opponents={p: file_hash(p) for p in opponents}, runtime=runtime_identity(),
                    implementation=file_hash(__file__), require_sha256=file_hash(args.require),
                    seed=1940000000, sims=3200, free_games=24, matched_games=26,
                    ranking='maximize minimum color score; then overall score; then path',
                    confirmation_seed_reservation=[1960000000, 1969999999])
    path = root/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Screen provenance changed')
    atomic_json(path, manifest)
    env = dict(os.environ, MONSTER_PINNED_INPUT='1')
    def execute(command):
        subprocess.run([sys.executable, *command], env=env, check=True)
    book = root/'selection_book.json'
    if not book.exists():
        execute(['tools/make_book.py', '--model', opponents[0], '--model', opponents[1],
                 '--model', opponents[2], '--entries', '52', '--sims', '700', '--seed', '1940000000',
                 '--workers', '8', '--out', str(book)])
    rows = []
    for index, candidate in enumerate(candidates):
        scores = {'white': [], 'black': []}
        for j, opponent in enumerate(opponents):
            for kind, games in (('free', 24), ('matched', 26)):
                report = root/f'candidate{index}_opponent{j}_{kind}.json'
                command = ['tools/match.py', '--model-a', candidate, '--model-b', opponent,
                           '--games', str(games), '--sims', '3200', '--engine', 'native', '--workers', '8',
                           '--seed', str(1941000000+j*10000), '--report-path', str(report),
                           '--game-log', str(report.with_suffix('.jsonl')), '--resume']
                if kind == 'matched':
                    command += ['--book', str(book), '--book-offset', str(j*13)]
                execute(command)
                result = json.loads(report.read_text())
                if result['games'] != games:
                    raise ValueError('Incomplete screen leg')
                for side in scores:
                    cell = result[f'a_as_{side}']
                    scores[side].append((cell['score'], games//2))
        averages = {side: sum(score*n for score,n in values)/sum(n for _,n in values)
                    for side,values in scores.items()}
        rows.append(dict(path=candidate, sha256=file_hash(candidate), **averages,
                         overall=(averages['white']+averages['black'])/2))
        atomic_json(root/'progress.json', dict(candidates=rows))
    selected = sorted(rows, key=lambda r: (-min(r['white'],r['black']), -r['overall'], r['path']))[0]
    atomic_json(root/'selection.json', dict(complete=True, selected=selected['path'],
                selected_sha256=selected['sha256'], candidates=rows,
                manifest_sha256=file_hash(path), book_sha256=file_hash(book),
                note='Teacher selection only, not a promotion or confirmation.'))


if __name__ == '__main__':
    main()
