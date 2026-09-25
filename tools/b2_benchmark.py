"""Bounded multi-opponent B2 checkpoint screens, finalist tests and self-skew."""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from match_evidence import atomic_json, file_hash, runtime_identity

ARMS = ('control', 'state_cnn', 'hybrid')
OPPONENTS = [f'models/bootstrap_v{v}/best_value_net.pt' for v in (24,25,26,27)]


def shortlist(directory):
    epochs = sorted(directory.glob('selected_epoch_*.pt'))
    if not epochs:
        raise ValueError(f'No epochs in {directory}')
    paths = [epochs[math.ceil(len(epochs)*f)-1] for f in (1/3,2/3,1)]
    metadata = sorted(directory.glob('train_run_*.json'))
    if len(metadata) != 1:
        raise ValueError(f'Expected one completed training metadata file: {directory}')
    best_epoch = json.loads(metadata[0].read_text())['best_epoch']
    # torch.save uses different archive names for best/epoch files; file hashes
    # alone cannot identify that they hold the same weights. Use its epoch ID.
    paths.append(directory/f'selected_epoch_{best_epoch:03d}.pt')
    result, seen = [], set()
    for path in paths:
        sha = file_hash(path)
        if sha not in seen:
            result.append(str(path))
            seen.add(sha)
    return result


def aggregate(reports):
    totals = {side: dict(wins=0, draws=0, losses=0, games=0) for side in ('white','black')}
    for report in reports:
        if report.get('partial') or report.get('games',0) <= 0:
            raise ValueError('Incomplete match report')
        for side, total in totals.items():
            cell = report[f'a_as_{side}']
            if cell['wins']+cell['draws']+cell['losses'] != cell['games']:
                raise ValueError('Invalid result counts')
            for k in total:
                total[k] += cell[k]
    for total in totals.values():
        total['score'] = float(Fraction(2*total['wins']+total['draws'], 2*total['games']))
    return totals


def rank(row):
    scores = [Fraction(2*c['wins']+c['draws'],2*c['games']) for c in row['scores'].values()]
    return (-min(scores), -sum(scores), row['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='benchmarks/b2_001_comparison_20260909')
    parser.add_argument('--require', default='iterations/b2_001/model_smoke.json')
    args = parser.parse_args()
    smoke = json.loads(Path(args.require).read_text())
    if smoke.get('complete') is not True:
        raise ValueError('All-arm inference validation must pass first')
    root = Path(args.output)
    candidates = {arm: shortlist(ROOT/f'models/candidates/b2_001_{arm}') for arm in ARMS}
    references = ['models/bootstrap_v27/best_value_net.pt',
                  'models/candidates/bootstrap_main_gen_0047/arena_selected.pt']
    for arm in ARMS:
        if file_hash(ROOT/f'models/candidates/b2_001_{arm}/best_value_net.pt') != smoke['results'][arm]['model_sha256']:
            raise ValueError('Candidate changed after smoke validation')
    models = sorted(set(references + OPPONENTS + [p for paths in candidates.values() for p in paths]))
    manifest = dict(runtime=runtime_identity(), implementation=file_hash(__file__),
        models={p:file_hash(p) for p in models}, shortlist=candidates, references=references,
        workers=8, sims=3200, screen_games=200, finalist_games=400, self_games=200,
        screen_seed=1980000000, finalist_seed=1990000000,
        ranking='exact W/D/L fractions: minimum color score, overall, path',
        scope='Selection benchmark only. No promotion, equal-time or second-seed confirmation.')
    manifest_path = root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Benchmark provenance changed')
    atomic_json(manifest_path, manifest)
    env = dict(os.environ, MONSTER_PINNED_INPUT='1')
    def command(arguments):
        subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT, env=env, check=True)
    def book(stage, count, seed):
        path = root/f'{stage}_book.json'
        receipt = path.with_suffix('.receipt.json')
        if receipt.exists():
            if json.loads(receipt.read_text())['sha256'] != file_hash(path):
                raise ValueError('Book changed')
        else:
            if path.exists():
                raise ValueError('Unreceipted book; inspect before resume')
            command(['tools/make_book.py', '--model', OPPONENTS[0], '--model', OPPONENTS[1],
                     '--model', OPPONENTS[2], '--entries',str(count),'--seed',str(seed),
                     '--sims','700','--workers','8','--out',str(path)])
            atomic_json(receipt, dict(sha256=file_hash(path)))
        return path
    def match(name, candidate, opponent, games, seed, book_path=None, offset=0):
        path = root/f'{name}.json'
        command_args = ['tools/match.py','--model-a',candidate,'--model-b',opponent,
            '--games',str(games),'--seed',str(seed),'--sims','3200','--workers','8',
            '--engine','native','--report-path',str(path),'--game-log',str(path.with_suffix('.jsonl')),'--resume']
        if book_path:
            command_args += ['--book',str(book_path),'--book-offset',str(offset)]
        print(f'BENCHMARK {name}', flush=True)
        command(command_args)
        report = json.loads(path.read_text())
        if report.get('partial') or report.get('games') != games:
            raise ValueError('Match did not finish')
        return report
    def panel(stage, name, candidate, seed, book_path, free, matched):
        reports=[]
        for j, opponent in enumerate(OPPONENTS):
            for kind, games in (('free',free),('matched',matched)):
                reports.append(match(f'{stage}_{name}_o{j}_{kind}',candidate,opponent,games,seed+10000+j*10000,
                    book_path if kind=='matched' else None, j*(matched//2)))
        return dict(path=candidate, scores=aggregate(reports))
    screen_book = book('screen',52,1980000000)
    screen = {}
    for arm, paths in candidates.items():
        screen[arm] = []
        for i, path in enumerate(paths):
            screen[arm].append(panel('screen',f'{arm}_{i}',path,1980000000,screen_book,24,26))
            atomic_json(root/'screen_progress.json',screen)
    for i,path in enumerate(references):
        screen[f'reference{i}'] = [panel('screen',f'reference{i}',path,1980000000,screen_book,24,26)]
    finalists = {arm: sorted(screen[arm],key=rank)[:2] for arm in ARMS}
    atomic_json(root/'screen_complete.json',dict(complete=True,results=screen,finalists=finalists))
    final_book = book('finalist',100,1990000000)
    final = {}
    for arm, rows in finalists.items():
        final[arm] = []
        for i,row in enumerate(rows):
            final[arm].append(panel('finalist',f'{arm}_{i}',row['path'],1990000000,final_book,50,50))
            atomic_json(root/'finalist_progress.json',final)
    for i,path in enumerate(references):
        final[f'reference{i}'] = [panel('finalist',f'reference{i}',path,1990000000,final_book,50,50)]
    selected={arm:sorted(final[arm],key=rank)[0] for arm in ARMS}
    selfplay={}
    for i,(arm,row) in enumerate(selected.items()):
        selfplay[arm]=match(f'self_{arm}',row['path'],row['path'],200,2000000000+i*1000000)
    atomic_json(root/'summary.json',dict(complete=True,screen=screen,finalists=final,
        selected=selected,selfplay=selfplay,manifest_sha256=file_hash(manifest_path),
        note='Equal-simulation selection evidence. Equal-time, second-seed and untouched confirmation remain required.'))


if __name__ == '__main__':
    main()
