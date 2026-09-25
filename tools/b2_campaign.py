"""Resumable B2 generation/preparation/three-arm training; no automatic promotion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from match_evidence import atomic_json, file_hash, runtime_identity


def run_stage(root, name, command, artifacts):
    receipt = root / 'receipts' / f'{name}.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['command'] != command or any(not Path(p).exists() or file_hash(p) != h
                                              for p, h in saved['artifacts'].items()):
            raise ValueError(f'Changed stage {name}')
        return
    print(f'STAGE {name}: {command}', flush=True)
    env = dict(os.environ, MONSTER_PINNED_INPUT='1')
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    files = []
    for artifact in artifacts:
        if artifact.is_dir():
            files.extend(p for p in artifact.rglob('*') if p.is_file())
        else:
            files.append(artifact)
    hashes = {str(p): file_hash(p) for p in files}
    atomic_json(receipt, dict(command=command, artifacts=hashes))


def merge_sources(sources, output):
    for index, source in enumerate(sources):
        for path in sorted(source.rglob('*.jsonl')):
            destination = output / f'teacher{index}' / path.relative_to(source)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for row in rows:
                if row.get('source_record'):
                    row['source_record']['path'] = f"teacher{index}/" + row['source_record']['path']
            content = ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows)
            if destination.exists():
                if destination.read_text() != content:
                    raise ValueError(f'Changed merge output {destination}')
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix('.tmp')
                temporary.write_text(content, encoding='utf-8')
                os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--teacher', help='Defaults to the selected model in --teacher-evidence')
    parser.add_argument('--teacher-evidence', help='Broad teacher-selection receipt; required for production')
    parser.add_argument('--rehearsal', action='store_true')
    parser.add_argument('--through', choices=('generate', 'prepare', 'train'), default='train')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if root == ROOT or ROOT not in root.parents:
        raise ValueError('Campaign root must be a new subdirectory of this repository')
    if not args.rehearsal and not args.dry_run:
        if not args.teacher_evidence:
            raise ValueError('Production requires broad teacher-selection evidence')
        evidence = json.loads(Path(args.teacher_evidence).read_text())
        args.teacher = args.teacher or evidence.get('selected')
        if evidence.get('complete') is not True or evidence.get('selected_sha256') != file_hash(args.teacher):
            raise ValueError('Teacher evidence incomplete or mismatched')
    args.teacher = args.teacher or 'models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
    teachers = ['models/bootstrap_v27/best_value_net.pt', args.teacher]
    recipe = json.loads((ROOT / 'tools/recipes/gen47.json').read_text())
    recipes = []
    for index, teacher in enumerate(teachers):
        config = dict(recipe, model=teacher, seed=1900000000 + index * 1000000,
                      free_games=3000, fresh_games=750, league_games=1500, fork_games=750,
                      fork_sampling='mixed_outcome_surprise')
        if args.rehearsal:
            config.update(free_games=4, fresh_games=4, league_games=4, fork_games=2,
                          sims=8, fork_sims=8)
        recipes.append(config)
    plan = dict(recipes=recipes, arms=[dict(name='control', channels=15, attention=0),
                dict(name='state_cnn', channels=24, attention=0),
                dict(name='hybrid', channels=24, attention=2)], rehearsal=args.rehearsal,
                runtime=runtime_identity(),
                tools={p.name: file_hash(p) for p in [Path(__file__), ROOT/'tools/b2_prepare.py',
                       ROOT/'tools/reanalyze_b2.py', ROOT/'tools/stateful_generation.py',
                       ROOT/'tools/reanalyze_stateful.py', ROOT/'tools/reanalyze.py',
                       ROOT/'tools/process_families.py']},
                models={p: file_hash(p) for p in set(teachers + recipe['opponents'] + recipe['prefix_models'])})
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    manifest = root / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError('Campaign provenance changed; refusing mixed-runtime resume')
    atomic_json(manifest, plan)
    python = sys.executable
    sources = []
    for index, config in enumerate(recipes):
        work = root / f'generation{index}'
        source = work / 'raw'
        sources.append(source)
        config_path = work / 'recipe.json'
        atomic_json(config_path, config)
        summary = work / 'summary.json'
        run_stage(root, f'generate{index}', [python, 'tools/stateful_generation.py', '--config', str(config_path),
                  '--raw', str(source), '--summary', str(summary)], [summary, source])
        run_stage(root, f'reanalyze{index}', [python, 'tools/reanalyze_b2.py', '--source-dir', str(source),
                  '--model', config['model'], '--output-dir', str(source/'deep'), '--sample',
                  '8' if args.rehearsal else '40000', '--keep', '4' if args.rehearsal else '20000',
                  '--simulations', '8' if args.rehearsal else '3200', '--workers', '8',
                  '--seed', str(1920000000+index*1000000), '--journal', str(work/'reanalyze_journal'),
                  '--resume'], [source/'deep'])
    if args.through == 'generate':
        return
    raw = root / 'raw'
    merge_sources(sources, raw)
    for channels in (15, 24):
        processed = root / f'processed{channels}'
        run_stage(root, f'prepare{channels}', [python, 'tools/b2_prepare.py', '--raw', str(raw),
                  '--output', str(processed), '--channels', str(channels)], [processed])
    a = json.loads((root/'processed15/b2_audit.json').read_text())
    b = json.loads((root/'processed24/b2_audit.json').read_text())
    if a != b:
        raise ValueError('Arm corpus or splits differ')
    if args.through == 'prepare':
        return
    model_paths = {}
    for arm in plan['arms']:
        model_dir = (root / 'models' / arm['name'] if args.rehearsal else
                     ROOT / 'models' / 'candidates' / f"{root.name}_{arm['name']}")
        model_paths[arm['name']] = model_dir/'best_value_net.pt'
        command = [python, 'src/train.py', '--data-dir', str(root/f"processed{arm['channels']}"),
                   '--model-dir', str(model_dir), '--attention-blocks', str(arm['attention']),
                   '--epochs', '1' if args.rehearsal else '30', '--patience', '10', '--batch-size', '256',
                   '--lr', '0.002', '--ema-decay', '0.999', '--warmup-epochs', '3', '--seed', '3173',
                   '--policy-head', 'attention', '--policy-attention-channels', '64', '--stem-channels', '64',
                   '--res-channels', '64,64,128,128,128,128,128,128', '--target', 'game_result',
                   '--save-selection-snapshots', '--memory-map-data']
        run_stage(root, f"train_{arm['name']}", command, [model_dir])
    atomic_json(root/'training_complete.json', dict(complete=True, rehearsal=args.rehearsal,
                models={name: file_hash(path) for name,path in model_paths.items()},
                paths={name: str(path) for name,path in model_paths.items()}))


if __name__ == '__main__':
    main()
