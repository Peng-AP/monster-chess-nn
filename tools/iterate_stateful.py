"""Opt-in data adapters; retain the canonical iteration's training and gates.

The command-plan extension is local to this process, so in-flight older runs
keep their runtime identity. Resume through this entry point with the same recipe.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import iterate
from match_evidence import atomic_json, file_hash

ADAPTERS = ('tools/iterate_stateful.py', 'tools/stateful_generation.py',
            'tools/reanalyze_stateful.py', 'tools/reanalyze.py',
            'tools/reanalysis_journal.py', 'tools/process_families.py',
            'tools/reanalyze_coverage.py')


def validate_recipe(recipe):
    for key in ('free_games', 'fresh_games', 'league_games', 'fork_games'):
        if not isinstance(recipe[key], int) or recipe[key] < 0:
            raise ValueError(f'invalid {key}')
    if not 1 <= recipe['workers'] <= 8 or recipe['sims'] <= 0 or recipe['fork_sims'] <= 0:
        raise ValueError('workers 1..8 and positive simulations required')
    if recipe['league_games'] % 2:
        raise ValueError('league must have equal teacher colors')
    if recipe['fork_games'] > recipe['free_games'] + recipe['fresh_games'] + recipe['league_games']:
        raise ValueError('at most one fork per parent game')
    for key, count in (('prefix_models', 'fresh_games'), ('opponents', 'league_games')):
        if recipe[count] and not recipe[key]:
            raise ValueError(f'empty {key}')
    if not 0 <= recipe['seed'] < 2**32 - 1000000:
        raise ValueError('seed namespace overflows')
    for path in [recipe['model'], *recipe['prefix_models'], *recipe['opponents']]:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)


def extend_plan(base, recipe_path, recipe, args, generation, incumbent, architecture, paths, replay_sources):
    plan = base(args, generation, incumbent, architecture, paths, replay_sources)
    if architecture['promotion_policy']:
        raise ValueError('stateful processor adapter currently supports legacy attention policy only')
    if Path(incumbent).resolve() != (ROOT / recipe['model']).resolve() or args.workers != recipe['workers'] or args.sims != recipe['sims']:
        raise ValueError('iteration and recipe model/workers/sims differ')
    summary = str(paths['reports'] / 'stateful_generation_summary.json')
    plan['generate'] = dict(commands=[['tools/stateful_generation.py', '--config', str(recipe_path),
                                      '--raw', str(paths['raw']), '--summary', summary]], outputs=[summary])
    plan['reanalyze']['commands'][0][0] = 'tools/reanalyze_stateful.py'
    if recipe.get('coverage_reanalysis', False):
        coverage = str(paths['reports'] / 'reanalysis_coverage.json')
        plan['reanalyze']['commands'][0][0] = 'tools/reanalyze_coverage.py'
        plan['reanalyze']['commands'][0] += ['--coverage-report', coverage]
        plan['reanalyze']['outputs'].append(coverage)
    plan['process']['commands'][0] = [
        'tools/process_families.py', '--raw-dir', str(paths['raw']),
        '--output-dir', str(paths['new_processed']), '--seed', str(args.seed),
        '--channels', str(architecture['input_channels']), '--value-floor', str(args.value_floor),
        '--value-horizon', str(args.value_horizon)]
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--expected-generation', type=int, required=True)
    ap.add_argument('--after-state')
    args, remaining = ap.parse_known_args()
    if remaining[:1] == ['--']:
        remaining = remaining[1:]
    recipe_path = (ROOT / args.recipe).resolve()
    recipe = json.loads(recipe_path.read_text())
    validate_recipe(recipe)
    parsed = iterate.build_parser().parse_args(remaining)
    iterate._validate_args(parsed)
    if not parsed.dry_run and args.after_state:
        previous = json.loads((ROOT / args.after_state).read_text())
        if previous.get('status') != 'complete':
            raise ValueError('predecessor diagnostics did not complete successfully')
    run_root = iterate._absolute(parsed.run_root)
    generation = iterate._latest_generation(run_root) if parsed.resume else iterate._next_generation(run_root)
    if generation != args.expected_generation:
        raise ValueError(f'expected gen{args.expected_generation}, next/resume is {generation}')
    evidence_path = run_root / f'gen_{generation:04d}' / 'stateful_recipe.json'
    evidence = dict(recipe=recipe, recipe_path=str(recipe_path),
                    implementations={p: file_hash(ROOT / p) for p in ADAPTERS})
    if evidence_path.exists() and json.loads(evidence_path.read_text()) != evidence:
        raise ValueError('stateful adapter or recipe changed since preflight/start')
    # Do not create a generation directory here: canonical allocation must see
    # the same next generation. The plan callback writes provenance after that.
    base = iterate._command_plan
    def plan(*plan_args):
        result = extend_plan(base, recipe_path, recipe, *plan_args)
        if not parsed.dry_run:
            atomic_json(evidence_path, evidence)
        return result
    iterate._command_plan = plan
    sys.argv = [__file__, *remaining]
    iterate.main()


if __name__ == '__main__':
    main()
