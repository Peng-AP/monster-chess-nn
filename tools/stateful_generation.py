"""State-preserving self-play, league, fresh-prefix and completed-fork data.

Separate opt-in backend: does not change an in-flight legacy campaign. Every
position carries its complete action prefix; forks retain source-family ancestry.
Only completed games are published. Existing outputs are hash-checked on resume.
"""
import argparse
import concurrent.futures as futures
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from match_evidence import atomic_json, file_hash, runtime_identity, model_identity, digest
from worker_lease import worker_lease

_evaluators = {}


def restore_state(state):
    import chess
    from monster_chess import MonsterChessGame
    from repetition import RepetitionTracker
    game = MonsterChessGame(fen=state['initial_fen'])
    tracker = RepetitionTracker()
    tracker.record(game, 0)
    for i, uci in enumerate(state['moves']):
        if game.is_terminal() or tracker.fired_at is not None:
            raise ValueError('prefix continues after terminal state')
        game.apply_search_action(chess.Move.from_uci(uci))
        tracker.record(game, i + 1)
    if game.fen() != state['fen'] or game.turn_count != state['turn_count'] or bool(game.white_half_pending) != bool(state['half']):
        raise ValueError('state reconstruction mismatch')
    return game, tracker


def snapshot(game, initial_fen, moves):
    return dict(initial_fen=initial_fen, moves=list(moves), fen=game.fen(),
                turn_count=game.turn_count, half=int(game.white_half_pending))


def init_worker(models):
    import torch
    from evaluation import NNEvaluator
    torch.set_num_threads(1)
    global _evaluators
    _evaluators = {p: NNEvaluator(p) for p in dict.fromkeys(models)}


def engine(model, sims):
    from native_mcts import NativeMCTS
    return NativeMCTS(num_simulations=sims, eval_fn=_evaluators[model],
                      root_noise=True, allow_early_stop=False)


def play_task(task):
    import numpy as np
    import torch
    from monster_chess import MonsterChessGame
    from repetition import RepetitionTracker
    from config import TEMPERATURE_MOVES, TEMPERATURE_HIGH, TEMPERATURE_LOW
    from data_generation import _finisher_settings, _finisher_applicable, _finisher_move
    random.seed(task['seed'])
    np.random.seed(task['seed'])
    torch.manual_seed(task['seed'])
    deadline = time.monotonic() + task.get('deadline_seconds', 1800)
    initial = task.get('state')
    if initial:
        game, repetition = restore_state(initial)
        initial_fen, moves = initial['initial_fen'], list(initial['moves'])
    else:
        game = MonsterChessGame()
        repetition = RepetitionTracker()
        repetition.record(game, 0)
        initial_fen, moves = game.fen(), []
    if game.is_terminal() or repetition.fired_at is not None:
        raise ValueError('cannot continue terminal position')
    engines = {p: engine(p, task['sims']) for p in dict.fromkeys([task['model'], task.get('other', task['model'])])}
    prefix_engine = engines.get(task.get('other'))
    records = []
    finisher_on, depth, nodes, material_max = _finisher_settings()
    reason = None
    while not game.is_terminal():
        if time.monotonic() > deadline:
            raise TimeoutError(f"game deadline exceeded: {task['id']}")
        prefix = task['kind'] == 'fresh' and len(moves) < task.get('prefix_plies', 8)
        model = task['model']
        if task['kind'] == 'league' and game.is_white_turn != (task['train_side'] == 'white'):
            model = task['other']
        chosen_engine = prefix_engine if prefix else engines[model]
        action = None
        proven = False
        if finisher_on and not game.is_white_turn and _finisher_applicable(game, material_max):
            action = _finisher_move(game, depth, nodes)
            proven = action is not None
        if action is None:
            # Recipes may lengthen the exploratory opening (gen51: 30 plies);
            # tasks without the key keep the historical config value.
            plies = task.get('temperature_plies', TEMPERATURE_MOVES)
            temperature = TEMPERATURE_HIGH if len(moves) < plies else TEMPERATURE_LOW
            action, policy, value = chosen_engine.get_best_action(game, temperature=temperature)
        else:
            from evaluation import evaluate
            policy, value = {action.uci(): 1.0}, -evaluate(game)
        if action is None:
            raise ValueError('nonterminal search returned no action; refusing false outcome')
        if not prefix:
            record = dict(fen=game.fen(), half=int(game.white_half_pending),
                          current_player='white' if game.is_white_turn else 'black',
                          policy=policy, mcts_value=float(value),
                          policy_weight=1.0 if model == task['model'] or proven else 0.0,
                          state=snapshot(game, initial_fen, moves),
                          played_action=action.uci(), source='stateful_' + task['kind'],
                          generator_model=model, task_id=task['id'])
            if task.get('source_record'):
                record['source_record'] = task['source_record']
            records.append(record)
        game.apply_search_action(action)
        moves.append(action.uci())
        if repetition.record(game, len(moves)):
            reason = 'repetition'
            break
    if not records:
        raise ValueError('prefix ended game before continuation; no sample published')
    result = repetition.draw_result if reason == 'repetition' else game.get_result()
    reason = reason or ('king_capture' if abs(result) == 1 else 'turn_cap')
    for i, record in enumerate(records):
        record.update(game_result=result, plies_to_end=len(records)-1-i, termination=reason)
    return records


def with_exploration(task, config):
    """Copy an explicit recipe exploration length into the task.

    Only when the recipe declares it, so every pre-existing task digest (and
    therefore every completed generation receipt) is unchanged.
    """
    if 'temperature_plies' in config:
        plies = config['temperature_plies']
        if not isinstance(plies, int) or not 0 <= plies <= 200:
            raise ValueError('temperature_plies must be an integer in 0..200')
        task['temperature_plies'] = plies
    return task


def split_count(count, groups):
    return [count // groups + int(i < count % groups) for i in range(groups)]


def ordinary_batches(config):
    batches = []
    serial = 0
    def add(kind, count, other=None, side=None):
        nonlocal serial
        tasks = []
        for _ in range(count):
            task = dict(id=f'{kind}/game_{serial:05d}', kind=kind, model=config['model'],
                        sims=config['sims'], seed=config['seed'] + serial)
            if other:
                task['other'] = other
            if side:
                task['train_side'] = side
            with_exploration(task, config)
            tasks.append(task)
            serial += 1
        if tasks:
            batches.append(tasks)
    add('selfplay', config['free_games'])
    for other, count in zip(config['prefix_models'], split_count(config['fresh_games'], len(config['prefix_models']))):
        add('fresh', count, other)
    for side, side_count in zip(('white', 'black'), split_count(config['league_games'], 2)):
        for other, count in zip(config['opponents'], split_count(side_count, len(config['opponents']))):
            add('league', count, other, side)
    return batches


def fork_tasks(config, raw):
    rng = random.Random(config['seed'] + 500000)
    paths = sorted(raw.glob('*/*.jsonl'))
    paths = [p for p in paths if p.parent.name in ('selfplay', 'fresh', 'league')]
    rng.shuffle(paths)
    requested = config['fork_games']
    quotas = {'black': int(requested * .6 + .5)}
    quotas['white'] = requested - quotas['black']
    tasks = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        eligible = [(i, r) for i, r in enumerate(rows) if not r['half'] and len(r['state']['moves']) >= 8 and quotas[r['current_player']] > 0]
        if not eligible:
            continue
        if config.get('fork_sampling') == 'mixed_outcome_surprise' and rng.random() < .5:
            # Generic value/outcome disagreement, not a hand-written tactical
            # rule. Half of roots remain uniform to preserve broad coverage.
            def surprise(item):
                record = item[1]
                target = record['game_result'] * (1 if record['current_player'] == 'white' else -1)
                return abs(float(record['mcts_value']) - target)
            i, row = max(eligible, key=surprise)
        else:
            i, row = rng.choice(eligible)
        quotas[row['current_player']] -= 1
        index = len(tasks)
        tasks.append(with_exploration(dict(id=f'fork/game_{index:05d}', kind='fork', model=config['model'],
                          sims=config['fork_sims'], seed=config['seed'] + 600000 + index,
                          state=row['state'], source_record=dict(path=path.relative_to(raw).as_posix(), line=i+1)), config))
        if len(tasks) == requested:
            break
    if len(tasks) != requested:
        raise ValueError(f'insufficient distinct parent games for fork quotas: {quotas}')
    return tasks


def run_batch(tasks, raw, evidence, workers):
    from data_generation import terminate_pool
    pending_tasks = []
    for task in tasks:
        output = raw / (task['id'] + '.jsonl')
        receipt = evidence / (task['id'].replace('/', '_') + '.json')
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['task'] != digest(task) or not output.exists() or file_hash(output) != saved['sha256']:
                raise ValueError(f'changed generation output: {output}')
        elif output.exists():
            raise ValueError(f'unreceipted game exists: {output}; inspect before resuming')
        else:
            pending_tasks.append(task)
    if not pending_tasks:
        return
    models = list(dict.fromkeys([tasks[0]['model'], tasks[0].get('other', tasks[0]['model'])]))
    pool = futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                       initializer=init_worker, initargs=(models,))
    pending = {pool.submit(play_task, task): task for task in pending_tasks}
    done_count = len(tasks) - len(pending_tasks)
    try:
        while pending:
            done, _ = futures.wait(pending, timeout=600, return_when=futures.FIRST_COMPLETED)
            if not done:
                raise TimeoutError('generation stalled for 600 seconds')
            for future in done:
                task = pending.pop(future)
                rows = future.result()
                output = raw / (task['id'] + '.jsonl')
                output.parent.mkdir(parents=True, exist_ok=True)
                temp = output.with_suffix('.tmp')
                with temp.open('w', encoding='utf-8') as stream:
                    for row in rows:
                        stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, output)
                atomic_json(evidence / (task['id'].replace('/', '_') + '.json'),
                            dict(task=digest(task), sha256=file_hash(output), rows=len(rows)))
                done_count += 1
                if done_count % 25 == 0 or done_count == len(tasks):
                    print(f"{task['kind']} {done_count}/{len(tasks)} completed", flush=True)
    except BaseException:
        terminate_pool(pool)
        raise
    else:
        pool.shutdown(wait=True)


def run(config, raw, summary):
    raw, summary = Path(raw), Path(summary)
    evidence = summary.parent / 'generation_receipts'
    manifest_path = summary.parent / 'stateful_generation_manifest.json'
    manifest = dict(config=config, runtime=runtime_identity(), implementation=file_hash(__file__),
                    models=[model_identity(p) for p in dict.fromkeys([config['model'], *config['prefix_models'], *config['opponents']])])
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('generation resume provenance changed')
    atomic_json(manifest_path, manifest)
    with worker_lease():
        for batch in ordinary_batches(config):
            run_batch(batch, raw, evidence, config['workers'])
        forks = fork_tasks(config, raw) if config['fork_games'] else []
        if forks:
            run_batch(forks, raw, evidence, config['workers'])
    total = sum(config[k] for k in ('free_games', 'fresh_games', 'league_games', 'fork_games'))
    atomic_json(summary, dict(num_games_requested=total, saved_games=total, failed_games=0,
                             recipe=config, manifest_sha256=file_hash(manifest_path)))


if __name__ == '__main__':
    mp.freeze_support()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--summary', required=True)
    args = ap.parse_args()
    run(json.loads(Path(args.config).read_text()), args.raw, args.summary)
