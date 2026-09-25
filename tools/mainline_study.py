"""Full-history conditional games and root-turn probes; never training data.

Completed tasks publish atomically and are replay-audited before saving/reuse.
Game settings match the ordinary native benchmark. Pure-MCTS root probes are
explicitly separate: no early stopping or finisher, no inferred game outcome.
"""
import argparse
from collections import Counter, defaultdict
import concurrent.futures as futures
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from benchmark import _build_engine
from match import game_score
from match_evidence import atomic_json, digest, file_hash, model_identity, runtime_identity
from monster_chess import MonsterChessGame
from repetition import RepetitionTracker
from worker_lease import worker_lease

_engines = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pin(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f'Changed pinned evidence: {path}')
    else:
        atomic_json(path, value)


def snapshot(game, initial_fen, moves):
    return dict(initial_fen=initial_fen, moves=list(moves), fen=game.fen(),
                half=bool(game.white_half_pending), turn_count=game.turn_count)


def checked_restore(state):
    import chess
    game = MonsterChessGame()
    if state['initial_fen'] != game.fen():
        raise ValueError('Missing normal-start history')
    repetition = RepetitionTracker()
    repetition.record(game, 0)
    for i, uci in enumerate(state['moves']):
        if game.is_terminal() or repetition.fired_at is not None:
            raise ValueError('Prefix continues after termination')
        move = chess.Move.from_uci(uci)
        if move not in game.get_search_actions():
            raise ValueError('Illegal prefix action')
        game.apply_search_action(move)
        repetition.record(game, i + 1)
    if snapshot(game, state['initial_fen'], state['moves']) != state:
        raise ValueError('Reconstructed state/half/clock mismatch')
    if game.is_terminal() or repetition.fired_at is not None:
        raise ValueError('Cannot start from a terminal prefix')
    return game, repetition


def record_state(game, ply):
    return dict(fen=game.fen(), half=bool(game.white_half_pending),
                turn_count=game.turn_count, absolute_ply=ply)


def initialize(keys, probe):
    import torch
    torch.set_num_threads(1)
    global _engines
    _engines = {}
    if len(keys) != 2:
        raise ValueError('Explicit White and Black engine specifications required')
    for role, (model, sims) in zip((True, False), keys):
        if probe and not role:
            break
        engine, _ = _build_engine(model, sims, engine='native')
        if probe:
            engine = getattr(engine, '_inner', engine)
            engine.allow_early_stop = False
        # Equal weights do NOT mean equal search-tree ownership. The ordinary
        # match harness gives each side its own tree, including self-play.
        _engines['probe' if probe else role] = engine


def run_task(task):
    import numpy as np
    import torch
    random.seed(task['seed'])
    np.random.seed(task['seed'])
    torch.manual_seed(task['seed'])
    game, repetition = checked_restore(task['state'])
    start_ply = len(task['state']['moves'])
    root_white = game.is_white_turn
    engines = ({root_white: _engines['probe']} if task['kind'] == 'probe'
               else {True: _engines[True], False: _engines[False]})
    for engine in engines.values():
        inner = getattr(engine, '_inner', engine)
        inner._reuse_tree = inner._reuse_key = None
        inner._decisions = 0
    trace = [record_state(game, start_ply)]
    started = time.monotonic()
    repeated = False
    while not game.is_terminal() and start_ply + len(trace) - 1 < 600:
        if time.monotonic() - started > task['deadline_seconds']:
            raise TimeoutError('Conditional task exceeded deadline')
        engine = engines[game.is_white_turn]
        absolute_ply = start_ply + len(trace) - 1
        temperature = .5 if task['kind'] == 'game' and absolute_ply < 16 else 0.
        hits = getattr(engine, 'finisher_hits', 0)
        tick = time.monotonic()
        action, policy, value = engine.get_best_action(game, temperature=temperature)
        decision_seconds = time.monotonic() - tick
        if action is None or action not in game.get_search_actions():
            raise ValueError('Nonterminal search returned no/illegal action')
        if value is not None and not math.isfinite(float(value)):
            raise ValueError('Nonfinite search value')
        top = sorted(((str(k), float(v)) for k, v in (policy or {}).items()),
                     key=lambda item: (-item[1], item[0]))[:8]
        if any(not math.isfinite(v) or v < 0 for _, v in top):
            raise ValueError('Invalid policy')
        game.apply_search_action(action)
        repeated = repetition.record(game, absolute_ply + 1)
        trace.append(dict(record_state(game, absolute_ply + 1), action=action.uci(),
                          root_value=None if value is None else float(value), policy_top=top,
                          decision_seconds=decision_seconds,
                          finisher=getattr(engine, 'finisher_hits', 0) > hits))
        if repeated or (task['kind'] == 'probe' and game.is_white_turn != root_white):
            break
    raw = 0. if repeated else game.get_result()
    ending = ('repetition' if repeated else 'king_capture' if abs(raw) == 1
              else 'turn_cap' if game.is_terminal() else 'ply_cap' if trace[-1]['absolute_ply'] >= 600
              else 'diagnostic_limit')
    if task['kind'] == 'game' and ending == 'diagnostic_limit':
        raise ValueError('Unfinished conditional game')
    if ending == 'diagnostic_limit':
        raw = None
    return dict(task_id=digest(task), case_id=task['case_id'], kind=task['kind'],
                root_state_sha256=digest(task['state']), trajectory=trace,
                trajectory_sha256=digest(trace), ending=ending, result_white=raw,
                white_score=None if raw is None else game_score(raw),
                continuation_plies=len(trace) - 1, elapsed_sec=time.monotonic() - started,
                final_repetition_sha256=digest(sorted(repetition.counts.items())))


def audit_result(task, result):
    import chess
    if (result['task_id'] != digest(task) or result['root_state_sha256'] != digest(task['state'])
            or result['case_id'] != task['case_id'] or result['kind'] != task['kind']):
        raise ValueError('Result task identity mismatch')
    game, repetition = checked_restore(task['state'])
    root_white = game.is_white_turn
    start = len(task['state']['moves'])
    trace = result['trajectory']
    if (not trace or len(trace) != result['continuation_plies'] + 1
            or digest(trace) != result['trajectory_sha256'] or start + len(trace) - 1 > 600):
        raise ValueError('Trajectory digest/count mismatch')
    repeated = False
    for i, saved in enumerate(trace):
        if i:
            if game.is_terminal() or repeated or (task['kind'] == 'probe' and game.is_white_turn != root_white):
                raise ValueError('Continuation after task termination')
            move = chess.Move.from_uci(saved['action'])
            if move not in game.get_search_actions():
                raise ValueError('Illegal recorded action')
            game.apply_search_action(move)
            repeated = repetition.record(game, start + i)
        expected = record_state(game, start + i)
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('Replayed state/half/clock mismatch')
    raw = 0. if repeated else game.get_result()
    ending = ('repetition' if repeated else 'king_capture' if abs(raw) == 1
              else 'turn_cap' if game.is_terminal() else 'ply_cap' if start + len(trace) - 1 == 600
              else 'diagnostic_limit')
    if ending == 'diagnostic_limit':
        if task['kind'] != 'probe' or game.is_white_turn == root_white:
            raise ValueError('Incomplete game or root turn')
        raw = None
    if (result['ending'] != ending or result['result_white'] != raw
            or result['white_score'] != (None if raw is None else game_score(raw))
            or result['final_repetition_sha256'] != digest(sorted(repetition.counts.items()))):
        raise ValueError('Ending/outcome/repetition mismatch')


def load_result(path, task):
    saved = read(path)
    result = saved['result']
    if saved['result_sha256'] != digest(result):
        raise ValueError('Changed completed task result')
    audit_result(task, result)
    return result


def summarize(tasks, results):
    groups = defaultdict(list)
    probes = []
    for task, result in zip(tasks, results):
        if task['kind'] == 'probe':
            probes.append(dict(case=task['case_id'], model=task['white_model'],
                sims=task['white_sims'], decisions=[dict(action=r['action'], value=r['root_value'],
                policy_top=r['policy_top'], seconds=r['decision_seconds']) for r in result['trajectory'][1:]],
                ending=result['ending'], result_white=result['result_white'], task_id=result['task_id']))
        else:
            groups[(task['case_id'], task['white_model'], task['black_model'],
                    task['white_sims'], task['black_sims'])].append(result)
    rows = []
    for key, values in groups.items():
        scores = [v['white_score'] for v in values]
        rows.append(dict(case=key[0], white_model=key[1], black_model=key[2],
            white_sims=key[3], black_sims=key[4], games=len(values), white_wins=scores.count(1.),
            black_wins=scores.count(0.), draws=scores.count(.5), white_score=sum(scores)/len(scores),
            endings=dict(Counter(v['ending'] for v in values)),
            mean_continuation_plies=sum(v['continuation_plies'] for v in values)/len(values)))
    return dict(groups=rows, probes=probes)


def run(config_path, output):
    config = read(config_path)
    tasks = config['tasks']
    expected = {digest(task): task for task in tasks}
    if not tasks or len(expected) != len(tasks):
        raise ValueError('Empty or duplicate task schedule')
    if any(t['kind'] not in ('game', 'probe') or min(t['white_sims'], t['black_sims']) <= 0
           or not 0 <= t['seed'] < 2**32 for t in tasks):
        raise ValueError('Invalid task setting')
    output = Path(output)
    files = output / 'tasks'
    files.mkdir(parents=True, exist_ok=True)
    manifest = dict(config_sha256=file_hash(config_path), task_ids=list(expected),
        runtime=runtime_identity(), implementation_sha256=file_hash(__file__),
        models=[model_identity(p) for p in sorted({t[k] for t in tasks for k in ('white_model', 'black_model')})],
        workers=8, probe_workers=4, engine_ownership='separate per-color trees for games; root-actor-only probes',
        history='complete driver prefix; fresh search tree; existing native recent-history semantics',
        game_search='benchmark defaults, early stop/reuse/finisher retained',
        probe_search='pure neural MCTS, early stopping off, root actor turn only')
    pin(output / 'manifest.json', manifest)
    if {p.stem for p in files.glob('*.json')} - set(expected):
        raise ValueError('Unexpected task files in this suite')
    groups = defaultdict(list)
    done = 0
    for task in tasks:
        path = files / (digest(task) + '.json')
        if path.exists():
            load_result(path, task)
            done += 1
        else:
            groups[(task['white_model'], task['black_model'], task['white_sims'],
                    task['black_sims'], task['kind'])].append(task)
    started = time.monotonic()
    with worker_lease():
        for key, todo in groups.items():
            print(f'STUDY {key[4]} {Path(key[0]).parent.name} W/{Path(key[1]).parent.name} B '
                  f'{key[2]}/{key[3]} sims: {len(todo)} tasks', flush=True)
            keys = [(key[0], key[2]), (key[1], key[3])]
            pool = futures.ProcessPoolExecutor(max_workers=min(4 if key[4] == 'probe' else 8, len(todo)),
                mp_context=mp.get_context('spawn'), initializer=initialize, initargs=(keys, key[4] == 'probe'))
            pending = {pool.submit(run_task, task): task for task in todo}
            try:
                while pending:
                    ready, _ = futures.wait(pending, timeout=1800, return_when=futures.FIRST_COMPLETED)
                    if not ready:
                        raise TimeoutError('No study task completed for 1800 seconds')
                    for future in ready:
                        task = pending.pop(future)
                        result = future.result()
                        audit_result(task, result)
                        atomic_json(files / (digest(task) + '.json'),
                                    dict(result_sha256=digest(result), result=result))
                        done += 1
                        if done % 8 == 0 or done == len(tasks):
                            print(f'STUDY completed {done}/{len(tasks)}; {time.monotonic()-started:.1f}s this invocation', flush=True)
            except BaseException:
                from data_generation import terminate_pool
                terminate_pool(pool)
                raise
            else:
                pool.shutdown(wait=True)
    if runtime_identity() != manifest['runtime'] or any(model_identity(p['path']) != p for p in manifest['models']):
        raise ValueError('Runtime/model changed during study')
    results = [load_result(files / (digest(t) + '.json'), t) for t in tasks]
    summary = dict(complete=True, tasks=len(tasks), games=sum(t['kind']=='game' for t in tasks),
                   root_probes=sum(t['kind']=='probe' for t in tasks), replay_audited=True,
                   manifest_sha256=file_hash(output/'manifest.json'),
                   evidence={str(p):file_hash(p) for p in sorted(files.glob('*.json'))},
                   **summarize(tasks, results))
    pin(output / 'summary.json', summary)
    print(f'STUDY COMPLETE {output}/summary.json', flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.config, args.output)
