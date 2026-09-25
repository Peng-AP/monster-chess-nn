"""Read-only replay audit for completed normal-start match journals."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from benchmark import _state_record
from free_gate_stats import leg_stats
from gate_sampled import match_settings
from match import build_tasks, game_score
from match_evidence import digest, file_hash, read_rows
from sampled_gate_stats import self_par


def audit_game(row):
    import chess
    from monster_chess import MonsterChessGame
    from repetition import RepetitionTracker

    if row.get('pair') is not None or row.get('entry') is not None:
        raise ValueError('Expected normal-start game, not book evidence')
    trace = row['game']['trajectory']
    if not trace or len(trace) != row['plies'] + 1 or digest(trace) != row['game']['trajectory_sha256']:
        raise ValueError('Trajectory digest/count mismatch')
    if row['plies'] > 600:
        raise ValueError('Game exceeds declared ply limit')
    game = MonsterChessGame()
    tracker = RepetitionTracker()
    tracker.record(game, 0)
    repeated = False
    opening_index = min(16, row['plies'])
    for i, saved in enumerate(trace):
        if i:
            if game.is_terminal() or repeated:
                raise ValueError('Continuation after termination')
            moves = [chess.Move.from_uci(m) for m in saved['action']]
            action = moves[0] if len(moves) == 1 else tuple(moves)
            if action not in game.get_search_actions():
                raise ValueError('Illegal logged action')
            game.apply_search_action(action)
            repeated = tracker.record(game, i)
        state = _state_record(game, i, 16)
        expected = dict(state, action=saved['action']) if i else state
        if saved != expected:
            raise ValueError('Replayed state/half/clock mismatch')
        if i == opening_index:
            opening = dict(state, history_sha256=digest(trace[:i + 1]),
                           repetition_sha256=digest(sorted(tracker.counts.items())))
            if row['opening'] != opening:
                raise ValueError('Opening history/repetition provenance mismatch')
    raw = 0 if repeated else game.get_result()
    reason = ('repetition' if repeated else 'king_capture' if abs(raw) == 1
              else 'ply_cap' if row['plies'] == 600 else 'turn_cap' if game.is_terminal()
              else 'unfinished')
    if reason == 'unfinished' or reason != row['game']['termination']:
        raise ValueError('Termination mismatch or unfinished game')
    if row['result_for_a'] != (raw if row['a_is_white'] else -raw):
        raise ValueError('Outcome mismatch')
    if row['white_score'] != game_score(raw):
        raise ValueError('Captures-only score mismatch')
    return 'white_capture' if raw == 1 else 'black_capture' if raw == -1 else reason


def audit_match(path, model_a, model_b, games, seed, sims, sims_b=None):
    import json
    from match_evidence import task_id
    path = Path(path)
    rows = read_rows(path)
    tasks = {task_id(t): t for t in build_tasks(games, seed, 16)}
    meta_path = path.with_suffix('.jsonl.manifest.json')
    meta = json.loads(meta_path.read_text())
    settings = match_settings(str(model_a), str(model_b), dict(games=games, seed=seed),
                              SimpleNamespace(sims=sims, workers=8))
    if sims_b is not None:
        settings['sims_b'] = sims_b
    if meta != dict(schema_version=1, settings=settings, tasks=list(tasks)):
        raise ValueError('Normal-start journal settings/tasks mismatch')
    if len(rows) != len(tasks) or {r['task_id'] for r in rows} != set(tasks):
        raise ValueError('Missing or duplicate match tasks')
    endings = Counter()
    for row in rows:
        task = tasks[row['task_id']]
        if (row['a_is_white'], row['seed'], row['pair']) != (task[0], task[1], task[4]):
            raise ValueError('Task metadata mismatch')
        endings[audit_game(row)] += 1
    counts = Counter((r['opening']['fen'], r['opening']['half'], r['opening']['turn_count']) for r in rows)
    result = dict(replay_audited=True, diagnostics=leg_stats(rows), endings=dict(endings),
                  opening_concentration=dict(unique_actual_endpoints=len(counts),
                      largest_endpoint_count=max(counts.values()),
                      top_five_counts=sorted(counts.values(), reverse=True)[:5]),
                  evidence={str(p): file_hash(p) for p in (path, meta_path)})
    if Path(model_a).resolve() == Path(model_b).resolve() and (sims_b is None or sims_b == sims):
        result['actual_color_self_par'] = self_par(rows)
    return result
