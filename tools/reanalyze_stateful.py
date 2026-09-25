"""Opt-in full-state adapter around the existing journaled reanalysis runner."""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalyze
from stateful_generation import restore_state


def reanalyze_one(item):
    record = item['record']
    if not record.get('state'):
        raise ValueError('stateful reanalysis refuses FEN-only source records')
    game, repetition = restore_state(record['state'])
    if game.is_terminal() or repetition.fired_at is not None:
        raise ValueError('terminal state supplied for reanalysis')
    if game.fen() != record['fen'] or game.is_white_turn != (record['current_player'] == 'white') or int(game.white_half_pending) != record['half']:
        raise ValueError('record and state disagree')
    # Reuse the evaluator, not a previous task's root tree.
    reanalyze._worker_engine._reuse_tree = None
    reanalyze._worker_engine._reuse_key = None
    _, policy, value = reanalyze._worker_engine.get_best_action(game, temperature=0.0)
    if not policy:
        raise ValueError('deep search returned empty policy')
    return dict(identity=reanalyze.record_identity(item), source_path=item['path'],
                source_line=item['line'], fen=record['fen'], current_player=record['current_player'],
                half=record['half'], game_result=record['game_result'],
                plies_to_end=record.get('plies_to_end'), deep_policy=policy, deep_value=float(value),
                metrics=reanalyze.disagreement_score(record['policy'], policy, record.get('mcts_value', 0), value))


def main():
    reanalyze._reanalyze_one = reanalyze_one
    # Journal records this adapter's implementation, not only the legacy runner.
    reanalyze.__file__ = __file__
    reanalyze.main()


if __name__ == '__main__':
    mp.freeze_support()
    main()
