"""Timed opt-in alpha-beta versus incumbent PUCT diagnostic. One resident worker.

Each half-move gets the same budget (White gets two, as in simulation gates).
Actual end-to-end times and overruns are logged; PUCT stops between batches.
No finisher wrapper: both search cores use normal native legality/safety; report
this distinction from standard wrapped incumbent gates. Not a promotion gate.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ('src', 'native', 'tools'):
    sys.path.insert(0, str(ROOT / folder))
import monster_native as native
from monster_chess import MonsterChessGame
from repetition import RepetitionTracker
from evaluation import NNEvaluator
from native_mcts import NativeMCTS
from match_evidence import atomic_json, file_hash
from worker_lease import worker_lease
from match import load_book


def prior_keys(tracker, game):
    keys = []
    current = tuple(game.fen().split()[:4]) + (False,)
    for key, count in tracker.counts.items():
        n = count - int(not game.white_half_pending and key == current)
        keys.extend([' '.join(key[:4])] * n)
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--value', required=True)
    ap.add_argument('--reference', default='models/candidates/bootstrap_main_gen_0047/arena_selected.pt')
    ap.add_argument('--book', default='benchmarks/b2_challenger_confirmation_20260910/confirmation_book.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pairs', type=int, default=10)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--seconds', type=float, default=.05)
    ap.add_argument('--seed', type=int, default=9173)
    ap.add_argument('--free', action='store_true')
    selfplay = ap.add_mutually_exclusive_group()
    selfplay.add_argument('--selfplay', action='store_true', help='Alpha-beta plays both sides')
    selfplay.add_argument('--reference-selfplay', action='store_true', help='PUCT plays both sides on common starts')
    ap.add_argument('--extensions',type=int,default=0)
    ap.add_argument('--require-report', type=Path,
                    help='Require a successful profile with the currently installed native hash')
    args = ap.parse_args()
    if args.pairs < 1 or args.seconds <= 0:
        raise ValueError('positive pairs/time required')
    if args.require_report:
        prerequisite=json.loads(args.require_report.read_text())
        if not prerequisite.get('complete') or prerequisite.get('runtime')!=file_hash(ROOT/'native/monster_native.pyd'):
            raise ValueError('Required profile is incomplete or native runtime changed')
    entries, _ = load_book(args.book)
    entries = entries[args.offset:args.offset+args.pairs]
    if len(entries) != args.pairs:
        raise ValueError('insufficient openings')
    args.out.mkdir(parents=True, exist_ok=False)
    sources = ['native/monster_native.pyd', 'native/src/alphabeta.rs', 'native/src/cheap_value.rs',
               'native/src/mcts.rs', 'src/native_mcts.py', __file__, args.value, args.reference, args.book]
    sources += ['native/src/search_order.rs','native/src/search_cache.rs','native/src/search_bounds.rs',
                'native/src/game.rs','native/src/tactical.rs','native/src/float_dot.rs','native/src/relative_features.rs',
                'native/src/lib.rs','src/evaluation.py','src/monster_chess.py','src/repetition.py']
    if args.require_report:sources.append(args.require_report)
    atomic_json(args.out/'manifest.json', dict(
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        hashes={str(p):file_hash(p) for p in sources},
        budget_unit='per half-move', incumbent_finisher=False,
        cap='capture-only draw', repetition='threefold settled positions',
        caveat='PUCT checks time between batches; actual overrun recorded; reused diagnostic book'))
    rows = []
    with worker_lease():
        net = native.CheapValue(args.value)
        nn = NNEvaluator(args.reference)
        engine = NativeMCTS(10_000_000, nn, batch_size=16, root_noise=False,
                            allow_early_stop=False, reuse_across_moves=True, seed=args.seed)
        # Pay graph initialization outside measured games; record startup separately.
        startup = time.monotonic()
        engine.get_best_action(MonsterChessGame(), temperature=0, seconds=.05)
        atomic_json(args.out/'warmup.json', dict(seconds=time.monotonic()-startup))
        with (args.out/'games.jsonl').open('x', encoding='utf-8') as log:
            for pair, entry in enumerate(entries):
                for a_white in ([True] if args.selfplay or args.reference_selfplay else [True, False]):
                    random.seed(args.seed+pair)
                    np.random.seed(args.seed+pair)
                    game = MonsterChessGame() if args.free else MonsterChessGame(entry['fen'])
                    if not args.free:
                        game.white_half_pending = bool(entry['half'])
                        game.turn_count = int(entry['turn_count'])
                    initial = dict(fen=game.fen(), half=game.white_half_pending, turn_count=game.turn_count)
                    tracker = RepetitionTracker()
                    tracker.record(game)
                    engine._reuse_tree = engine._reuse_key = None
                    engine._decisions = 0
                    decisions = []
                    while not game.is_terminal() and tracker.fired_at is None:
                        start = time.monotonic()
                        use_ab = not args.reference_selfplay and (args.selfplay or game.is_white_turn == a_white)
                        if use_ab:
                            r = native.alphabeta_search(game.board.fen(en_passant='fen'),
                                pending=game.white_half_pending, turn_count=game.turn_count,
                                seconds=args.seconds, prior_positions=prior_keys(tracker, game), evaluator=net,
                                extension_turns=args.extensions)
                            legal = game.get_search_actions()
                            action = next((m for m in legal if m.uci() == r.action), None)
                            detail = dict(depth=r.completed_depth, nodes=r.nodes, value=r.value,
                                          interrupted=r.interrupted, core_seconds=r.elapsed_seconds,
                                          eval_cache_hits=r.eval_cache_hits, tt_hits=r.tt_hits,
                                          tt_cutoffs=r.tt_cutoffs, extension_nodes=r.extension_nodes,
                                          max_ply_reached=r.max_ply_reached)
                        else:
                            action, _, value = engine.get_best_action(game, temperature=0, seconds=args.seconds)
                            detail = dict(value=value, timing=engine.last_search_timing)
                        elapsed = time.monotonic()-start
                        if action is None:
                            raise RuntimeError(f'No legal action in live game: {game.fen()}')
                        decisions.append(dict(action=action.uci(), white=game.is_white_turn,
                            half=game.white_half_pending, alphabeta=use_ab, seconds=elapsed,
                            overrun_seconds=max(0,elapsed-args.seconds), **detail))
                        game.apply_search_action(action)
                        tracker.record(game, len(decisions))
                    result = game.get_result() if game.is_terminal() else 0
                    result = int(result) if abs(result) >= 1 else 0
                    row = dict(pair=pair, a_white=a_white, result_white=result,
                               result_a=result if a_white else -result,
                               start=initial, final_fen=game.fen(), decisions=decisions,
                               repetition=tracker.fired_at is not None)
                    rows.append(row)
                    log.write(json.dumps(row)+'\n'); log.flush()
                    summary = dict(games=len(rows), white_wins=sum(r['result_white']>0 for r in rows),
                        black_wins=sum(r['result_white']<0 for r in rows),
                        draws=sum(r['result_white']==0 for r in rows),
                        score=sum((r['result_a']+1)/2 for r in rows)/len(rows))
                    for color in (True,False):
                        subset=[r for r in rows if r['a_white']==color]
                        summary['as_white' if color else 'as_black'] = (
                            sum((r['result_a']+1)/2 for r in subset)/len(subset) if subset else None)
                    atomic_json(args.out/'progress.json', summary)
                    print(json.dumps(summary), flush=True)
    atomic_json(args.out/'complete.json', summary)


if __name__ == '__main__':
    main()
