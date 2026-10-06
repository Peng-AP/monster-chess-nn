"""Fixed-state native evaluation parity and speed, before/after CPU optimization.

No GPU. Includes Python/FEN overhead in inference timing. Full-search throughput
is measured separately. Run only when timed matches are idle.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for folder in ('src', 'native', 'tools'):
    sys.path.insert(0, str(ROOT / folder))
import chess
import numpy as np
import torch
import monster_native as native
from monster_chess import MonsterChessGame
from train_search_value import model
from match_evidence import atomic_json, file_hash
from worker_lease import worker_lease


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--compare', type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    source = ROOT / 'benchmarks/search_first_20260911/exact_frontier_300ms/games.jsonl'
    states = []
    for row in map(json.loads, source.read_text().splitlines()):
        g = MonsterChessGame(row['start']['fen'])
        g.white_half_pending = row['start']['half']
        g.turn_count = row['start']['turn_count']
        for ply, decision in enumerate(row['decisions']):
            if ply % 5 == 0:
                states.append((g.board.fen(en_passant='fen'), g.white_half_pending, g.turn_count))
            g.apply_search_action(chess.Move.from_uci(decision['action']))
    states = states[:256]
    report = dict(runtime=file_hash(ROOT / 'native/monster_native.pyd'),
                  source=file_hash(source), states=states, models={}, complete=False)
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    if baseline and baseline['states'] != [list(s) for s in states]:
        raise ValueError('Baseline states differ')
    with worker_lease():
        for width, folder, epoch in [(128, 'search_first_distilled_001', 22),
                                     (512, 'search_first_distilled_w512_001', 7)]:
            path = ROOT / 'models/candidates' / folder / f'epoch_{epoch:03}.bin'
            evaluator = native.CheapValue(str(path))
            net = model(width, 32).eval()
            net.load_state_dict(torch.load(path.with_suffix('.pt'), weights_only=True, map_location='cpu'))
            inputs = np.zeros((len(states), 840), dtype=np.float32)
            for row, state in enumerate(states):
                for feature, value in native.CheapValue.features(*state):
                    inputs[row, feature] = value
            with torch.no_grad():
                expected = net(torch.from_numpy(inputs)).numpy().ravel()
            actual = np.array([evaluator.evaluate(*state) for state in states])
            error = float(np.max(np.abs(expected - actual)))
            if error > 1e-5:
                raise AssertionError(f'Native/Torch mismatch width{width}: {error}')
            durations = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(10):
                    for state in states:
                        evaluator.evaluate(*state)
                durations.append(time.perf_counter() - start)
            searches = []
            for state in states[::max(1, len(states)//12)][:12]:
                r = native.alphabeta_search(state[0], pending=state[1], turn_count=state[2],
                                            seconds=.3, evaluator=evaluator)
                searches.append(dict(state=state, nodes=r.nodes, seconds=r.elapsed_seconds,
                                     depth=r.completed_depth, action=r.action, value=r.value))
            record = dict(hash=file_hash(path), torch_max_error=error, values=actual.tolist(),
                          median_eval_seconds=float(np.median(durations))/(10*len(states)),
                          search=searches)
            if baseline:
                before = baseline['models'][str(width)]
                if before['hash'] != record['hash']:
                    raise ValueError('Baseline model differs')
                delta = float(np.max(np.abs(actual - np.array(before['values']))))
                if delta > 1e-5:
                    raise AssertionError(f'Optimization changed predictions: {delta}')
                record['baseline_max_error'] = delta
                record['eval_speedup'] = before['median_eval_seconds']/record['median_eval_seconds']
            report['models'][str(width)] = record
            print(json.dumps({k:v for k,v in record.items() if k not in ('values','search')}), flush=True)
    report['complete'] = True
    atomic_json(args.out, report)


if __name__ == '__main__':
    main()
