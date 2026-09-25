"""Exercise trained B2 checkpoints through CUDA graphs and native search."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from match_evidence import atomic_json, file_hash
from worker_lease import worker_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    training = json.loads((root/'training_complete.json').read_text())
    if not training.get('complete'):
        raise ValueError('Training incomplete')
    import torch
    import numpy as np
    from evaluation import NNEvaluator
    from native_mcts import NativeMCTS, make_bridge
    from monster_chess import MonsterChessGame
    from encoding import fen_to_tensor
    results = {}
    torch.set_num_threads(1)
    with worker_lease():
        for arm in ('control', 'state_cnn', 'hybrid'):
            path = Path(training.get('paths', {}).get(arm, root/'models'/arm/'best_value_net.pt'))
            if file_hash(path) != training['models'][arm]:
                raise ValueError('Trained checkpoint changed')
            evaluator = NNEvaluator(str(path))
            bridge, channels = make_bridge(evaluator, graph_width=16)
            game = MonsterChessGame()
            tensor = fen_to_tensor(game.fen(), input_channels=channels, turn_count=0).transpose(2, 0, 1)
            timings = {}
            for n in (1, 4, 8, 16):
                data = np.repeat(tensor[None], n, axis=0).tobytes()
                for _ in range(3):
                    value, policy = bridge(data, n, channels)
                started = time.perf_counter()
                for _ in range(30):
                    value, policy = bridge(data, n, channels)
                timings[n] = (time.perf_counter()-started)/30
                if not np.isfinite(np.frombuffer(value, np.float32)).all() or not np.isfinite(np.frombuffer(policy, np.float32)).all():
                    raise ValueError('Nonfinite inference output')
            engine = NativeMCTS(num_simulations=32, eval_fn=evaluator, seed=3173)
            moves = []
            for _ in range(6):
                if game.is_terminal():
                    break
                move, policy, value = engine.get_best_action(game, temperature=0)
                if move is None or not policy or not np.isfinite(value):
                    raise ValueError('Native B2 search failed')
                moves.append(move.uci())
                game.apply_search_action(move)
            results[arm] = dict(model_sha256=file_hash(path), channels=channels,
                parameters=sum(p.numel() for p in evaluator.model.parameters()),
                callback_seconds=timings, native_moves=moves)
            del engine, bridge, evaluator
            torch.cuda.empty_cache()
    atomic_json(root/'model_smoke.json', dict(complete=True, results=results))


if __name__ == '__main__':
    main()
