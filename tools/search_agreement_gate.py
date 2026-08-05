"""E3 gate (a): selected-move agreement between the engines on the probe deck.

With root noise off and temperature 0, PUCT is deterministic given identical
evaluations — so unlike the UCB1 path (which shuffles untried actions), this
comparison admits no legitimate disagreement. The gate is **>=99% with every
disagreement triaged**, tightened from the directive's original 95% because a
5% budget over 400 positions would let 20 positions differ silently, and §4
names FPU/backprop perspective bugs as the top risk.

Both engines run their full decision, overrides included, because that is what
the caller sees.

    py -3 tools/search_agreement_gate.py --positions 400 --sims 400
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "native"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import monster_native as mn  # noqa: E402
from evaluation import NNEvaluator  # noqa: E402
from mcts import MCTS  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from nn_bridge import make_eval_fn  # noqa: E402

DECK = os.path.join(ROOT, "data", "start_fens", "promotion_defense_deck_v1.jsonl")


def load_deck(limit):
    fens = []
    with open(DECK, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fen = rec.get("fen") if isinstance(rec, dict) else None
            if fen:
                fens.append(fen)
            if len(fens) >= limit:
                break
    return fens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="models/fresh_start_v20/best_value_net.pt")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    fens = load_deck(args.positions)
    eval_fn, channels = make_eval_fn(args.model)
    evaluator = NNEvaluator(args.model)

    # Warm the GPU: the first forward carries CUDA init and cuDNN autotune, and
    # timing or comparing across it is meaningless.
    warm = mn.Tree(fens[0])
    warm.run_batched_puct(32, eval_fn, batch_size=args.batch, channels=channels,
                          allow_early_stop=False)

    agree = 0
    disagreements = []
    started = time.time()
    for i, fen in enumerate(fens):
        tree = mn.Tree(fen)
        tree.run_batched_puct(args.sims, eval_fn, batch_size=args.batch,
                              channels=channels, allow_early_stop=False,
                              root_noise=False)
        native_action, native_probs, native_value = tree.best_action(temperature=0.0)

        search = MCTS(num_simulations=args.sims, eval_fn=evaluator,
                      root_noise=False, allow_early_stop=False,
                      batch_size=args.batch)
        py_action, py_probs, py_value = search.get_best_action(
            MonsterChessGame(fen), temperature=0.0)
        py_key = py_action.uci() if py_action is not None else None

        if native_action == py_key:
            agree += 1
        else:
            top = sorted(py_probs.items(), key=lambda kv: -kv[1])
            rank = next((r + 1 for r, (k, _) in enumerate(top) if k == native_action), None)
            disagreements.append({
                "fen": fen,
                "native": native_action,
                "python": py_key,
                "native_rank_in_python": rank,
                "python_top_prob": top[0][1] if top else None,
                "native_prob": dict(top).get(native_action),
                "native_value": native_value,
                "python_value": py_value,
            })
    elapsed = time.time() - started

    total = len(fens)
    summary = {
        "positions": total,
        "sims": args.sims,
        "batch": args.batch,
        "agreement": agree,
        "agreement_rate": round(agree / total, 4) if total else None,
        "gate": 0.99,
        "passed": (agree / total) >= 0.99 if total else False,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "disagreements": disagreements[:40],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"search_agreement_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "disagreements"},
                     indent=2))
    for d in disagreements[:8]:
        print(" ", json.dumps(d)[:260])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
