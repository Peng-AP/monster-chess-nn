"""What the native core has bought so far, measured rather than extrapolated.

Important caveat on every number below: the MCTS is still Python (E3), so each
native call here crosses the PyO3 boundary **per operation**. In the finished
design the whole search runs native and crossings happen once per leaf batch
(~25-100 per decision, D3). So these figures are a *lower bound* on the eventual
gain for everything except the boundary itself.

Reported per component because they enter a decision in different proportions
(CONTEXT law 18: movegen ~35%, clone/apply 25-32%, tree ~25%, NN 14%).
"""
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "native"))

import chess  # noqa: E402
import monster_native as mn  # noqa: E402
from evaluation import evaluate  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def build_states(n, seed=5):
    """Matched Python/native states reached by the same action sequence."""
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        py = MonsterChessGame(START_FEN)
        rs = mn.Game(START_FEN)
        for _ in range(120):
            if py.is_terminal() or len(pairs) >= n:
                break
            pairs.append((py, rs))
            actions = [m.uci() for m in py.get_search_actions()]
            if not actions:
                break
            choice = rng.choice(actions)
            py = py.clone()
            rs = rs.clone_game()
            py.apply_search_action(
                next(m for m in py.get_search_actions() if m.uci() == choice))
            rs.apply_search_action(choice)
    return pairs[:n]


def timed(fn, reps=1):
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return time.perf_counter() - start


def main():
    results = {}
    states = build_states(1500)
    py_states = [p for p, _ in states]
    rs_states = [r for _, r in states]

    # --- heuristic evaluation ------------------------------------------
    t_py = timed(lambda: [evaluate(g) for g in py_states])
    t_rs = timed(lambda: [g.evaluate() for g in rs_states])
    results["heuristic_evaluate"] = (len(states), t_py, t_rs)

    # --- half-move generation ------------------------------------------
    t_py = timed(lambda: [g.get_search_actions() for g in py_states])
    t_rs = timed(lambda: [g.search_actions() for g in rs_states])
    results["search_actions"] = (len(states), t_py, t_rs)

    # --- atomic White pair generation (the ~900-branch one) -------------
    white_py = [g for g in py_states if g.is_white_turn][:400]
    white_rs = [r for p, r in states if p.is_white_turn][:400]
    t_py = timed(lambda: [g.get_legal_actions() for g in white_py])
    t_rs = timed(lambda: [g.legal_actions(False) for g in white_rs])
    results["white_atomic_pairs"] = (len(white_py), t_py, t_rs)

    # --- clone ----------------------------------------------------------
    t_py = timed(lambda: [g.clone() for g in py_states])
    t_rs = timed(lambda: [g.clone_game() for g in rs_states])
    results["clone"] = (len(states), t_py, t_rs)

    # --- full random playout (composite: gen + apply + terminal + fen) ---
    def playout(native, seed):
        rng = random.Random(seed)
        for i in range(40):
            g = mn.Game(START_FEN) if native else MonsterChessGame(START_FEN)
            for _ in range(200):
                if g.is_terminal():
                    break
                acts = g.search_actions() if native else g.get_search_actions()
                if not acts:
                    break
                if native:
                    g.apply_search_action(rng.choice(acts))
                else:
                    g.apply_search_action(rng.choice(acts))

    t_py = timed(lambda: playout(False, 99))
    t_rs = timed(lambda: playout(True, 99))
    results["random_playout_40_games"] = (40, t_py, t_rs)

    print("%-26s %8s %12s %12s %9s" % ("component", "n", "python(ms)", "native(ms)", "speedup"))
    print("-" * 72)
    out = {}
    for name, (n, tp, tr) in results.items():
        speed = tp / tr if tr else float("inf")
        out[name] = {"n": n, "python_ms": round(tp * 1000, 1),
                     "native_ms": round(tr * 1000, 1), "speedup": round(speed, 2)}
        print("%-26s %8d %12.1f %12.1f %8.2fx" % (name, n, tp * 1000, tr * 1000, speed))

    path = os.path.join(ROOT, "benchmarks",
                        f"native_speedup_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"components": out, "note": "per-call PyO3 crossing included; "
                   "lower bound for the finished design",
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
