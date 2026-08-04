"""E1: lockstep the native game state machine against the Python engine.

Both engines start from the same FEN and are driven through the *same* chosen
action every ply. After each application we compare the whole observable state:
FEN, `is_white_turn`, `turn_count`, `white_half_pending`, terminality, result,
and both action lists.

Lockstep is what makes this meaningful. Comparing two independent playouts
would let a divergence hide behind different move choices; here any difference
is attributable to the ply that just happened.

    py -3 tools/state_machine_parity.py --games 400
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "native"))

import monster_native as mn  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def py_search_actions(game):
    return [m.uci() for m in game.get_search_actions()]


def py_legal_actions(game):
    actions = game.get_legal_actions()
    if game.is_white_turn:
        return [f"{m1.uci()},{m2.uci()}" for m1, m2 in actions]
    return [m.uci() for m in actions]


def compare(py, rs, phase, fails):
    def note(field, a, b):
        if len(fails) < 30:
            fails.append({"phase": phase, "field": field,
                          "python": a if not isinstance(a, set) else sorted(a)[:8],
                          "native": b if not isinstance(b, set) else sorted(b)[:8],
                          "fen": py.fen()})

    if py.fen() != rs.fen():
        note("fen", py.fen(), rs.fen())
        return False
    if py.is_white_turn != rs.is_white_turn:
        note("is_white_turn", py.is_white_turn, rs.is_white_turn)
    if py.turn_count != rs.turn_count:
        note("turn_count", py.turn_count, rs.turn_count)
    if py.white_half_pending != rs.white_half_pending:
        note("white_half_pending", py.white_half_pending, rs.white_half_pending)

    py_term = py.is_terminal()
    if py_term != rs.is_terminal():
        note("is_terminal", py_term, rs.is_terminal())
    if py_term:
        # At the cap the native side reports None on purpose: the ±0.5 relabel
        # needs E2's heuristic. Only decisive results are comparable today.
        if not rs.at_turn_cap():
            if float(py.get_result()) != rs.result():
                note("result", py.get_result(), rs.result())
        return False

    if set(py_search_actions(py)) != set(rs.search_actions()):
        note("search_actions", set(py_search_actions(py)), set(rs.search_actions()))
    if set(py_legal_actions(py)) != set(rs.legal_actions(False)):
        note("legal_actions", set(py_legal_actions(py)), set(rs.legal_actions(False)))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fails = []
    plies = 0
    games_completed = 0
    started = time.time()

    for _ in range(args.games):
        py = MonsterChessGame(START_FEN)
        rs = mn.Game(START_FEN)
        if not compare(py, rs, "start", fails):
            continue
        for _ in range(args.max_plies):
            actions = py_search_actions(py)
            if not actions:
                break
            choice = rng.choice(actions)
            py.apply_search_action(next(m for m in py.get_search_actions()
                                        if m.uci() == choice))
            rs.apply_search_action(choice)
            plies += 1
            if not compare(py, rs, "after " + choice, fails):
                break
        games_completed += 1

    elapsed = time.time() - started
    summary = {
        "games": games_completed,
        "plies_compared": plies,
        "failures": len(fails),
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "examples": fails[:12],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"state_machine_parity_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
    for f in fails[:8]:
        print(" ", json.dumps(f)[:300])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
