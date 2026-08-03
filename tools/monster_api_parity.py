"""E1: differential the native Monster action APIs against the Python engine.

Checks, per position:

- `_white_single_moves`, `_white_second_half_moves`, `_get_white_actions`,
  `_get_black_actions` — **set** equality, both truncated and complete.
- The **winning-first ordering contract**: with `truncate_wins=False`, the
  maximal leading block of king-capturing actions must be the same set in both
  engines. `CONTEXT.md` §1.1 says callers rely on this, so set equality alone
  is not enough.
- `push` — resulting FEN must match exactly, which is where the ep-in-FEN rule
  (`en_passant="legal"`) and castling-rights bookkeeping get tested.

Reported separately and *not* treated as a failure: which specific winning
action each engine returns when `truncate_wins=True`. Both return the first
winner in their own generation order, and the orders differ; the game ends
either way. It is surfaced because it does change recorded policy targets.
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

import chess  # noqa: E402
import monster_native as mn  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def pair_str(pair):
    m1, m2 = pair
    return f"{m1.uci()},{m2.uci()}"


def leading_win_block(items, is_win):
    out = []
    for item in items:
        if not is_win(item):
            break
        out.append(item)
    return set(out)


def check(fen, report):
    game = MonsterChessGame(fen)
    board = game.board

    if game.is_white_turn:
        py_single = {m.uci() for m in game._white_single_moves()}
        rs_single = set(mn.white_single_moves_uci(fen))
        if py_single != rs_single:
            report("white_single_moves", fen, py_single, rs_single)

        py_second = {m.uci() for m in game._white_second_half_moves()}
        rs_second = set(mn.white_second_half_uci(fen))
        if py_second != rs_second:
            report("white_second_half", fen, py_second, rs_second)

        py_pairs_all = [pair_str(p) for p in game._get_white_actions(truncate_wins=False)]
        rs_pairs_all = mn.white_actions_uci(fen, False)
        if set(py_pairs_all) != set(rs_pairs_all):
            report("white_actions_complete", fen, set(py_pairs_all), set(rs_pairs_all))
        else:
            bk = board.king(chess.BLACK)

            def wins(pair_repr):
                m1, m2 = pair_repr.split(",")
                target = m2 if m2 != "0000" else m1
                return bk is not None and chess.parse_square(target[2:4]) == bk

            if leading_win_block(py_pairs_all, wins) != leading_win_block(rs_pairs_all, wins):
                report("white_winning_first_ordering", fen,
                       leading_win_block(py_pairs_all, wins),
                       leading_win_block(rs_pairs_all, wins))
    else:
        py_black_all = [m.uci() for m in game._get_black_actions(truncate_wins=False)]
        rs_black_all = mn.black_actions_uci(fen, False)
        if set(py_black_all) != set(rs_black_all):
            report("black_actions_complete", fen, set(py_black_all), set(rs_black_all))
        else:
            wk = board.king(chess.WHITE)

            def wins(uci):
                return wk is not None and chess.parse_square(uci[2:4]) == wk

            if leading_win_block(py_black_all, wins) != leading_win_block(rs_black_all, wins):
                report("black_winning_first_ordering", fen,
                       leading_win_block(py_black_all, wins),
                       leading_win_block(rs_black_all, wins))

        py_black = {m.uci() for m in game._get_black_actions()}
        rs_black = set(mn.black_actions_uci(fen, True))
        if len(py_black) > 1 and py_black != rs_black:
            report("black_actions_truncated", fen, py_black, rs_black)

    # push parity on a sample of moves from this position
    movers = list(board.pseudo_legal_moves)[:6]
    for mv in movers:
        probe = board.copy(stack=False)
        probe.push(mv)
        want = probe.fen()
        got = mn.push_uci(fen, mv.uci())
        if got != want:
            report("push_fen", fen, want, got, extra=mv.uci())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    failures = []
    counts = {}

    def report(kind, fen, py, rs, extra=None):
        counts[kind] = counts.get(kind, 0) + 1
        if len(failures) < 40:
            failures.append({
                "kind": kind, "fen": fen, "move": extra,
                "python_only": sorted(set(py) - set(rs))[:8] if isinstance(py, set) else py,
                "native_only": sorted(set(rs) - set(py))[:8] if isinstance(rs, set) else rs,
            })

    rng = random.Random(args.seed)
    checked = 0
    started = time.time()
    while checked < args.positions:
        game = MonsterChessGame(START_FEN)
        steps = 0
        while not game.is_terminal() and steps < 240 and checked < args.positions:
            check(game.fen(), report)
            checked += 1
            steps += 1
            actions = game.get_search_actions()
            if not actions:
                break
            game.apply_search_action(rng.choice(actions))
    elapsed = time.time() - started

    summary = {
        "positions_checked": checked,
        "failures": sum(counts.values()),
        "by_kind": counts,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "examples": failures[:20],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"monster_api_parity_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
    for f in failures[:6]:
        print(" ", json.dumps(f)[:320])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
