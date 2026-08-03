"""Does the existing scripted oracle convert the positions it took over?"""
import glob
import json
import os
import sys

sys.path.insert(0, "src")
import chess  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from scripted_mate import ScriptedMate, mate_algo_applicable  # noqa: E402
from forced_capture import forced_capture_depth  # noqa: E402

d = json.load(open(sorted(glob.glob("benchmarks/forced_capture_v20asym*.json"))[-1]))

rows = []
for g in d["games"]:
    hits = [h for h in g["hits"] if h["verdict"] == "walked_past"]
    if not hits:
        continue
    h = hits[0]
    st = MonsterChessGame(h["fen"])
    if mate_algo_applicable(st):
        rows.append((os.path.basename(g["path"]), h))

print("walked-past positions where the oracle is applicable:", len(rows))
print()


def white_best_defence(state):
    """White picks the reply that survives longest against a depth-2 threat."""
    best, best_score = None, -1
    for a in state.get_legal_actions():
        c = state.clone()
        c.apply_action(a)
        if c.board.king(chess.BLACK) is None:
            return a  # White just wins
        c.turn_count = 0
        c._terminal = False
        c._result = None
        try:
            dd = forced_capture_depth(c, max_black_moves=2, node_budget=60000)
        except Exception:
            dd = None
        score = 99 if dd is None else dd
        if score > best_score:
            best, best_score = a, score
    return best


converted = 0
sample = rows[:8]
for name, h in sample:
    st = MonsterChessGame(h["fen"])
    st.turn_count = 0
    st._terminal = False
    st._result = None
    bot = ScriptedMate()
    captured_at = None
    for ply in range(1, 41):
        m = bot.select_move(st)
        if m is None:
            break
        st.apply_action(m)
        if st.board.king(chess.WHITE) is None:
            captured_at = ply
            break
        wa = white_best_defence(st)
        if wa is None:
            break
        st.apply_action(wa)
        st.turn_count = 0
        st._terminal = False
        st._result = None
    if captured_at:
        converted += 1
    print("  %-20s solver: forced capture in %d | oracle: %s"
          % (name, h["depth"],
             ("captured after %d Black moves" % captured_at) if captured_at
             else ">40 Black moves, NO CAPTURE"))

print()
print("oracle converted %d of %d sampled walked-past positions" % (converted, len(sample)))
