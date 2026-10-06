"""Descriptive repertoire snapshot from completed journals; no search or training."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("gen44", "iterations/gen_0045/reports/sampled_gate/vs_bar.jsonl", 400),
    ("gen44", "iterations/gen_0045/reports/sampled_gate/vs_bar_confirm.jsonl", 400),
    ("gen42", "benchmarks/gen45_vs_gen42_free_3200.lines.jsonl", 200),
    ("self", "iterations/gen_0045/reports/self_skew.jsonl", 200),
]


def notation(before, uci):
    board = chess.Board(before["fen"])
    move = chess.Move.from_uci(uci)
    piece = board.piece_at(move.from_square)
    assert piece is not None
    capture = board.piece_at(move.to_square) is not None or board.is_en_passant(move)
    name = "" if piece.piece_type == chess.PAWN else piece.symbol().upper()
    if capture and piece.piece_type == chess.PAWN:
        name = chess.square_name(move.from_square)[0]
    return name + ("x" if capture else "") + chess.square_name(move.to_square) + (
        "=" + chess.piece_symbol(move.promotion).upper() if move.promotion else "")


def opening(row):
    frames = row["game"]["trajectory"]
    turns, pending = [], []
    for previous, frame in zip(frames, frames[1:]):
        assert len(frame["action"]) == 1
        pending.append(notation(previous, frame["action"][0]))
        if not frame["half"]:
            turns.append("/".join(pending))
            pending = []
        if len(turns) == 6:
            break
    return turns


def counts(values, limit=8):
    counter = Counter(values)
    total = sum(counter.values())
    return [{"line": key, "n": n, "percent": round(n / total * 100, 2)}
            for key, n in counter.most_common(limit)]


def run():
    games, hashes = [], {}
    for opponent, relative, expected in SOURCES:
        path = ROOT / relative
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len(rows) == expected, (relative, len(rows))
        assert len({row["task_id"] for row in rows}) == expected
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        for row in rows:
            turns = opening(row)
            assert len(turns) >= 2
            white_result = row["result_for_a"] * (1 if row["a_is_white"] else -1)
            games.append({"opponent": opponent, "candidate_white": row["a_is_white"],
                          "turns": turns, "white_result": white_result})
    report = {"source_hashes": hashes, "total_games": len(games), "sides": {}}
    lines = ["# Gen45 observed opening repertoire — September 6, 2026", "",
             "All sources are completed free-opening games at 3,200 simulations per move: "
             "800 against gen44, 200 against gen42, and 200 self-play games.", "",
             "These are observed frequencies under temperature 0.5 for the first 16 search plies, "
             "not zero-temperature rankings or a uniform opening book. Self-play contributes "
             "to both color maps; opponent moves are conditional context, not gen45 choices.", "",
             "Notation: `e4/e5` means White uses both moves to advance e2–e4–e5. "
             "Piece letters/captures are shown; check markers and SAN disambiguation are omitted.", ""]
    for side in ("white", "black"):
        subset = [g for g in games if g["opponent"] == "self" or g["candidate_white"] == (side == "white")]
        side_out = {"games": len(subset), "by_opponent": {}, "first_white_turn": counts(g["turns"][0] for g in subset)}
        lines += [f"## Gen45 as {side.title()} ({len(subset)} games)", ""]
        for opponent in ("self", "gen44", "gen42"):
            group = [g for g in subset if g["opponent"] == opponent]
            responses = defaultdict(list)
            for g in group:
                turns = g["turns"]
                context = (f"1. {turns[0]} {turns[1]}" if side == "white" else f"1. {turns[0]}")
                response_index = 2 if side == "white" else 1
                if len(turns) > response_index:
                    responses[context].append(turns[response_index])
            branches = [{"context": context, "n": len(values), "responses": counts(values, 5)}
                        for context, values in sorted(responses.items(), key=lambda item: -len(item[1]))]
            prefixes = counts(" ".join(f"{i//2+1}. {g['turns'][i]} {g['turns'][i+1] if len(g['turns'])>i+1 else ''}"
                                      for i in range(0, min(6, len(g['turns'])), 2)) for g in group)
            first = counts(g["turns"][0] for g in group)
            side_out["by_opponent"][opponent] = {"games": len(group), "first_white_turn": first,
                                                "responses": branches, "top_three_turn_lines": prefixes}
            lines += [f"### Against {opponent} ({len(group)} games)", "",
                      "First White turn: " + "; ".join(f"{r['line']} {r['n']}/{len(group)} ({r['percent']}%)" for r in first), "",
                      "| Position reached | Games | Gen45 response frequencies within that branch |",
                      "|---|---:|---|"]
            for branch in branches[:8]:
                text = "; ".join(f"{r['line']}: {r['n']}/{branch['n']} ({r['percent']}%)" for r in branch["responses"])
                lines.append(f"| {branch['context']} | {branch['n']} | {text} |")
            lines += ["", "Most frequent complete three-turn prefixes:", ""]
            lines += [f"- `{r['line']}` — {r['n']}/{len(group)} ({r['percent']}%)." for r in prefixes]
            lines += [""]
        report["sides"][side] = side_out
    return report, "\n".join(lines)


if __name__ == "__main__":
    report, markdown = run()
    # Outputs are new derived diagnostics, never source journals.
    for name, content in (("gen45_opening_map_20260906.json", json.dumps(report, indent=2)),
                          ("gen45_opening_map_20260906.md", markdown)):
        with (ROOT / "benchmarks" / name).open("x", encoding="utf-8") as stream:
            stream.write(content + "\n")
    print(markdown)
