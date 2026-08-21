"""Replay named opening lines from a FEN and export them to scrubbable HTML.

`export_selfplay_replays.py` always begins at the true opening and picks games
by seed, so it cannot show a specific LINE. This starts from given positions
instead, which is what an opening study needs.

Written for the 2026-08-20/21 result: exhaustive enumeration of White's first
turn gives 58 families, of which only two survive deep search -- e2-e4-e5 and
d2-d4-d5, going 33W/0L/5D across all 38 legal Black replies at 3200 sims. The
five drawing lines are the whole remaining question, and they are the ones
worth reading move by move.

Play here is DETERMINISTIC: temperature 0, no root noise, so one position is
one game and re-running reproduces it exactly. Threefold repetition is applied,
matching every other current path -- the sibling exporter had shipped without
it once and its drawn games all ran to the 150-turn cap, which is the bug this
file must not repeat.

Rendering is reused from export_selfplay_replays so both exports look the same
and there is one board widget to maintain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark import _apply, _build_engine  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from export_selfplay_replays import render_html  # noqa: E402


def play_line(label, fen, engine_w, engine_b, max_plies=600,
              start_half=False, start_turn_count=0):
    """Play one position out and return every frame."""
    from repetition import RepetitionTracker

    game = MonsterChessGame(fen)
    game.white_half_pending = bool(start_half)
    game.turn_count = int(start_turn_count)
    frames = [{"fen": game.fen(), "move": None, "actor": None,
               "white_half": None, "label": f"{label} -- start"}]
    repetition = RepetitionTracker()
    repetition.record(game, 0)
    repeated = False
    plies = 0
    while not game.is_terminal() and plies < max_plies:
        is_white = bool(game.is_white_turn)
        pending = bool(getattr(game, "white_half_pending", False))
        engine = engine_w if is_white else engine_b
        action, _p, _v = engine.get_best_action(game, temperature=0.0)
        if action is None:
            break
        uci = action.uci()
        _apply(game, action)
        plies += 1
        half = 2 if is_white and pending else (1 if is_white else None)
        actor = "White" if is_white else "Black"
        detail = (f" ({'first' if half == 1 else 'second'} move)"
                  if half else "")
        frames.append({"fen": game.fen(), "move": uci, "actor": actor,
                       "white_half": half,
                       "label": f"{plies}. {actor}{detail}: {uci}"})
        if repetition.record(game, plies):
            repeated = True
            frames[-1]["label"] += "  -- threefold repetition, drawn"
            break
    result = (float(repetition.draw_result) if repeated
              else float(game.get_result()))
    verdict = ("White wins" if result >= 1 else
               "Black wins" if result <= -1 else "Draw")
    return {"seed": 0, "category": "line", "category_label": label,
            "title": f"{label} -- {verdict} in {plies} plies",
            "result_white_perspective": result, "plies": plies,
            "frames": frames}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--lines", required=True,
                    help="JSON: [{label, fen, half?, turn_count?}, ...]")
    ap.add_argument("--sims", type=int, default=3200)
    ap.add_argument("--c-puct", type=float, default=None)
    ap.add_argument("--fpu-reduction", type=float, default=None)
    ap.add_argument("--policy-temperature", type=float, default=None)
    ap.add_argument("--title", default="Opening lines")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = str(ROOT / args.model) if not Path(args.model).is_absolute() \
        else args.model
    spec = json.loads((ROOT / args.lines).read_text(encoding="utf-8")) \
        if not Path(args.lines).is_absolute() \
        else json.loads(Path(args.lines).read_text(encoding="utf-8"))

    kwargs = {}
    for key, val in (("c_puct", args.c_puct),
                     ("fpu_reduction", args.fpu_reduction),
                     ("policy_temperature", args.policy_temperature)):
        if val is not None:
            kwargs[key] = val
    # TWO engines, one per colour -- not one shared object. NativeMCTS reuses
    # its tree across moves (reuse_across_moves defaults on), so a single
    # instance driving both sides carries one side's tree into the other's
    # search. tools/match.py builds `_engines["a"]` and `_engines["b"]`
    # separately for exactly this reason. Sharing one engine here made this
    # exporter disagree with the match it is supposed to illustrate: e2-e4-e5
    # against ...f6 is a DRAW in 45 plies in the 3200-sim map and came out
    # "White wins in 50" from the shared-engine build.
    engine_w, _label = _build_engine(model, args.sims, engine="native",
                                     **kwargs)
    engine_b, _label_b = _build_engine(model, args.sims, engine="native",
                                       **kwargs)

    games = []
    for entry in spec:
        label = entry["label"]
        print(f"  playing {label} ...", flush=True)
        game = play_line(label, entry["fen"], engine_w, engine_b,
                         start_half=entry.get("half", False),
                         start_turn_count=entry.get("turn_count", 0))
        print(f"    {game['title']}", flush=True)
        games.append(game)

    payload = {"title": args.title, "checkpoint": args.model,
               "sims": args.sims, "search": kwargs or "defaults",
               "selection": {"lines": len(games)}, "games": games}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2),
                                        encoding="utf-8")
    out.with_suffix(".html").write_text(render_html(payload),
                                        encoding="utf-8")
    print(json.dumps({"lines": len(games),
                      "html": str(out.with_suffix('.html')),
                      "json": str(out.with_suffix('.json'))}, indent=2))


if __name__ == "__main__":
    main()
