"""Fetch human-vs-human Monster games and build the data bundle for the viewer.

Reads ps_monster_census.json, keeps only human-vs-human games, pulls each game's
PGN from /game/export/{id}, replays it through our own rules to validate, and
emits ps_games.json:

    {"players": {name: {...h2h...}}, "games": [{id, p1, p2, result, uci, san, ...}]}

Stores UCI moves rather than per-ply FENs so the page stays small; the viewer
applies them client-side. Every game here has been replayed successfully by
MonsterChessGame + play.parse_move, so the move list is known-good.
"""
import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import chess  # noqa: E402
from config import STARTING_FEN  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from play import parse_move  # noqa: E402
from import_playstrategy import movetext_tokens  # noqa: E402

UA = {"User-Agent": "monster-chess-nn research (contact: perfpeng@gmail.com)",
      "Accept": "application/x-chess-pgn"}
CENSUS = os.path.join(ROOT, "data", "playstrategy", "ps_monster_census.json")
OUT = os.path.join(ROOT, "data", "playstrategy", "ps_games.json")
RESULT = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def fetch_game(gid):
    req = urllib.request.Request(f"https://playstrategy.org/game/export/{gid}",
                                 headers=UA)
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(60)
                continue
            return ""
        except Exception:
            time.sleep(4)
    return ""


def replay(movetext):
    """-> (uci_per_ply, san_per_ply, turn_index_per_ply) or (None, err, None)."""
    game = MonsterChessGame(fen=STARTING_FEN)
    uci, san, turn = [], [], []
    t = 0
    for token in movetext_tokens(movetext):
        if game.is_terminal():
            return None, "moves after terminal", None
        for s in (token.split(",") if "," in token else [token]):
            legal = game.get_search_actions()
            mv = parse_move(s, game.board, legal_set=legal)
            if mv is None:
                return None, f"unparseable {s!r}", None
            uci.append(mv.uci())
            san.append(s)
            turn.append(t)
            game.apply_search_action(mv)
        t += 1
    return uci, san, turn


def main():
    census = json.load(open(CENSUS, encoding="utf-8"))
    isbot = lambda h, p: h.get(f"{p}Title") == "BOT"  # noqa: E731
    hh = {k: h for k, h in census.items()
          if not isbot(h, "P1") and not isbot(h, "P2")}
    print(f"{len(census)} census games -> {len(hh)} human-vs-human", flush=True)

    games, failed = [], collections.Counter()
    if os.path.exists(OUT):                       # resume: keep what we already have
        try:
            games = json.load(open(OUT, encoding="utf-8"))["games"]
            have = {g["id"] for g in games}
            hh = {k: v for k, v in hh.items() if k not in have}
            print(f"resumed: {len(games)} already fetched, {len(hh)} to go", flush=True)
        except Exception:
            games = []

    for i, (gid, h) in enumerate(sorted(hh.items()), 1):
        pgn = fetch_game(gid)
        time.sleep(1.1)
        if not pgn:
            failed["fetch"] += 1
            continue
        headers = dict(re.findall(r'\[(\w+)\s+"([^"]*)"\]', pgn))
        if headers.get("FEN", STARTING_FEN) != STARTING_FEN:
            failed["non-standard start"] += 1
            continue
        # split on the blank line after the header block: clock comments
        # like "{ [%clk 0:03:00] }" contain "]", so rsplit("]") is wrong
        movetext = pgn.split("\n\n", 1)[1].strip() if "\n\n" in pgn else ""
        uci, san, turn = replay(movetext)
        if uci is None:
            failed[f"replay: {san}"] += 1
            continue
        games.append({
            "id": gid,
            "p1": headers.get("P1", "?"), "p2": headers.get("P2", "?"),
            "e1": headers.get("P1Elo", ""), "e2": headers.get("P2Elo", ""),
            "res": RESULT.get(headers.get("Result"), 0),
            "date": headers.get("UTCDate", ""), "tc": headers.get("TimeControl", ""),
            "ev": headers.get("Event", ""), "term": headers.get("Termination", ""),
            "uci": uci, "san": san, "turn": turn,
        })
        if i % 25 == 0:
            print(f"  [{i}/{len(hh)}] ok={len(games)} failed={sum(failed.values())}",
                  flush=True)
            json.dump({"games": games}, open(OUT, "w", encoding="utf-8"))

    # per-player records and head-to-head
    players = collections.defaultdict(
        lambda: {"w": 0, "d": 0, "l": 0, "asW": [0, 0, 0], "asB": [0, 0, 0],
                 "elo": 0, "vs": {}})
    for g in games:
        for side, name in (("W", g["p1"]), ("B", g["p2"])):
            opp = g["p2"] if side == "W" else g["p1"]
            won = (g["res"] > 0) if side == "W" else (g["res"] < 0)
            drew = g["res"] == 0
            p = players[name]
            key = "w" if won else ("d" if drew else "l")
            p[key] += 1
            p["asW" if side == "W" else "asB"][0 if won else (1 if drew else 2)] += 1
            elo = g["e1"] if side == "W" else g["e2"]
            if elo.isdigit():
                p["elo"] = max(p["elo"], int(elo))
            rec = p["vs"].setdefault(opp, [0, 0, 0])
            rec[0 if won else (1 if drew else 2)] += 1

    json.dump({"games": games, "players": players},
              open(OUT, "w", encoding="utf-8"))
    print(f"\nusable games: {len(games)}   players: {len(players)}")
    print("failures:", dict(failed))
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
