"""Import Monster-chess games from playstrategy.org into our raw JSONL schema.

Why this is safe to do at all (verified 2026-07-25, not assumed):

  * Their starting FEN is byte-identical to config.STARTING_FEN.
  * Their PGN encodes White's double move as a comma pair, "1. f4,Kf2 e6",
    which is exactly our atomic (m1, m2) action.
  * play.parse_move already handles the Monster-legal SAN that standard
    validators reject (king-captures-king, first half-move through check).
  * Their win condition is checkmate, ours is king capture. A PGN checkmate is
    a FORCED king capture one turn later — enumerated on a real game: all 12
    Black replies lost the king. So Result maps directly to game_result and the
    only cost is ~2 missing plies, worth gamma^2 ~ 0.977 on the ramp label.

What an imported record can and cannot carry: PGN has no search output, so
there is no mcts_value and no visit distribution. Records are POLICY TEACHERS
with outcome labels: policy is a one-hot on the move actually played, and
mcts_value is 0.0 (unused unless VALUE_TARGET="mcts_value", which is not the
recipe). Do not read these as if they carried search information.

By default only the WINNER's moves teach policy, mirroring the anti-echo rule
for human games (policy_weight_for_record): a loser's moves in a mismatched
pairing are not demonstrations worth imitating. --all-moves overrides.

    py -3 tools/import_playstrategy.py --users woll,oruro --out-dir data/raw/ps_monster
    py -3 tools/import_playstrategy.py --census ps_monster_census.json --validate-only
"""
import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chess  # noqa: E402

from config import STARTING_FEN  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402
from play import parse_move  # noqa: E402

UA = {"User-Agent": "monster-chess-nn research (contact: perfpeng@gmail.com)",
      "Accept": "application/x-chess-pgn"}
API = "https://playstrategy.org/api/games/user/{}?perfType=monster&moves=true"
RESULT_MAP = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def fetch_user(user, timeout=180):
    req = urllib.request.Request(API.format(urllib.parse.quote(user)), headers=UA)
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(60)
                continue
            return ""
        except Exception:
            time.sleep(5)
    return ""


def split_games(pgn):
    """Yield (headers, movetext) per game in a concatenated PGN stream."""
    for block in pgn.split("[Event "):
        if not block.strip():
            continue
        block = "[Event " + block
        headers = dict(re.findall(r'\[(\w+)\s+"([^"]*)"\]', block))
        # Split on the blank line after the headers. Not rsplit("]"): clock
        # comments ("{ [%clk 0:03:00] }") contain a bracket and would win.
        tail = block.split("\n\n", 1)[1] if "\n\n" in block else ""
        yield headers, tail.strip()


def movetext_tokens(movetext):
    """PGN movetext -> per-turn tokens, move numbers and result stripped."""
    body = re.sub(r"\{[^}]*\}", " ", movetext)      # comments
    body = re.sub(r"\$\d+", " ", body)              # NAGs
    body = re.sub(r"\b\d+\.(\.\.)?", " ", body)     # move numbers
    body = re.sub(r"\b(1-0|0-1|1/2-1/2|\*)\s*$", " ", body)
    return [t for t in body.split() if t and t not in RESULT_MAP and t != "*"]


def convert_game(headers, movetext, winner_only=True):
    """Replay a PGN game through our rules; return (records, error).

    One record per TURN (White's turn is the atomic m1,m2 pair), matching what
    the notebook writes for human games.
    """
    result = RESULT_MAP.get(headers.get("Result"))
    if result is None:
        return None, f"unmapped Result {headers.get('Result')!r}"
    start = headers.get("FEN", STARTING_FEN)
    if start != STARTING_FEN:
        return None, "non-standard start position"

    game = MonsterChessGame(fen=start)
    records = []
    for token in movetext_tokens(movetext):
        if game.is_terminal():
            return None, "moves remain after terminal position"
        is_white = game.is_white_turn
        fen_before = game.fen() if callable(getattr(game, "fen", None)) else game.board.fen()
        sans = token.split(",") if "," in token else [token]
        if is_white and len(sans) > 2:
            return None, f"White token with {len(sans)} moves: {token!r}"

        played = []
        for san in sans:
            legal = game.get_search_actions()
            move = parse_move(san, game.board, legal_set=legal)
            if move is None:
                return None, f"unparseable SAN {san!r} in {token!r}"
            played.append(move)
            game.apply_search_action(move)

        if is_white:
            m1 = played[0]
            m2 = played[1] if len(played) > 1 else chess.Move.null()
            action_str = f"{m1.uci()},{m2.uci()}"
        else:
            action_str = played[0].uci()

        records.append({
            "fen": fen_before,
            "mcts_value": 0.0,               # PGN carries no search value
            "current_player": "white" if is_white else "black",
            "source": "playstrategy",
            "actor": "human",
            "game_result": result,
            "policy": {action_str: 1.0},     # one-hot on the move actually played
        })

    if not records:
        return None, "no moves"

    if winner_only and result != 0:
        win_side = "white" if result > 0 else "black"
        for rec in records:
            rec["policy_weight"] = 1.0 if rec["current_player"] == win_side else 0.0
    return records, None


def convert_from_bundle(g, winner_only=True):
    """Same records as convert_game, replayed from stored UCI (no network).

    The bundle written by ps_build_viewer_data.py has already been replayed
    once through these rules, so this path re-derives FENs without re-parsing
    SAN or re-fetching.
    """
    result = g["res"]
    game = MonsterChessGame(fen=STARTING_FEN)
    records = []
    turns = collections.defaultdict(list)
    for u, t in zip(g["uci"], g["turn"]):
        turns[t].append(u)

    for t in sorted(turns):
        if game.is_terminal():
            return None, "moves after terminal"
        is_white = game.is_white_turn
        fen_before = game.board.fen()
        played = []
        for u in turns[t]:
            mv = chess.Move.from_uci(u)
            played.append(mv)
            game.apply_search_action(mv)
        if is_white:
            m2 = played[1] if len(played) > 1 else chess.Move.null()
            action = f"{played[0].uci()},{m2.uci()}"
        else:
            action = played[0].uci()
        records.append({
            "fen": fen_before, "mcts_value": 0.0,
            "current_player": "white" if is_white else "black",
            "source": "playstrategy", "actor": "human",
            "game_result": result, "policy": {action: 1.0},
        })
    if not records:
        return None, "no moves"
    if winner_only and result != 0:
        win = "white" if result > 0 else "black"
        for r in records:
            r["policy_weight"] = 1.0 if r["current_player"] == win else 0.0
    return records, None


def balance_by_position(kept, seed):
    """Drop games from the majority outcome until POSITION counts match.

    Positions, not games: Black's wins are systematically longer, so the two
    balance at different cuts. Deterministic given the seed, and it drops whole
    games rather than truncating any, so no game is left half-labelled.
    """
    import random

    by = {1: [], -1: []}
    for gid, recs in kept:
        by[1 if recs[0]["game_result"] > 0 else -1].append((gid, recs))
    pos = {k: sum(len(r) for _, r in v) for k, v in by.items()}
    minority = min(pos, key=pos.get)
    target = pos[minority]

    majority = -minority
    pool = list(by[majority])
    random.Random(seed).shuffle(pool)
    chosen, running = [], 0
    for gid, recs in pool:
        if running + len(recs) > target:
            continue                      # keep scanning: shorter games may still fit
        chosen.append((gid, recs))
        running += len(recs)

    print(f"\n=== balance ===")
    side = lambda k: "White-win" if k == 1 else "Black-win"  # noqa: E731
    print(f"  before: {side(1)} {pos[1]:,} pos ({len(by[1])} games)   "
          f"{side(-1)} {pos[-1]:,} pos ({len(by[-1])} games)")
    print(f"  dropped {len(by[majority]) - len(chosen)} {side(majority)} games "
          f"to match {side(minority)}")
    print(f"  after:  {side(minority)} {target:,} pos   "
          f"{side(majority)} {running:,} pos   (delta {abs(target-running)})")
    return by[minority] + chosen


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--users", help="comma-separated playstrategy usernames")
    src.add_argument("--census", help="ps_monster_census.json from ps_census.py")
    src.add_argument("--bundle", help="ps_games.json from ps_build_viewer_data.py "
                                      "(already fetched and replayed; no network)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "raw", "ps_monster"))
    ap.add_argument("--validate-only", action="store_true",
                    help="replay and report, write nothing")
    ap.add_argument("--include-bots", action="store_true",
                    help="keep games where either side is a BOT (default: drop)")
    ap.add_argument("--include-draws", action="store_true")
    ap.add_argument("--all-moves", action="store_true",
                    help="both sides teach policy (default: winner only)")
    ap.add_argument("--min-elo", type=int, default=0,
                    help="drop games unless BOTH humans are rated at least this. "
                         "Elo, not win rate: score in a closed pool averages 0.5 "
                         "by construction, so a high win rate can just mean weak "
                         "opposition. Requiring both sides keeps games where the "
                         "winner's moves were actually tested")
    ap.add_argument("--balance", action="store_true",
                    help="equalise POSITIONS labelled White-win vs Black-win by "
                         "dropping games from the majority outcome (seeded, "
                         "reproducible). Black wins run longer, so game counts "
                         "and position counts do not balance at the same cut")
    ap.add_argument("--balance-seed", type=int, default=42)
    ap.add_argument("--manifest", help="write the selected game ids here (JSON)")
    args = ap.parse_args()

    if args.bundle:
        bundle = json.load(open(args.bundle, encoding="utf-8"))["games"]
        # best Elo seen per player, so a filter reflects strength not one game
        best = collections.defaultdict(int)
        for g in bundle:
            for name, elo in ((g["p1"], g["e1"]), (g["p2"], g["e2"])):
                if str(elo).isdigit():
                    best[name] = max(best[name], int(elo))
        kept, stats = [], collections.Counter()
        for g in bundle:
            if g["res"] == 0 and not args.include_draws:
                stats["skip: draw"] += 1
                continue
            if args.min_elo and min(best[g["p1"]], best[g["p2"]]) < args.min_elo:
                stats["skip: below --min-elo"] += 1
                continue
            recs, err = convert_from_bundle(g, winner_only=not args.all_moves)
            if err:
                stats[f"FAIL: {err}"] += 1
                continue
            stats["converted"] += 1
            kept.append((g["id"], recs))
        print(f"bundle: {len(bundle)} games -> {len(kept)} selected")
        if args.balance:
            kept = balance_by_position(kept, args.balance_seed)
        print("\n=== summary ===")
        for k, v in stats.most_common():
            print(f"  {k:34s} {v}")
        print(f"  final games                        {len(kept)}")
        print(f"  final positions                    {sum(len(r) for _, r in kept)}")
        if args.manifest:
            json.dump(sorted(gid for gid, _ in kept),
                      open(args.manifest, "w", encoding="utf-8"))
            print(f"  manifest -> {args.manifest}")
        if args.validate_only:
            print("\n--validate-only: nothing written.")
            return
        os.makedirs(args.out_dir, exist_ok=True)
        for gid, recs in kept:
            with open(os.path.join(args.out_dir, f"ps_{gid}.jsonl"), "w",
                      encoding="utf-8", newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(kept)} files -> {args.out_dir}")
        return

    if args.census:
        census = json.load(open(args.census, encoding="utf-8"))
        users = sorted({h[p] for h in census.values() for p in ("P1", "P2")
                        if h.get(p) and h.get(f"{p}Title") != "BOT"})
    else:
        users = [u.strip() for u in args.users.split(",") if u.strip()]
    print(f"{len(users)} user(s) to query")

    seen, kept, stats = set(), [], collections.Counter()
    for i, user in enumerate(users, 1):
        pgn = fetch_user(user)
        time.sleep(1.5)
        n_user = 0
        for headers, movetext in split_games(pgn):
            if headers.get("Variant") != "Monster":
                continue
            gid = headers.get("Site", "").rsplit("/", 1)[-1]
            if not gid or gid in seen:
                continue
            seen.add(gid)
            n_user += 1

            is_bot = any(headers.get(f"{p}Title") == "BOT" for p in ("P1", "P2"))
            if is_bot and not args.include_bots:
                stats["skip: bot game"] += 1
                continue
            if headers.get("Result") == "1/2-1/2" and not args.include_draws:
                stats["skip: draw"] += 1
                continue
            if args.min_elo:
                elos = [int(headers.get(f"{p}Elo", 0) or 0) for p in ("P1", "P2")]
                if min(elos) < args.min_elo:
                    stats["skip: below --min-elo"] += 1
                    continue

            records, err = convert_game(headers, movetext,
                                        winner_only=not args.all_moves)
            if err:
                stats[f"FAIL: {err.split(':')[0]}"] += 1
                continue
            stats["converted"] += 1
            kept.append((gid, records))
        print(f"  [{i}/{len(users)}] {user:26s} {n_user:5d} games seen, "
              f"{stats['converted']:5d} converted so far", flush=True)

    if args.balance:
        kept = balance_by_position(kept, args.balance_seed)

    print("\n=== summary ===")
    for k, v in stats.most_common():
        print(f"  {k:34s} {v}")
    total_plies = sum(len(r) for _, r in kept)
    print(f"  distinct games seen                {len(seen)}")
    print(f"  usable games                       {len(kept)}")
    print(f"  positions (turn records)           {total_plies}")
    if stats["converted"]:
        rate = stats["converted"] / max(1, stats["converted"] + sum(
            v for k, v in stats.items() if k.startswith("FAIL")))
        print(f"  replay success rate                {rate:.1%}")

    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(sorted(gid for gid, _ in kept), f)
        print(f"manifest -> {args.manifest} ({len(kept)} ids)")

    if args.validate_only:
        print("\n--validate-only: nothing written.")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    for gid, records in kept:
        with open(os.path.join(args.out_dir, f"ps_{gid}.jsonl"), "w",
                  encoding="utf-8", newline="\n") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(kept)} files -> {args.out_dir}")


if __name__ == "__main__":
    main()
