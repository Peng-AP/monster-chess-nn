"""E1 exit gate, part 1: replay recorded games move-for-move against the engine.

For every recorded game, walk its stored positions and ask the engine to
reproduce each transition: from `records[i].fen`, does some action the engine
calls legal land exactly on `records[i+1].fen`? A game that cannot be replayed
means the engine and the record disagree about the rules.

Run against the **Python** engine this is an integrity check on the corpus and
on the harness itself. Run against the native core (`--engine native`, once it
builds) it is the parity gate the port must clear — which is why the comparison
is written against stored FENs rather than against the Python engine's live
behaviour: a bug shared by both engines would cancel out, a bug against the
record cannot.

FEN comparison uses the first four fields (placement, side to move, castling,
ep). The halfmove and fullmove counters are deliberately excluded: Monster
Chess forces `board.turn` back to WHITE between White's two half-moves, so the
counters advance on a cadence that is an artefact of the decomposition rather
than of the rules.

    py -3 tools/replay_parity.py --source data/raw/ps_monster --workers 6
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chess  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402


def complete_half_actions(game):
    """Every half-move the rules permit, with the search shortcut removed.

    `get_search_actions` truncates to a single action when a king capture is
    available. That is a search optimisation, not a rules claim — and a human
    who overlooked a king capture produces a recorded game the truncated
    generator cannot replay. Classifying those as rules divergences would
    inflate the failure count with games the engine understands fine.
    """
    if not game.is_white_turn:
        return game._get_black_actions(truncate_wins=False)
    game.board.turn = chess.WHITE
    return list(game.board.pseudo_legal_moves)


def fen_key(fen):
    """Placement + turn + castling + ep. See module docstring for the omission."""
    return " ".join(fen.split()[:4])


def is_half_ply_format(records):
    """Generated corpora record every half-move; owner games record turns.

    `ps_monster` and generated games carry `half` 0/1 on White records and one
    record per Black move. The owner's games carry only White-to-move positions
    with an atomic "m1,m2" policy, so consecutive FENs differ by White's whole
    double move *and* Black's reply. Replaying the second format with the first
    format's walk reports every game as a rules divergence — which is what it
    did before this split existed.
    """
    return any(r.get("half") in (0, 1) for r in records)


def replay_turn_format(records):
    """Walk White-turn snapshots: one atomic White pair plus one Black reply.

    Note this can only check parity at turn boundaries, because the
    intermediate positions were never recorded. E1's gate says "FEN parity at
    every ply"; for owner games every ply does not exist on disk.
    """
    game = MonsterChessGame(records[0]["fen"])
    for i in range(len(records) - 1):
        want = fen_key(records[i + 1]["fen"])
        if fen_key(game.fen()) != fen_key(records[i]["fen"]):
            return False, {"index": i, "reason": "desynchronised before the turn",
                           "engine_fen": game.fen(), "record_fen": records[i]["fen"]}
        found = None
        for white_action in game.get_legal_actions():
            after_white = game.clone()
            try:
                after_white.apply_action(white_action)
            except Exception:
                continue
            if after_white.board.king(chess.BLACK) is None:
                continue  # game would have ended here
            if fen_key(after_white.fen()) == want:
                found = after_white  # Black had no reply to record
                break
            for black_move in after_white.get_legal_actions():
                after_black = after_white.clone()
                try:
                    after_black.apply_action(black_move)
                except Exception:
                    continue
                if fen_key(after_black.fen()) == want:
                    found = after_black
                    break
            if found is not None:
                break
        if found is None:
            return False, {"index": i, "reason": "no White pair + Black reply reaches the next turn",
                           "from_fen": records[i]["fen"], "to_fen": records[i + 1]["fen"]}
        game = found
    return True, None


def replay_game(records):
    """Return (ok, detail). `detail` describes the first divergence, if any."""
    if len(records) < 2:
        return True, None
    if not is_half_ply_format(records):
        return replay_turn_format(records)
    game = MonsterChessGame(records[0]["fen"])
    for i in range(len(records) - 1):
        want = fen_key(records[i + 1]["fen"])
        if fen_key(game.fen()) != fen_key(records[i]["fen"]):
            return False, {
                "index": i,
                "reason": "desynchronised before the move",
                "engine_fen": game.fen(),
                "record_fen": records[i]["fen"],
            }
        def find(actions):
            for action in actions:
                probe = game.clone()
                try:
                    probe.apply_search_action(action)
                except Exception:
                    continue
                if fen_key(probe.fen()) == want:
                    return action
            return None

        offered = game.get_search_actions()
        matched = find(offered)
        classification = None
        if matched is None:
            # Was it merely hidden by the winning-capture shortcut?
            matched = find(complete_half_actions(game))
            classification = "declined_available_king_capture" if matched else None
        if matched is None:
            return False, {
                "index": i,
                "reason": "no legal action reproduces the next position",
                "from_fen": records[i]["fen"],
                "to_fen": records[i + 1]["fen"],
                "offered": len(offered),
            }
        if classification:
            return True, {"index": i, "reason": classification}
        game.apply_search_action(matched)
    return True, None


def _replay_file(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except Exception as exc:
        return {"path": path, "ok": False, "detail": {"reason": f"unreadable: {exc}"}}
    if not records:
        return {"path": path, "ok": True, "detail": None, "plies": 0}
    ok, detail = replay_game(records)
    return {"path": path, "ok": ok, "detail": detail, "plies": len(records)}


def iter_files(sources):
    for source in sources:
        for dirpath, _dirs, names in os.walk(source):
            for name in sorted(names):
                if name.endswith(".jsonl"):
                    yield os.path.join(dirpath, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--label", default="replay")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    sources = args.source or [
        os.path.join(ROOT, "data", "raw", "ps_monster"),
        os.path.join(ROOT, "data", "raw", "human_games"),
    ]
    files = list(iter_files(sources))
    if args.max_games:
        files = files[: args.max_games]
    print(f"replaying {len(files)} games from {len(sources)} source(s)", flush=True)

    started = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_replay_file, files)
    else:
        results = [_replay_file(f) for f in files]
    elapsed = time.time() - started

    failures = [r for r in results if not r["ok"]]
    declined = [r for r in results
                if r["ok"] and r.get("detail", {}) and
                r["detail"].get("reason") == "declined_available_king_capture"]
    summary = {
        "label": args.label,
        "sources": sources,
        "games": len(results),
        "plies": sum(r.get("plies", 0) for r in results),
        "replayed_ok": len(results) - len(failures),
        "diverged": len(failures),
        "stopped_at_declined_king_capture": len(declined),
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "failures": [
            {"path": os.path.relpath(f["path"], ROOT), "detail": f["detail"]}
            for f in failures[:50]
        ],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"replay_parity_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if failures:
        print(f"\nfirst {min(5, len(failures))} divergences:")
        for f in failures[:5]:
            print(" ", os.path.relpath(f["path"], ROOT), "->", f["detail"])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
