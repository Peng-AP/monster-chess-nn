"""Why does one side's games end drawn? Split the draws by cause and material.

A match report counts draws but never says what kind they were, and the two
kinds mean opposite things. A **repetition** draw is a position the engine
chose to repeat; a **cap** draw ran out of plies; a **structural** draw is one
neither side could ever have won. Against v22 the chain's Black cut its draw
rate 30.0% -> 13.3% over six generations while White's stayed put at 35-43%, so
the question "is White's ceiling strength or conversion?" is now the one that
decides where work goes -- and no artifact in `benchmarks/` can answer it.

Material is what separates the cases for White. White is a king plus four
pawns and wins only by capturing the black king; a White that has lost every
pawn is a lone double-moving king against a full army and is not drawing a won
game, it is drawing a drawn one. So the split that matters is:

    drawn, White still has pawns   -> conversion failure, worth attacking
    drawn, White is a bare king    -> structural, no technique recovers it

Games are played exactly as `benchmark.play_one` plays them -- same repetition
rule, same ply cap, same book start state including the half-move flag and
turn_count -- so the draw rate here is comparable to a match report's.

    py -3 tools/draw_anatomy.py --white models/candidates/.../selected_epoch_010.pt \
        --black models/fresh_start_v22/best_value_net.pt \
        --book books/gate_v26_mixed_p8_20260817.json --book-offset 2820 --games 150
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

_engines = {}


def _init(white_model, black_model, sims, engine):
    from benchmark import _build_engine
    _engines["w"], _ = _build_engine(white_model, sims, engine=engine)
    _engines["b"], _ = _build_engine(black_model, sims, engine=engine)


def _material(board_fen):
    """(white pawns, black pieces) from the board half of a FEN."""
    placement = board_fen.split()[0]
    white_pawns = placement.count("P")
    black = sum(placement.count(c) for c in "pnbrqk")
    return white_pawns, black


def _play(task):
    """One game, returning the outcome plus why it ended."""
    from benchmark import _apply
    from monster_chess import MonsterChessGame
    from repetition import RepetitionTracker

    fen, half, turn_count, max_plies = task
    game = MonsterChessGame(fen=fen) if fen else MonsterChessGame()
    if fen:
        game.white_half_pending = bool(half)
        game.turn_count = int(turn_count)
    repetition = RepetitionTracker()
    repetition.record(game, 0)
    repeated = False
    plies = 0
    while not game.is_terminal() and plies < max_plies:
        engine = _engines["w"] if game.is_white_turn else _engines["b"]
        action, _probs, _val = engine.get_best_action(game, temperature=0.0)
        if action is None:
            break
        _apply(game, action)
        plies += 1
        if repetition.record(game, plies):
            repeated = True
            break
    outcome = repetition.draw_result if repeated else game.get_result()
    if repeated:
        reason = "repetition"
    elif game.is_terminal():
        reason = "terminal"
    elif plies >= max_plies:
        reason = "ply_cap"
    else:
        reason = "no_move"
    white_pawns, black_pieces = _material(game.fen())
    return {"result": float(outcome), "plies": plies, "reason": reason,
            "white_pawns": white_pawns, "black_pieces": black_pieces,
            "final_fen": game.fen()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--white", required=True, help="model playing White")
    ap.add_argument("--black", required=True, help="model playing Black")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--book", default=None)
    ap.add_argument("--book-offset", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=600)
    ap.add_argument("--report-path", default=None)
    args = ap.parse_args()

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(ROOT, path)

    tasks = []
    if args.book:
        with open(resolve(args.book), encoding="utf-8") as handle:
            entries = json.load(handle)["entries"]
        window = entries[args.book_offset:args.book_offset + args.games]
        if len(window) < args.games:
            raise SystemExit(
                f"book has {len(window)} entries from offset {args.book_offset}, "
                f"need {args.games}")
        tasks = [(e["fen"], e.get("half", False), e.get("turn_count", 0),
                  args.max_plies) for e in window]
    else:
        tasks = [(None, False, 0, args.max_plies)] * args.games

    print(f"{args.games} games, White={os.path.basename(os.path.dirname(args.white))} "
          f"Black={os.path.basename(os.path.dirname(args.black))} at {args.sims} sims",
          flush=True)
    pool = mp.Pool(args.workers, initializer=_init,
                   initargs=(resolve(args.white), resolve(args.black),
                             args.sims, args.engine))
    games = []
    try:
        for i, row in enumerate(pool.imap_unordered(_play, tasks), 1):
            games.append(row)
            if i % max(1, args.games // 10) == 0:
                print(f"  [{i}/{args.games}]", flush=True)
    finally:
        pool.terminate()
        pool.join()

    wins = [g for g in games if g["result"] > 0]
    losses = [g for g in games if g["result"] < 0]
    draws = [g for g in games if g["result"] == 0]
    total = len(games)
    print(f"\nWhite {len(wins)}W {len(losses)}L {len(draws)}D "
          f"-> score {(len(wins) + 0.5 * len(draws)) / total:.4f}")

    reasons = collections.Counter(g["reason"] for g in draws)
    print(f"\ndraws by cause ({len(draws)} of {total}, "
          f"{100 * len(draws) / total:.1f}%):")
    for reason, count in reasons.most_common():
        print(f"   {reason:<12} {count:>4}  ({100 * count / max(1, len(draws)):.1f}% of draws)")

    # The split that decides where work goes.
    live = [g for g in draws if g["white_pawns"] > 0]
    bare = [g for g in draws if g["white_pawns"] == 0]
    print(f"\ndraws by White material:")
    print(f"   White still has pawns  {len(live):>4}  "
          f"({100 * len(live) / max(1, len(draws)):.1f}% of draws) "
          f"-- conversion failures")
    print(f"   White is a bare king   {len(bare):>4}  "
          f"({100 * len(bare) / max(1, len(draws)):.1f}% of draws) "
          f"-- structural, unwinnable")
    if live:
        pawns = collections.Counter(g["white_pawns"] for g in live)
        print("   pawn count among the conversion failures: "
              + ", ".join(f"{k}p x{v}" for k, v in sorted(pawns.items())))
        black_left = sum(g["black_pieces"] for g in live) / len(live)
        print(f"   mean black pieces remaining in those: {black_left:.1f}")

    doc = {"white": args.white, "black": args.black, "games": total,
           "sims": args.sims, "book": args.book,
           "book_offset": args.book_offset,
           "score": (len(wins) + 0.5 * len(draws)) / total,
           "wins": len(wins), "losses": len(losses), "draws": len(draws),
           "draw_reasons": dict(reasons),
           "draws_white_has_pawns": len(live),
           "draws_white_bare_king": len(bare),
           "games_detail": games}
    if args.report_path:
        out = resolve(args.report_path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=1)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
