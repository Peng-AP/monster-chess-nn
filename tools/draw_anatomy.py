"""Why does one side's games end drawn? Split them by cause and by GAME LENGTH.

A match report counts draws but never says what kind they were. Against v22 the
chain's Black cut its draw rate 30.0% -> 13.3% over six generations while
White's stayed at 35-43%, so "is White's ceiling strength or conversion?" is
the question that decides where work goes, and no artifact in `benchmarks/`
could answer it.

**Material is not the answer, and this tool used to claim it was.** The first
version split draws by White's remaining pawns, on the reasoning that White is
a king plus four pawns and a pawnless White must be structurally drawn. The
data refuted it flatly (150 games, gen16 vs v22, 2026-08-17):

    score from pawnless positions   0.6141  (n=92)
    score with pawns remaining      0.6293  (n=58)
    of 62 White wins, 34 ended with ZERO pawns

`evaluation.py` said so all along -- "a lone king can still hunt and capture
Black's king. White without pawns is NOT lost" -- and the double-moving king
really does hunt. The training signal supposedly missing is present too:
`WHITE_PAWN_VALUE` is 0.18, *larger* than `PAWN_ELIMINATION_BONUS` at 0.14, and
a lost pawn costs White 0.32 across the two terms.

**Time is what separates the outcomes.** On the same 150 games:

    win    n=62  mean  30.0 plies   90% of wins land by ply 50, 95% by ply 60
    draw   n=62  mean  82.4 plies   median 82, every one a repetition
    loss   n=26  mean  71.7 plies   none earlier than ply 46

White either captures the king early or never does; a game still alive past
ply 60 is a draw or a loss with near-certainty, and the material on both sides
is the same in all three buckets. So the reported split is outcome x length,
and material is printed as description only -- never as a verdict.

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

    # Outcome x length: the axis the outcomes actually separate on.
    def stats(rows):
        if not rows:
            return None
        lengths = sorted(r["plies"] for r in rows)
        return {"n": len(rows),
                "mean_plies": round(sum(lengths) / len(lengths), 1),
                "median_plies": lengths[len(lengths) // 2],
                "min_plies": lengths[0], "max_plies": lengths[-1],
                "mean_white_pawns": round(
                    sum(r["white_pawns"] for r in rows) / len(rows), 2),
                "mean_black_pieces": round(
                    sum(r["black_pieces"] for r in rows) / len(rows), 2)}

    buckets = {"win": stats(wins), "draw": stats(draws), "loss": stats(losses)}
    print("\noutcome by game length:")
    print("   %-6s %4s %9s %8s %8s %10s %8s"
          % ("", "n", "mean", "median", "range", "wht pawns", "blk pcs"))
    for label in ("win", "draw", "loss"):
        s = buckets[label]
        if not s:
            continue
        print("   %-6s %4d %9.1f %8d %4d-%-4d %10.2f %8.2f"
              % (label, s["n"], s["mean_plies"], s["median_plies"],
                 s["min_plies"], s["max_plies"], s["mean_white_pawns"],
                 s["mean_black_pieces"]))

    # How long the winning window stays open. A game past the last cutoff with
    # no result is, empirically, no longer winnable.
    horizon = {}
    if wins:
        win_lengths = sorted(r["plies"] for r in wins)
        print("\nwins landing by ply:")
        for cut in (20, 30, 40, 50, 60, 80, 120):
            landed = sum(1 for p in win_lengths if p <= cut)
            horizon[cut] = landed
            print("   by %3d: %3d/%3d (%.0f%%)"
                  % (cut, landed, len(win_lengths), 100 * landed / len(win_lengths)))

    # Material, reported because it is cheap to record -- NOT as a verdict.
    # Splitting draws on pawn count is what this tool got wrong on 2026-08-17;
    # the two scores below are the refutation, printed every run so the mistake
    # cannot quietly return.
    def score_of(rows):
        if not rows:
            return None
        return sum(1 if r["result"] > 0 else 0.5 if r["result"] == 0 else 0
                   for r in rows) / len(rows)

    pawnless = [g for g in games if g["white_pawns"] == 0]
    with_pawns = [g for g in games if g["white_pawns"] > 0]
    print("\nmaterial does not predict the result (descriptive only):")
    print("   White ended pawnless   n=%-4d score %s"
          % (len(pawnless), f"{score_of(pawnless):.4f}" if pawnless else "-"))
    print("   White kept a pawn      n=%-4d score %s"
          % (len(with_pawns), f"{score_of(with_pawns):.4f}" if with_pawns else "-"))
    if wins:
        bare_wins = sum(1 for r in wins if r["white_pawns"] == 0)
        print("   of %d wins, %d ended with ZERO pawns -- the double-moving "
              "king hunts" % (len(wins), bare_wins))

    doc = {"white": args.white, "black": args.black, "games": total,
           "sims": args.sims, "book": args.book,
           "book_offset": args.book_offset,
           "score": (len(wins) + 0.5 * len(draws)) / total,
           "wins": len(wins), "losses": len(losses), "draws": len(draws),
           "draw_reasons": dict(reasons),
           "by_outcome": buckets,
           "wins_landing_by_ply": horizon,
           "score_when_pawnless": score_of(pawnless),
           "score_when_pawns_remain": score_of(with_pawns),
           "games_detail": games}
    if args.report_path:
        out = resolve(args.report_path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=1)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
