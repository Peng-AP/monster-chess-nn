"""Review a saved human game move by move and name the moves that cost it.

The owner reports (2026-08-17) that the chain models are past their play
strength in both colours, so a session no longer answers "is the model good".
What it can still answer is "where did THIS game turn", and nothing in the repo
did that: `tools/human_eval.py` reports game-level win/loss/draw by side, and
`play.ipynb` shows a live evaluation that is gone as soon as the move is made.

For each human move the reviewer searches the position twice -- before the move
and after it -- and reports the swing from the human's own perspective, plus
the move the engine would have played. A large negative swing is a move that
gave something away; the engine's preference is the concrete alternative.

Two caveats the output repeats, because both make a naive reading wrong:

  * A saved record stores a FEN, not the half-move flag. For a game where the
    human is BLACK every human record is a full Black turn and the FEN is
    complete. For a human playing WHITE it is not -- White's turn is two half
    moves and the FEN cannot say which half is pending -- so those games are
    reviewed at Black-turn boundaries only, and the tool says so.
  * The evaluations come from the model you pass, at the simulation count you
    pass. They are that model's opinion, not ground truth. Values from
    different models or different sim counts are not comparable.

    py -3 tools/review_game.py --game data/raw/human_games/black_2026_07/game_00032.jsonl \
        --model models/candidates/bootstrap_main_gen_0015/selected_epoch_016.pt --sims 800
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def load_records(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def describe_move(before_fen, after_fen):
    """Name the move that took `before_fen` to `after_fen`, or None.

    Squares are compared directly rather than trusting a move list: a saved
    game stores positions, and a promotion or a capture changes the piece on
    the destination, which a from/to pair alone would misreport.
    """
    import chess

    before = chess.Board(before_fen)
    after = chess.Board(after_fen)
    for move in before.legal_moves:
        trial = before.copy(stack=False)
        trial.push(move)
        if trial.board_fen() == after.board_fen():
            return before.san(move)
    # Monster Chess lets White make two moves per turn, so a White transition
    # is generally not a single legal move. Fall back to a square diff.
    changed = []
    for square in chess.SQUARES:
        if before.piece_at(square) != after.piece_at(square):
            changed.append(chess.square_name(square))
    return "(" + " ".join(changed) + ")" if changed else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True,
                    help="a saved .jsonl, or a directory to review the newest")
    ap.add_argument("--model", default=None,
                    help="evaluator; omit for the heuristic anchor")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--engine", default="native")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="swing that counts as a mistake worth listing")
    ap.add_argument("--report-path", default=None)
    args = ap.parse_args()

    path = args.game if os.path.isabs(args.game) else os.path.join(ROOT, args.game)
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "*.jsonl")),
                       key=os.path.getmtime)
        if not found:
            raise SystemExit(f"no .jsonl games under {path}")
        path = found[-1]
    records = load_records(path)
    if not records:
        raise SystemExit(f"{path} is empty")

    from benchmark import _build_engine
    from monster_chess import MonsterChessGame

    model = args.model
    if model and not os.path.isabs(model):
        model = os.path.join(ROOT, model)
    engine, label = _build_engine(model, args.sims, engine=args.engine)

    human_sides = {r.get("current_player") for r in records
                   if r.get("actor") == "human"}
    print(f"game    {os.path.relpath(path, ROOT)}")
    print(f"records {len(records)}  human side: {', '.join(sorted(human_sides))}")
    print(f"eval    {label} at {args.sims} sims")
    if "white" in human_sides:
        print("NOTE    the human played White; a stored FEN cannot record which "
              "half of a\n        White turn is pending, so White moves are "
              "reviewed as whole turns.")
    print()

    def search(fen):
        """Return (preferred action, root value) -- value is side-to-move.

        One search yields both. Asking separately doubled the cost of every
        reviewed move for nothing.
        """
        game = MonsterChessGame(fen)
        action, _probs, val = engine.get_best_action(game, temperature=0.0)
        return action, val

    rows = []
    for index, record in enumerate(records):
        if record.get("actor") != "human" or index + 1 >= len(records):
            continue
        before_fen = record["fen"]
        after_fen = records[index + 1]["fen"]
        # Both values are read from the mover's own side, so a drop always
        # means "this got worse for me" regardless of colour.
        engine_move, before = search(before_fen)
        _reply, after = search(after_fen)
        after = -after                 # after the move the opponent is to move
        rows.append({
            "record": index,
            "side": record.get("current_player"),
            "move": describe_move(before_fen, after_fen),
            "engine_move": str(engine_move),
            "before": round(float(before), 4),
            "after": round(float(after), 4),
            "swing": round(float(after - before), 4),
            "fen": before_fen,
        })
        print(f"  [{len(rows)}] rec {index:>3}  {record.get('current_player'):<5} "
              f"{str(rows[-1]['move']):<10} {before:+.3f} -> {after:+.3f}  "
              f"{after - before:+.3f}", flush=True)

    print()
    worst = sorted(rows, key=lambda r: r["swing"])
    mistakes = [r for r in worst if r["swing"] <= -args.threshold]
    if not mistakes:
        print(f"no move cost more than {args.threshold:.2f} by this model's "
              f"reckoning.")
    else:
        print(f"moves that cost {args.threshold:.2f} or more, worst first:")
        for row in mistakes:
            print(f"  rec {row['record']:>3}  played {row['move']}  "
                  f"({row['swing']:+.3f})  engine preferred {row['engine_move']}")
            print(f"        {row['fen']}")

    doc = {"game": os.path.relpath(path, ROOT).replace("\\", "/"),
           "model": args.model, "sims": args.sims, "engine": label,
           "threshold": args.threshold, "moves": rows}
    out = args.report_path
    if out:
        out = out if os.path.isabs(out) else os.path.join(ROOT, out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=1)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
