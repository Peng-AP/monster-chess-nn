"""Does the value head know how CLOSE a won position is to being won?

REPORT.md section 22 measured the defect: across 171 plies of a bare-king
position, v21's value moved from -0.679 to -0.677. The head knows Black is
winning and cannot tell 171 plies from one, so MCTS has nothing to rank and
Black shuffles.

A gate cannot see this directly -- it only sees the win rate that results, at
20 games per colour. This measures the mechanism instead: over positions whose
true distance-to-capture is known, how well does predicted value track that
distance? A head with a progress signal should trend toward -1 as the capture
approaches. A flat one is the defect.

Ground truth comes from the finished_conversions corpus, where every game ends
in a real king capture and carries a real `plies_to_end`.

    py -3 tools/value_gradient_probe.py --model models/fresh_start_v21/best_value_net.pt
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chess  # noqa: E402
from evaluation import NNEvaluator  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402


def white_pawns(fen):
    board = chess.Board(fen)
    return sum(1 for _s, p in board.piece_map().items()
               if p.color == chess.WHITE and p.piece_type == chess.PAWN)


def spearman(xs, ys):
    """Rank correlation without a scipy dependency; ties get average ranks."""
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="data/raw/finished_conversions")
    ap.add_argument("--max-positions", type=int, default=4000)
    ap.add_argument("--bare-king-only", action="store_true", default=True,
                    help="restrict to positions where White has no pawns -- "
                         "the region where the shuffling actually happens")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, args.source, "*.jsonl"))):
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if not recs or recs[0].get("game_result") != -1:
            continue
        for r in recs:
            pte = r.get("plies_to_end")
            if pte is None:
                continue
            if args.bare_king_only and white_pawns(r["fen"]) != 0:
                continue
            rows.append((r["fen"], bool(r.get("half")), int(pte)))
    rows = rows[:args.max_positions]
    if len(rows) < 50:
        raise SystemExit(f"only {len(rows)} usable positions -- need the "
                         f"finished_conversions corpus in {args.source}")

    nn = NNEvaluator(os.path.join(ROOT, args.model)
                     if not os.path.isabs(args.model) else args.model)
    states = []
    for fen, half, _pte in rows:
        g = MonsterChessGame(fen)
        g.white_half_pending = half
        states.append(g)
    values = nn.batch_evaluate(states)          # White perspective
    ptes = [r[2] for r in rows]

    rho = spearman(ptes, values)
    print(f"model      : {args.model}")
    print(f"positions  : {len(rows)} (bare White king, known distance to capture)")
    print(f"\nSpearman(plies_to_end, value) = {rho:+.4f}")
    print("  a head with progress sense scores NEGATIVE here: fewer plies left")
    print("  means closer to -1. Near zero is the section-22 defect.")

    print(f"\n{'plies to capture':>18} {'n':>6} {'mean value':>12} {'sd':>8}")
    buckets = [(0, 5), (6, 15), (16, 30), (31, 60), (61, 120), (121, 10 ** 6)]
    means = []
    for lo, hi in buckets:
        sel = [v for v, p in zip(values, ptes) if lo <= p <= hi]
        if not sel:
            continue
        m = sum(sel) / len(sel)
        sd = (sum((x - m) ** 2 for x in sel) / len(sel)) ** 0.5
        means.append(m)
        label = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
        print(f"{label:>18} {len(sel):6d} {m:12.4f} {sd:8.4f}")
    if len(means) >= 2:
        span = max(means) - min(means)
        print(f"\nspread across buckets: {span:.4f}")
        print("  under ~0.10 means the head cannot tell a nearly-won position")
        print("  from one a hundred plies away -- nothing for MCTS to rank.")


if __name__ == "__main__":
    main()
