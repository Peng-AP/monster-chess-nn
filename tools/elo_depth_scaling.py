"""Search-depth scaling for one model, matched game-for-game to v29's ladder.

Plays MODEL at 800, 3,200, 6,400 and 12,800 simulations against the round
robin's 16 rated models with `tools/elo_ladder.py`'s two-stage design. Each
depth reuses the opening seeds v29's ladder used at that depth (v29@800 was
setting 0, v29@6400 setting 7, v29@12800 setting 8), so the two scaling
curves differ only in the model. 3,200 has no v29 ladder row (v29 played the
round robin there) and takes the next free block, 9.

Question (owner, 2026-10-01): does gen52 Arm L, the 2x wider network, keep
gaining from search beyond 3,200 where v29 flattens? If it does, its deep
searches are worth distilling, and it is the teacher candidate for gen53.

    py -3 tools/elo_depth_scaling.py --name gen52L --model models/candidates/bootstrap_main_gen_0052_large/arena_selected.pt
"""
import argparse
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import elo_ladder as el  # noqa: E402
import elo_tournament as et  # noqa: E402

DEPTHS = [(800, 0), (6400, 7), (12800, 8), (3200, 9)]   # (simulations, v29 ladder seed block)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--depths", default=",".join(str(s) for s, _ in DEPTHS),
                    help="comma-separated subset of 800,6400,12800,3200 (each keeps its own seed block)")
    args = ap.parse_args()
    if not os.path.exists(os.path.join(ROOT, args.model)):
        raise SystemExit(f"missing model {args.model}")
    wanted = {int(s) for s in args.depths.split(",")}
    depths = [(s, b) for s, b in DEPTHS if s in wanted]
    ladder = [(f"{args.name}@{sims}", args.model, sims) for sims, _ in depths]
    blocks = [block for _, block in depths]
    original = el.seed_for
    el.seed_for = lambda cand, opp, stage: original(blocks[cand], opp, stage)
    el.play(SimpleNamespace(workers=args.workers, stage1=el.STAGE1_GAMES, stage2=el.STAGE2_GAMES), args.out, ladder)
    rep = el.write_ratings(args.out)
    et.print_table(rep)


if __name__ == "__main__":
    main()
