"""E2 exit gate: tensor byte-equality vs `encoding.fen_to_tensor`.

Byte-equality, not tolerance. The tensor is the network's entire view of a
position, so a single differing element is a different input — and because the
values are simple ratios narrowed to f32, exact equality is achievable and
anything less means a genuine bug.

Both layouts are checked. The 15-channel legacy encoding is not dead weight:
v16/v17 checkpoints still load with it for matches and owner play, and
`fen_to_tensor` selects by channel count precisely so an incompatible encoding
cannot load silently.

    py -3 tools/encoding_parity.py --positions 100000
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "native"))

import monster_native as mn  # noqa: E402
from encoding import fen_to_tensor  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def states(limit, seed):
    rng = random.Random(seed)
    seen = 0
    while seen < limit:
        game = MonsterChessGame(START_FEN)
        for _ in range(200):
            if game.is_terminal() or seen >= limit:
                break
            yield game
            seen += 1
            actions = game.get_search_actions()
            if not actions:
                break
            game.apply_search_action(rng.choice(actions))


def corpus_records(limit):
    count = 0
    for source in ("ps_monster", "human_games"):
        base = os.path.join(ROOT, "data", "raw", source)
        for dirpath, _dirs, names in os.walk(base):
            for name in sorted(names):
                if not name.endswith(".jsonl"):
                    continue
                with open(os.path.join(dirpath, name), "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            game = MonsterChessGame(rec["fen"])
                        except Exception:
                            continue
                        game.white_half_pending = rec.get("half") == 1
                        yield game
                        count += 1
                        if count >= limit:
                            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    per_layout = {}
    started = time.time()
    for channels in (15, 17):
        checked = unequal = 0
        worst = 0.0
        example = None
        half = args.positions // 2
        for source in (states(half, args.seed), corpus_records(args.positions - half)):
            for game in source:
                fen = game.fen()
                want = fen_to_tensor(fen, game.is_white_turn,
                                     game.white_half_pending,
                                     input_channels=channels)
                got = np.asarray(
                    mn.encode_fen(fen, game.is_white_turn, game.white_half_pending,
                                  channels),
                    dtype=np.float32).reshape(8, 8, channels)
                checked += 1
                if not np.array_equal(want, got):
                    unequal += 1
                    delta = float(np.abs(want - got).max())
                    if delta > worst:
                        worst = delta
                        example = {"fen": fen, "channels": channels,
                                   "max_abs_delta": delta}
        per_layout[str(channels)] = {
            "positions_checked": checked,
            "byte_unequal": unequal,
            "worst_abs_delta": worst,
            "example": example,
        }
    elapsed = time.time() - started

    summary = {
        "layouts": per_layout,
        "total_unequal": sum(v["byte_unequal"] for v in per_layout.values()),
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"encoding_parity_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
