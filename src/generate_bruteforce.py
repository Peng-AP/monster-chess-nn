"""Rollout / heuristic MCTS calibration and data generation for Monster Chess.

Tests whether heuristic-guided or rollout-based MCTS can produce Black-winning
games from curriculum starting positions — the key diagnostic for the plan.

Two evaluation modes:
  heuristic : evaluate() at MCTS leaf nodes.  Fast and Monster-Chess-aware
              (barrier detection, pawn danger, king exposure already encoded).
  rollout   : random playout to terminal at leaf nodes.  Slower but unbiased.
              Does not call the neural network.

Usage (full calibration matrix — takes 1-4 hours):
  py -3 src/generate_bruteforce.py --calibrate --games 12

Usage (single cell, for quick checks):
  py -3 src/generate_bruteforce.py --tiers 4 5 --sims 1000 --mode heuristic --games 20

Usage (save training data after calibration confirms viability):
  py -3 src/generate_bruteforce.py --tiers 4 5 6 --sims 1000 --mode heuristic \\
      --games 100 --output-dir data/raw/calibration_heuristic_t456_s1000
"""

import argparse
import json
import os
import random
import sys
import time

import chess  # noqa: F401  (needed by MonsterChessGame)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import CURRICULUM_FENS, CURRICULUM_TIER_BOUNDARIES
from evaluation import evaluate
from monster_chess import MonsterChessGame
from mcts import MCTS


# ──────────────────────────────────────────────────────────────────────────────
# Rollout leaf evaluator
# ──────────────────────────────────────────────────────────────────────────────

class RolloutEvaluator:
    """Evaluate a position by playing a fully random game to terminal.

    Passed as eval_fn to MCTS.  Because it has no batch_evaluate attribute,
    MCTS automatically dispatches to _run_sequential() (UCB1 tree search).
    """

    def __call__(self, game_state):
        game = game_state.clone()
        moves = 0
        cap = 300  # safety cap (2 × MAX_GAME_TURNS)
        while not game.is_terminal() and moves < cap:
            actions = game.get_legal_actions()
            if not actions:
                break
            game.apply_action(random.choice(actions))
            moves += 1
        return game.get_result()


# ──────────────────────────────────────────────────────────────────────────────
# Tier FEN helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tier_ranges():
    """Return list of (start, end) index pairs for each tier (0-indexed)."""
    bounds = [0] + list(CURRICULUM_TIER_BOUNDARIES) + [len(CURRICULUM_FENS)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def get_tier_fens(tiers):
    """Return FENs for the specified tier numbers (1-indexed)."""
    ranges = _tier_ranges()
    n_tiers = len(ranges)
    result = []
    for t in tiers:
        if t < 1 or t > n_tiers:
            raise ValueError(f"Tier {t} out of range (1–{n_tiers})")
        s, e = ranges[t - 1]
        result.extend(CURRICULUM_FENS[s:e])
    return result


def describe_tiers():
    """Print a summary of available tiers."""
    ranges = _tier_ranges()
    descriptions = [
        "Tier 1: Forced capture (king in corner, net complete, no pawns)",
        "Tier 2: One move from forced capture (no pawns)",
        "Tier 3: Isolation — king confined to edge strip (with pawns)",
        "Tier 4: Overwhelming material, king near edge (Q+2R/4R vs king+pawns)",
        "Tier 5: Mid-game Black advantage (promoted pieces, White has pawns)",
        "Tier 6: Realistic mid-game — 1-2 pawns eliminated, king advanced",
        "Tier 7: Opening positions from human games (moves 2-7, White to move)",
    ]
    print("\nAvailable curriculum tiers:")
    for i, (s, e) in enumerate(ranges, 1):
        desc = descriptions[i - 1] if i <= len(descriptions) else f"Tier {i}"
        print(f"  Tier {i} ({e - s} positions): {desc}")


# ──────────────────────────────────────────────────────────────────────────────
# Game play
# ──────────────────────────────────────────────────────────────────────────────

def play_one_game(start_fen, mcts_obj, temperature_moves=15):
    """Play one complete game from start_fen using mcts_obj.

    Returns:
        records    : list of position dicts (fen, mcts_value, policy,
                     current_player, game_result)
        game_result: float (-1 Black win, 0 draw, +1 White win, -0.5 Black adv)
        n_moves    : total half-moves played
    """
    game = MonsterChessGame(fen=start_fen)
    records = []
    move_num = 0

    while not game.is_terminal():
        temperature = 1.0 if move_num < temperature_moves else 0.1
        action, policy, root_value = mcts_obj.get_best_action(
            game, temperature=temperature
        )
        if action is None:
            break

        records.append({
            "fen": game.board.fen(),
            "mcts_value": round(float(root_value), 4),
            "policy": {k: round(v, 4) for k, v in policy.items()},
            "current_player": "white" if game.is_white_turn else "black",
        })

        game.apply_action(action)
        move_num += 1

    result = game.get_result()
    for rec in records:
        rec["game_result"] = result

    return records, result, move_num


# ──────────────────────────────────────────────────────────────────────────────
# Single calibration cell
# ──────────────────────────────────────────────────────────────────────────────

def run_cell(tiers, sims, mode, num_games, seed=42, verbose=True, output_dir=None):
    """Run one cell of the calibration matrix.

    Returns a stats dict with win rates, game lengths, timing.
    """
    fens = get_tier_fens(tiers)
    if not fens:
        return {"error": "no FENs for requested tiers", "tiers": tiers}

    if mode == "heuristic":
        eval_fn = evaluate
    elif mode == "rollout":
        eval_fn = RolloutEvaluator()
    else:
        raise ValueError(f"Unknown mode '{mode}' — use 'heuristic' or 'rollout'")

    mcts_obj = MCTS(num_simulations=sims, eval_fn=eval_fn)

    results_list = []
    lengths = []
    times_list = []
    all_records = []

    for i in range(num_games):
        fen = fens[i % len(fens)]
        random.seed(seed + i)

        t0 = time.time()
        records, result, n_moves = play_one_game(fen, mcts_obj)
        elapsed = time.time() - t0

        results_list.append(result)
        lengths.append(n_moves)
        times_list.append(elapsed)
        all_records.append(records)

        if verbose:
            winner = "White" if result > 0 else ("Black" if result < 0 else "Draw")
            print(f"    [{i+1:3d}/{num_games}] {winner:5s} | {n_moves:3d} moves | {elapsed:.1f}s")

    # Optionally save JSONL
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tier_str = "".join(str(t) for t in tiers)
        fname = f"games_t{tier_str}_s{sims}_{mode}.jsonl"
        out_path = os.path.join(output_dir, fname)
        n_pos = 0
        with open(out_path, "w") as f:
            for game_records in all_records:
                for rec in game_records:
                    f.write(json.dumps(rec) + "\n")
                    n_pos += 1
        print(f"    Saved {n_pos} positions to {out_path}")

    n = len(results_list)
    n_black = sum(1 for r in results_list if r < 0)
    n_white = sum(1 for r in results_list if r > 0)
    n_draw  = n - n_black - n_white

    return {
        "tiers":     tiers,
        "sims":      sims,
        "mode":      mode,
        "n":         n,
        "black":     n_black,
        "white":     n_white,
        "draw":      n_draw,
        "black_pct": n_black / n if n else 0.0,
        "avg_len":   sum(lengths) / n if n else 0.0,
        "avg_sec":   sum(times_list) / n if n else 0.0,
        "total_sec": sum(times_list),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Calibration matrix
# ──────────────────────────────────────────────────────────────────────────────

# Rows: (tiers, sims, modes_to_test)
# Rationale:
#   Tier 4-5: overwhelming Black material — should be easiest wins
#   Tier 6:   mid-game advantage — harder, key practical scenario
#   Tier 7:   opening (White to move) — expected hard; heuristic only to check floor
_MATRIX = [
    ([4, 5],  200, ["heuristic", "rollout"]),
    ([4, 5],  500, ["heuristic", "rollout"]),
    ([4, 5], 1000, ["heuristic", "rollout"]),
    ([6],     200, ["heuristic", "rollout"]),
    ([6],     500, ["heuristic", "rollout"]),
    ([6],    1000, ["heuristic", "rollout"]),
    ([7],     200, ["heuristic"]),
    ([7],     500, ["heuristic"]),
    ([7],    1000, ["heuristic"]),
]


def run_calibration(games_per_cell, seed):
    """Run the full calibration matrix and print a summary table."""
    all_stats = []
    t_start = time.time()

    for tiers, sims, modes in _MATRIX:
        for mode in modes:
            print(f"\n{'─' * 64}")
            print(f"  tiers={tiers}  sims={sims}  mode={mode}  games={games_per_cell}")
            print(f"{'─' * 64}")
            stats = run_cell(tiers, sims, mode, games_per_cell, seed=seed, verbose=True)
            all_stats.append(stats)
            print(f"  → Black {stats['black_pct']:.1%} "
                  f"({stats['black']}/{stats['n']})  "
                  f"avg_len={stats['avg_len']:.0f}  "
                  f"avg_sec={stats['avg_sec']:.1f}s")

    total_elapsed = time.time() - t_start

    _print_summary(all_stats, total_elapsed)
    _print_interpretation(all_stats)
    return all_stats


def _print_summary(all_stats, total_elapsed):
    w = 82
    print(f"\n\n{'═' * w}")
    print("  CALIBRATION MATRIX RESULTS")
    print(f"{'═' * w}")
    hdr = (f"  {'Tiers':<10} {'Sims':>5} {'Mode':>10}  "
           f"{'Black%':>7}  {'W/B/D':>11}  {'AvgLen':>7}  {'AvgSec':>7}")
    print(hdr)
    print(f"  {'─' * (w - 2)}")
    for s in all_stats:
        wbd = f"{s['white']}/{s['black']}/{s['draw']}"
        flag = "  ← ✓ viable" if s['black_pct'] >= 0.15 else ""
        print(
            f"  {str(s['tiers']):<10} {s['sims']:>5} {s['mode']:>10}  "
            f"{s['black_pct']:>6.1%}  {wbd:>11}  "
            f"{s['avg_len']:>7.0f}  {s['avg_sec']:>7.1f}s{flag}"
        )
    print(f"  {'─' * (w - 2)}")
    print(f"  Total time: {total_elapsed / 60:.1f} minutes")
    print(f"{'═' * w}")


def _print_interpretation(all_stats):
    """Print decision guidance based on results."""
    print("\n  INTERPRETATION:")

    def get(tiers_list, sims, mode):
        for s in all_stats:
            if sorted(s["tiers"]) == sorted(tiers_list) and s["sims"] == sims and s["mode"] == mode:
                return s
        return None

    # Key cells for decision
    t45_h1k = get([4, 5], 1000, "heuristic")
    t45_r500 = get([4, 5], 500, "rollout")
    t6_h1k  = get([6], 1000, "heuristic")
    t7_h1k  = get([7], 1000, "heuristic")

    print()
    print("  Tier 4-5 heuristic@1000:")
    if t45_h1k:
        bp = t45_h1k["black_pct"]
        if bp >= 0.30:
            print(f"    {bp:.1%} Black wins — EXCELLENT. Proceed to Tier 6 production generation.")
        elif bp >= 0.15:
            print(f"    {bp:.1%} Black wins — VIABLE. Generate 200 games from Tier 4-5 for training.")
        else:
            print(f"    {bp:.1%} Black wins — MARGINAL. Evaluation heuristic may not be strong enough.")
            print("    Consider increasing sims to 2000, or investigate evaluation.py weights.")

    print()
    print("  Tier 4-5 rollout@500 vs heuristic@500:")
    t45_h500 = get([4, 5], 500, "heuristic")
    if t45_r500 and t45_h500:
        print(f"    heuristic: {t45_h500['black_pct']:.1%}   rollout: {t45_r500['black_pct']:.1%}")
        if t45_h500["black_pct"] > t45_r500["black_pct"]:
            print("    Heuristic beats rollout → evaluation.py is a better guide than random play.")
            print("    Recommend: use heuristic mode for production data generation.")
        else:
            print("    Rollout ≥ heuristic → random play is sufficient from Tier 4-5.")

    print()
    print("  Tier 6 heuristic@1000:")
    if t6_h1k:
        bp = t6_h1k["black_pct"]
        if bp >= 0.15:
            print(f"    {bp:.1%} Black wins — Tier 6 viable. Include in production generation.")
        else:
            print(f"    {bp:.1%} Black wins — Tier 6 not producing enough wins even at 1000 sims.")
            print("    Production generation should focus on Tier 4-5 only.")

    print()
    print("  Tier 7 heuristic@1000 (opening baseline):")
    if t7_h1k:
        bp = t7_h1k["black_pct"]
        print(f"    {bp:.1%} Black wins — this is the opening scenario, expected to be low.")
        if bp >= 0.10:
            print("    Model shows some opening competence. Worth including in next training run.")
        else:
            print("    Negligible opening wins — opening positions need curriculum approach first.")

    print()
    print("  NEXT STEPS:")
    if t45_h1k and t45_h1k["black_pct"] >= 0.15:
        print("  [1] Generate 200 games from Tier 4-6 with heuristic mode at 1000 sims")
        print("  [2] Tag as source=bruteforce, save to data/raw/bruteforce_*/")
        print("  [3] Add to next training consolidation (no alternating, Black-only training)")
    else:
        print("  [1] The heuristic is not strong enough at available sim counts.")
        print("  [2] Investigate evaluation.py weights — barrier detection may be under-weighted.")
        print("  [3] Consider running at 2000+ sims before proceeding to production generation.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rollout/heuristic MCTS calibration and data generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--calibrate", action="store_true",
                        help="Run the full calibration matrix")
    parser.add_argument("--tiers", type=int, nargs="+", default=[4, 5],
                        help="Curriculum tiers to sample starting positions from (1-indexed)")
    parser.add_argument("--sims", type=int, default=500,
                        help="MCTS simulations per move")
    parser.add_argument("--mode", choices=["heuristic", "rollout"], default="heuristic",
                        help="Leaf evaluation mode")
    parser.add_argument("--games", type=int, default=12,
                        help="Games per cell (calibrate) or total games (single run)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="If set, save JSONL records to this directory")
    parser.add_argument("--list-tiers", action="store_true",
                        help="Show available tiers and exit")
    args = parser.parse_args()

    if args.list_tiers:
        describe_tiers()
        return

    if args.calibrate:
        print(f"Starting calibration matrix — {args.games} games per cell, seed={args.seed}")
        print("This will test heuristic and rollout modes across Tiers 4-5, 6, 7")
        print("at sim counts 200 / 500 / 1000.\n")
        run_calibration(games_per_cell=args.games, seed=args.seed)
    else:
        fens = get_tier_fens(args.tiers)
        print(f"Single-cell run: tiers={args.tiers}  sims={args.sims}  "
              f"mode={args.mode}  games={args.games}")
        print(f"Starting positions: {len(fens)} FENs across {len(args.tiers)} tier(s)")
        stats = run_cell(
            tiers=args.tiers,
            sims=args.sims,
            mode=args.mode,
            num_games=args.games,
            seed=args.seed,
            verbose=True,
            output_dir=args.output_dir,
        )
        print(f"\nResult: Black {stats['black_pct']:.1%} ({stats['black']}/{stats['n']})  "
              f"avg_len={stats['avg_len']:.0f}  avg_sec={stats['avg_sec']:.1f}s  "
              f"total={stats['total_sec']/60:.1f} min")


if __name__ == "__main__":
    main()
