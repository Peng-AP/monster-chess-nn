"""Benchmark anchor: score a candidate against the heuristic MCTS (REWORK_PLAN.md §0.2).

The heuristic UCB1 search is the only search in the repo known to play sensibly, so it
serves as a permanent, fixed yardstick.  Every candidate model is measured against it —
not only against the moving incumbent — so progress is absolute, not relative.

Runs both colors, temperature 0, no Dirichlet noise, no data saved.  Writes one JSON per
run to ``benchmarks/``.

    py -3 src/benchmark.py                       # heuristic vs heuristic (sanity)
    py -3 src/benchmark.py --model models/best_value_net.pt --games 100 --sims 400
"""
import argparse
import json
import os
import time

from config import MODEL_DIR, PROJECT_ROOT
from monster_chess import MonsterChessGame
from mcts import MCTS
from evaluation import evaluate


def _legal(game):
    """Legal actions in whatever granularity the search uses (half-move aware)."""
    fn = getattr(game, "get_search_actions", None)
    return fn() if fn else game.get_legal_actions()


def _apply(game, action):
    """Apply an action produced by get_best_action, half-move aware."""
    fn = getattr(game, "apply_search_action", None)
    (fn or game.apply_action)(action)


ENGINE_ENV = "MONSTER_ENGINE"


def _engine_choice(engine=None):
    """`python` (default) or `native`.

    Explicit argument wins, then the MONSTER_ENGINE environment variable, then
    the default. The env var exists so tools without their own flag can be
    switched for a whole run; D5 keeps the default on `python` until the E5
    re-baseline, so nothing changes underneath existing measurements.
    """
    choice = (engine or os.environ.get(ENGINE_ENV) or "python").lower()
    if choice not in ("python", "native"):
        raise ValueError(f"unknown engine {choice!r} (expected python|native)")
    return choice


def _build_engine(model_path, sims, batch_size=None, engine=None):
    """Return (engine, label). NN engine if a model is given, else heuristic.

    batch_size is the MCTS leaf-parallel width. The default (16) was chosen for
    the generation/match case, where worker processes keep the GPU busy and
    in-tree batching only degrades selection quality. It is exposed here so the
    tradeoff can be measured rather than assumed: wider batches buy simulations
    per second (b=256 measured at 1.53x b=16 on one tree) and cost selection
    quality, and only a match settles which wins.
    """
    if model_path:
        from evaluation import NNEvaluator
        eval_fn = NNEvaluator(model_path)
        label = f"nn:{os.path.basename(model_path)}"
    else:
        eval_fn = evaluate
        label = "heuristic"
    kwargs = {} if batch_size is None else {"batch_size": int(batch_size)}
    choice = _engine_choice(engine)
    if choice == "native":
        from native_mcts import NativeMCTS
        search = NativeMCTS(num_simulations=sims, eval_fn=eval_fn,
                            root_noise=False, allow_early_stop=True, **kwargs)
        label = f"{label}|native"
    else:
        search = MCTS(num_simulations=sims, eval_fn=eval_fn, root_noise=False,
                      allow_early_stop=True, **kwargs)
    return search, label


def play_one(white_engine, black_engine, start_fen=None, max_plies=600,
             opening_temp_plies=0, opening_temp=0.5):
    """Play a single game. Returns (result, n_plies, n_decisions).

    opening_temp_plies > 0 samples the first N plies at opening_temp instead
    of temp 0. REQUIRED for NN-vs-NN matches (iterate arena, tools/match.py):
    two deterministic engines at temp 0 replay the identical game no matter
    how the RNG is seeded, silently collapsing the sample size to 1.
    Heuristic-involved games diverge via seeded tie-breaks, so the anchor
    benchmark keeps 0 (yardstick unchanged).
    """
    game = MonsterChessGame(fen=start_fen) if start_fen else MonsterChessGame()
    decisions = 0
    plies = 0
    while not game.is_terminal() and plies < max_plies:
        engine = white_engine if game.is_white_turn else black_engine
        temp = opening_temp if plies < opening_temp_plies else 0.0
        action, _probs, _val = engine.get_best_action(game, temperature=temp)
        if action is None:
            break
        _apply(game, action)
        decisions += 1
        plies += 1
    return game.get_result(), plies, decisions


def _mean(values):
    """Mean of a list, or None when empty (keeps JSON explicit about 'no data')."""
    return round(sum(values) / len(values), 2) if values else None


def summarize_side(plies_by_result):
    """Aggregate one side's games into a strength block.

    plies_by_result: list of (candidate_result, plies) where candidate_result
    is from the CANDIDATE's perspective (>0 candidate win, <0 loss, 0 draw)
    regardless of which color the candidate played. White and Black are
    summarized identically — same shape, same stats, no side special-cased.

    Returns wins/losses/draws, score = (W + 0.5 D) / N, mean plies overall,
    and mean plies split by won/lost games (how decisively games go, in
    either direction).

    **Only a king capture is a win** (owner, 2026-08-03: *"a win by time
    shouldn't be counted the same as win by capturing the king"*). A game that
    reaches `MAX_GAME_TURNS` is relabelled +-0.5 by heuristic sign, and that
    relabel used to score here as a full win: `result > 0` caught +0.5. So a
    model that reached a winning position and then shuffled to the cap earned
    the same gate credit as one that finished, which is exactly the behaviour
    the owner reported at the board. Measured on `v20w`: its post-promotion
    "conversion" rose 0.28 -> 0.46 against v19 entirely through +-0.5 relabels
    (13 -> 33 of them) while true captures went 0.15 -> 0.13.

    Move-limit endings now score as draws. The +-0.5 *training label* is
    unchanged -- it is a position-dependent proxy that carries gradient
    (CONTEXT law 3); this is about what the gate calls a win.

    **Results from before 2026-08-03 were computed under the old rule and are
    not comparable to results after it.**
    """
    wins = losses = draws = 0
    time_wins = time_losses = 0
    all_plies, win_plies, loss_plies = [], [], []
    for result, plies in plies_by_result:
        all_plies.append(plies)
        if result >= 1:
            wins += 1
            win_plies.append(plies)
        elif result <= -1:
            losses += 1
            loss_plies.append(plies)
        else:
            draws += 1
            if result > 0:
                time_wins += 1
            elif result < 0:
                time_losses += 1
    n = len(plies_by_result)
    return {
        "games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        # Of the draws, how many were move-limit endings leaning each way.
        # Reported so "ahead at the cap" stays visible instead of vanishing
        # into the draw column.
        "time_leaning_wins": time_wins,
        "time_leaning_losses": time_losses,
        "score": round((wins + 0.5 * draws) / n, 4) if n else None,
        "mean_plies": _mean(all_plies),
        "mean_plies_when_won": _mean(win_plies),
        "mean_plies_when_lost": _mean(loss_plies),
    }


def run_benchmark(model_path, games, sims, anchor_sims, seed, start_fen=None):
    import random
    candidate, cand_label = _build_engine(model_path, sims)
    anchor, anchor_label = _build_engine(None, anchor_sims)

    n_white = games // 2       # candidate plays White
    n_black = games - n_white  # candidate plays Black

    # Per-game (candidate_result, plies), bucketed by the side the candidate
    # played.  play_one returns a White-perspective result; for the Black
    # games we negate it so both buckets are candidate-perspective.
    white_games = []
    black_games = []
    total_decisions = 0
    t0 = time.time()

    for i in range(n_white):
        random.seed(seed + i)
        result, plies, dec = play_one(candidate, anchor, start_fen=start_fen)
        white_games.append((result, plies))          # candidate is White
        total_decisions += dec

    for i in range(n_black):
        random.seed(seed + 1000 + i)
        result, plies, dec = play_one(anchor, candidate, start_fen=start_fen)
        black_games.append((-result, plies))          # candidate is Black
        total_decisions += dec

    elapsed = time.time() - t0

    white = summarize_side(white_games)   # model-as-White vs anchor-Black
    black = summarize_side(black_games)   # model-as-Black vs anchor-White

    cand_wins = white["wins"] + black["wins"]
    cand_losses = white["losses"] + black["losses"]
    cand_draws = white["draws"] + black["draws"]
    total_len = sum(p for _r, p in white_games) + sum(p for _r, p in black_games)
    score = (cand_wins + 0.5 * cand_draws) / games if games else 0.0
    # Share of the candidate's Black games that it won (kept for continuity
    # with older benchmark JSON; it is just black_strength restated).
    black_share = black["wins"] / n_black if n_black else 0.0

    return {
        "candidate": cand_label,
        "anchor": anchor_label,
        "games": games,
        "candidate_sims": sims,
        "anchor_sims": anchor_sims,
        "seed": seed,
        "start_fen": start_fen,
        # --- overall (merged both sides; kept for continuity) ---
        "candidate_wins": cand_wins,
        "candidate_draws": cand_draws,
        "candidate_losses": cand_losses,
        "candidate_score": round(score, 4),
        "candidate_black_win_share": round(black_share, 4),
        "mean_game_plies": round(total_len / games, 2) if games else 0.0,
        # --- per-side strength (isolates play quality by color) ---
        # white_strength = model-as-White score vs anchor-Black
        # black_strength = model-as-Black score vs anchor-White
        # Read plainly: the lower of the two is the side the model plays
        # worse and should improve. Track each across versions.
        "white_strength": white,
        "black_strength": black,
        "sec_per_decision": round(elapsed / total_decisions, 4) if total_decisions else None,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark a model against the heuristic anchor")
    p.add_argument("--model", type=str, default=None,
                   help="Candidate model path (omit = heuristic vs heuristic sanity)")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--sims", type=int, default=400, help="Candidate simulations")
    p.add_argument("--anchor-sims", type=int, default=None,
                   help="Heuristic anchor simulations (default: --sims)")
    p.add_argument("--seed", type=int, default=20260702)
    p.add_argument("--start-fen", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "benchmarks"))
    args = p.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be > 0")
    if args.sims <= 0:
        raise ValueError("--sims must be > 0")
    anchor_sims = args.anchor_sims if args.anchor_sims is not None else args.sims
    if anchor_sims <= 0:
        raise ValueError("--anchor-sims must be > 0")

    model_path = args.model
    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"--model not found: {model_path}")

    print(f"Benchmark: candidate={'heuristic' if not model_path else model_path} "
          f"({args.sims} sims) vs heuristic anchor ({anchor_sims} sims), {args.games} games")
    result = run_benchmark(
        model_path=model_path, games=args.games, sims=args.sims,
        anchor_sims=anchor_sims, seed=args.seed, start_fen=args.start_fen,
    )
    print(json.dumps(result, indent=2))

    w, b = result["white_strength"], result["black_strength"]
    print("\n--- Per-side strength (vs heuristic anchor) ---")
    for side, s in (("White", w), ("Black", b)):
        print(f"  As {side}: score {s['score']} "
              f"({s['wins']}-{s['losses']}-{s['draws']} W-L-D over {s['games']}), "
              f"mean plies {s['mean_plies']} "
              f"(won {s['mean_plies_when_won']}, lost {s['mean_plies_when_lost']})")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "heuristic" if not model_path else os.path.splitext(os.path.basename(model_path))[0]
    out_path = os.path.join(args.out_dir, f"benchmark_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
