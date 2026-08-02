"""M2/M3: how models actually behave defending against a promotion.

Two modes over the same deck.

**search** (M2) -- run MCTS on every deck position for every model and record
what search concluded, not just what the priors suggested. HANDOFF SS4.4 found
dup1 refusing to capture a pawn about to promote, rating the blunder
Q=-0.1478 against the capture's Q=-0.3235, while a *static* 400-position probe
put dup1 mid-pack on capture-is-top-1. That probe measured priors; the failure
lived in Q. This measures Q.

**outcomes** (M3 ground truth) -- play every deck position out once with the
heuristic engine on both sides. Deliberately model-independent: the realized
result is a property of the position, identical for all four models, so the
*comparison* between models is fair even though heuristic play is not perfect
play. Do not read the absolute calibration as truth.

    py -3 tools/promotion_defense_probe.py --mode search \\
        --model v17=models/fresh_start_v17/best_value_net.pt \\
        --model ramp=models/rejected/fresh_start_v18_ramp/best_value_net.pt \\
        --deck data/start_fens/promotion_defense_deck_v1.jsonl
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import DEFAULT_GAME_WORKERS  # noqa: E402

_state = {}


def _search_root(engine, game_state):
    """Run the search and hand back the root, mirroring get_best_action's
    dispatch. MCTS exposes no public 'search and give me the tree' call, and
    per-child Q is exactly what this probe is for."""
    root = engine._root_for_search(game_state)
    if engine._has_policy and engine._supports_batch:
        engine._run_batched_puct(root)
    else:
        engine._run_sequential(root)
    return root


def _init_search(models, sims):
    from benchmark import _build_engine
    from evaluation import NNEvaluator
    _state["engines"] = {}
    _state["evals"] = {}
    for name, path in models:
        engine, _label = _build_engine(path, sims)
        _state["engines"][name] = engine
        _state["evals"][name] = engine.eval_fn
    _state["names"] = [n for n, _p in models]
    assert all(isinstance(e, NNEvaluator) for e in _state["evals"].values())


def _probe_one(entry):
    """One deck position through every model. -> {model: row}"""
    import chess
    from monster_chess import MonsterChessGame

    fen = entry["fen"]
    captures = set(entry["captures"])
    rows = {}
    for name in _state["names"]:
        engine = _state["engines"][name]
        game = MonsterChessGame(fen=fen)
        # PERSPECTIVES, because these two do not agree and mixing them inverts
        # the finding: NNEvaluator.evaluate_with_policy returns a WHITE-
        # perspective value, while a root child's Q accumulates in the ROOT's
        # side-to-move perspective (mcts._backpropagate). Every deck position
        # is Black to move, so everything below is normalised to Black:
        # positive = good for Black, the defender.
        raw_value_white, _pol = _state["evals"][name].evaluate_with_policy(game)
        raw_value = -raw_value_white
        root = _search_root(engine, game)

        cap_q = cap_visits = cap_prior = None
        oth_q = oth_visits = None
        total_visits = sum(c.visit_count for c in root.children) or 1
        cap_visit_share = 0.0
        cap_seen = False
        for child in root.children:
            uci = child.action.uci() if isinstance(child.action, chess.Move) else str(child.action)
            is_cap = uci in captures
            # MCTSNode.q_value returns 0.0 for an UNVISITED child. Most Q here
            # is negative, so including unvisited children makes "best other"
            # 0.0 in almost every position and the comparison meaningless.
            # Only visited children carry a search opinion.
            visited = child.visit_count > 0
            q = child.q_value
            if is_cap:
                cap_seen = True
                cap_visit_share += child.visit_count / total_visits
                if visited and (cap_q is None or q > cap_q):
                    cap_q, cap_visits, cap_prior = q, child.visit_count, child.prior
                if cap_prior is None:
                    cap_prior = child.prior
            elif visited and (oth_q is None or q > oth_q):
                oth_q, oth_visits = q, child.visit_count

        action, _probs, search_value = engine.get_best_action(game, temperature=0.0)
        chosen = action.uci() if isinstance(action, chess.Move) else str(action)
        rows[name] = {
            "fen": fen,
            "chosen": chosen,
            "chose_capture": chosen in captures,
            "raw_value_black": round(float(raw_value), 4),
            "search_value_black": round(float(search_value), 4),
            "q_capture": None if cap_q is None else round(float(cap_q), 4),
            "q_best_other": None if oth_q is None else round(float(oth_q), 4),
            "visits_capture": cap_visits,
            "visits_best_other": oth_visits,
            "prior_capture": None if cap_prior is None else round(float(cap_prior), 4),
            "capture_visit_share": round(cap_visit_share, 4),
            "capture_visited": cap_visits is not None,
            "capture_offered": cap_seen,
        }
    return rows


def _init_outcomes(white_model, black_model, sims):
    from benchmark import _build_engine
    _state["white"], _ = _build_engine(white_model, sims)
    _state["black"], _ = _build_engine(black_model, sims)


def _play_out(task):
    from benchmark import play_one
    fen, seed = task
    random.seed(seed)
    result, plies, _dec = play_one(_state["white"], _state["black"], start_fen=fen)
    return {"fen": fen, "result": result, "plies": plies}


def summarize(rows_by_model):
    out = {}
    for name, rows in rows_by_model.items():
        n = len(rows)
        chose = sum(1 for r in rows if r["chose_capture"])
        gaps = [r["q_capture"] - r["q_best_other"] for r in rows
                if r["q_capture"] is not None and r["q_best_other"] is not None]
        # Only the positions where search actually formed an opinion on both
        # options can say anything about *why* a refusal happened.
        judged = [r for r in rows if r["q_capture"] is not None
                  and r["q_best_other"] is not None]
        refused_despite_better_q = [r for r in judged if not r["chose_capture"]
                                    and r["q_capture"] > r["q_best_other"]]
        out[name] = {
            "positions": n,
            "chose_capture": chose,
            "capture_rate": round(chose / n, 4) if n else None,
            "mean_capture_visit_share": round(
                statistics.fmean(r["capture_visit_share"] for r in rows), 4) if n else None,
            "mean_raw_value_black": round(statistics.fmean(r["raw_value_black"] for r in rows), 4) if n else None,
            "mean_search_value_black": round(
                statistics.fmean(r["search_value_black"] for r in rows), 4) if n else None,
            "mean_prior_capture": round(statistics.fmean(
                r["prior_capture"] for r in rows if r["prior_capture"] is not None), 4),
            "positions_with_both_q": len(judged),
            "mean_q_gap_capture_minus_other": round(statistics.fmean(gaps), 4) if gaps else None,
            "capture_unvisited": sum(1 for r in rows if not r["capture_visited"]),
            "refused_despite_higher_capture_q": len(refused_despite_better_q),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("search", "outcomes"), default="search")
    ap.add_argument("--deck", default=os.path.join(
        ROOT, "data", "start_fens", "promotion_defense_deck_v1.jsonl"))
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--workers", type=int, default=DEFAULT_GAME_WORKERS)
    ap.add_argument("--playout-white", default=None, metavar="PATH",
                    help="outcomes mode: model for White (default: heuristic)")
    ap.add_argument("--playout-black", default=None, metavar="PATH",
                    help="outcomes mode: model for Black (default: heuristic)")
    ap.add_argument("--label", default=None,
                    help="tag for the outcomes artifact filename")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks"))
    args = ap.parse_args()

    deck = [json.loads(l) for l in open(args.deck, encoding="utf-8") if l.strip()]
    if args.limit:
        deck = deck[:args.limit]
    print(f"deck: {len(deck)} positions from {os.path.relpath(args.deck, ROOT)}")

    t0 = time.time()
    if args.mode == "search":
        models = []
        for spec in args.model:
            name, _, path = spec.partition("=")
            if not path:
                ap.error(f"--model expects NAME=PATH, got {spec!r}")
            if not os.path.exists(path):
                ap.error(f"no such model: {path}")
            models.append((name, path))
        if not models:
            ap.error("--mode search needs at least one --model")
        print(f"models: {', '.join(n for n, _p in models)} @ {args.sims} sims")

        with mp.Pool(args.workers, initializer=_init_search,
                     initargs=(models, args.sims)) as pool:
            results = pool.map(_probe_one, deck)

        rows_by_model = {name: [] for name, _p in models}
        for row in results:
            for name, r in row.items():
                rows_by_model[name].append(r)
        payload = {
            "mode": "search", "sims": args.sims, "positions": len(deck),
            "deck": os.path.relpath(args.deck, ROOT).replace("\\", "/"),
            "summary": summarize(rows_by_model),
            "rows": rows_by_model,
            "elapsed_sec": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        stem = "promotion_defense_search"
    else:
        tasks = [(e["fen"], args.seed + i) for i, e in enumerate(deck)]
        def who(path):
            return os.path.basename(os.path.dirname(path)) if path else "heuristic"
        print(f"playout: White={who(args.playout_white)} "
              f"Black={who(args.playout_black)} @ {args.sims} sims")
        with mp.Pool(args.workers, initializer=_init_outcomes,
                     initargs=(args.playout_white, args.playout_black,
                               args.sims)) as pool:
            outcomes = pool.map(_play_out, tasks)
        black_wins = sum(1 for o in outcomes if o["result"] < 0)
        payload = {
            "mode": "outcomes", "sims": args.sims, "positions": len(deck),
            "deck": os.path.relpath(args.deck, ROOT).replace("\\", "/"),
            "white_player": who(args.playout_white),
            "black_player": who(args.playout_black),
            "black_win_rate": round(black_wins / len(outcomes), 4) if outcomes else None,
            "mean_plies": round(statistics.fmean(o["plies"] for o in outcomes), 1),
            "outcomes": outcomes,
            "elapsed_sec": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        stem = "promotion_defense_outcomes"
        if args.label:
            stem += f"_{args.label}"

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir,
                        f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("rows", "outcomes")}, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
