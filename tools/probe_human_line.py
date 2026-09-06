"""Held-out human-game diagnostics, including both White search half-moves.

Never writes training data. Reconstructs legal history from recorded FENs and
explicitly lists ambiguous White intermediate paths. Search begins with a fresh
tree: this is not claimed to reproduce the unrecorded original cached tree.
"""
import argparse
import gc
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark import _build_engine
from match_evidence import atomic_json, file_hash, model_identity, read_rows, runtime_identity
from monster_chess import MonsterChessGame
from repetition import RepetitionTracker
from worker_lease import exclusive_workers


def action_uci(action):
    return [m.uci() for m in action] if isinstance(action, (tuple, list)) else [action.uci()]


def reconstruct(rows, index):
    if not 0 <= index < len(rows):
        raise ValueError("row index outside game")
    game = MonsterChessGame(rows[0]["fen"])
    # Human logs start at the actual initial board; rejecting later starts
    # avoids inventing a turn count or repetition history from an isolated FEN.
    if game.fen() != MonsterChessGame().fen():
        raise ValueError("human log must start from the initial position")
    repetition = RepetitionTracker()
    repetition.record(game, 0)
    history = []
    for i in range(index):
        matches = []
        for action in game.get_legal_actions():
            child = game.clone()
            child.apply_action(action)
            if child.fen() == rows[i + 1]["fen"]:
                matches.append(action)
        if not matches:
            raise ValueError(f"no legal transition between human rows {i} and {i + 1}")
        matches.sort(key=action_uci)
        chosen = matches[0]
        history.append({"row": i, "selected_reconstruction": action_uci(chosen),
                        "possible_paths": [action_uci(a) for a in matches]})
        # Apply to the live game, preserving its full move stack for diagnostics.
        game.apply_action(chosen)
        if repetition.record(game, i + 1):
            raise ValueError("human prefix is already drawn by the current repetition rule")
    return game, repetition, history


def force(game, moves, repetition):
    for uci in moves:
        action = next((a for a in game.get_search_actions() if a.uci() == uci), None)
        if action is None:
            raise ValueError(f"illegal diagnostic continuation {uci} at {game.fen()}")
        game.apply_search_action(action)
        if repetition.record(game):
            raise ValueError("forced diagnostic prefix ends in repetition")


def state_record(game):
    return {"fen": game.fen(), "half": bool(game.white_half_pending),
            "turn_count": game.turn_count, "color": "white" if game.is_white_turn else "black"}


def decision(game, engine, temperature):
    before = state_record(game)
    started = time.monotonic()
    action, policy, value = engine.get_best_action(game, temperature=temperature)
    out = {"before": before, "action": action.uci() if action else None,
           "root_value_side_to_move": float(value) if value is not None else None,
           "policy": dict(sorted(((str(k), float(v)) for k, v in policy.items()),
                                 key=lambda item: (-item[1], item[0]))),
           "seconds": time.monotonic() - started}
    if action is not None:
        game.apply_search_action(action)
        out["after"] = state_record(game)
    return out


@exclusive_workers
def run(args):
    output = Path(args.output).resolve()
    expected = {"human_source": model_identity(args.human_game),
                "models": [model_identity(p) for p in args.models],
                "opponent": model_identity(args.opponent_model),
                "runtime": runtime_identity(), "implementation_sha256": file_hash(__file__),
                "config": {k: v for k, v in vars(args).items() if k not in ("resume", "output")}}
    if output.exists():
        import json
        report = json.loads(output.read_text(encoding="utf-8"))
        if not args.resume or report["manifest"] != expected:
            raise ValueError("probe exists or resume provenance changed")
        if report["complete"]:
            return report
    else:
        report = {"manifest": expected, "complete": False, "probes": [],
                  "interpretation": "held-out diagnostic; fresh tree, not original cached search; not training data"}
        atomic_json(output, report)
    rows = read_rows(args.human_game)
    done = {tuple(p["id"]) for p in report["probes"]}
    import numpy as np
    import torch
    for model in args.models:
        for sims in args.sims:
            engine, _ = _build_engine(model, sims, engine="native")
            opponent, _ = _build_engine(args.opponent_model or model, sims, engine="native")
            for case in args.cases:
                index_text, _, move_text = case.partition(":")
                index = int(index_text)
                moves = [m for m in move_text.split(",") if m]
                for seed in args.seeds:
                    ident = (model, sims, case, seed)
                    if ident in done:
                        continue
                    game, repetition, history = reconstruct(rows, index)
                    force(game, moves, repetition)
                    random.seed(seed)
                    np.random.seed(seed % (2 ** 32))
                    torch.manual_seed(seed)
                    for search in (engine, opponent):
                        inner = getattr(search, "_inner", search)
                        inner._reuse_tree = inner._reuse_key = None
                        inner._decisions = 0
                    root_color = game.is_white_turn
                    probe = {"id": list(ident), "root": state_record(game),
                             "human_recorded_value": rows[index].get("mcts_value"),
                             "human_value_note": "AI rows record only original first-half search; human rows are static evaluations",
                             "reconstructed_prefix": history,
                             "ambiguous_prefix_rows": [h["row"] for h in history if len(h["possible_paths"]) > 1],
                             "forced_moves": moves, "decisions": []}
                    # Always finish the root actor's turn (both White halves).
                    repeated, no_action = False, False
                    while not game.is_terminal():
                        root_turn = game.is_white_turn == root_color and not probe.get("root_turn_finished")
                        if not root_turn and len(probe["decisions"]) >= args.continue_plies:
                            break
                        selected_engine = engine if game.is_white_turn == root_color else opponent
                        record = decision(game, selected_engine, args.temperature)
                        probe["decisions"].append(record)
                        if record["action"] is None:
                            no_action = True
                            break
                        if game.is_white_turn != root_color:
                            probe["root_turn_finished"] = True
                        if repetition.record(game, len(probe["decisions"])):
                            repeated = True
                            break
                    terminal = game.is_terminal()
                    probe["ending"] = ("repetition" if repeated else "king_capture" if terminal and abs(game.get_result()) == 1
                                       else "turn_cap" if terminal else "no_action" if no_action else "diagnostic_limit")
                    probe["result_white"] = 0.0 if repeated else float(game.get_result()) if terminal else None
                    report["probes"].append(probe)
                    done.add(ident)
                    atomic_json(output, report)
                    actions = [r["action"] for r in probe["decisions"][:3]]
                    print(f"{Path(model).parent.name} sims={sims} case={case} seed={seed}: {actions} {probe['ending']}", flush=True)
            del engine, opponent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report["complete"] = True
    atomic_json(output, report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human-game", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--opponent-model")
    ap.add_argument("--cases", nargs="+", default=["14", "14:c4b3", "14:c4d3", "14:f2f3", "15", "7"],
                    help="zero-based log row, optionally followed by :move,move")
    ap.add_argument("--sims", type=int, nargs="+", default=[3200, 6400])
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 211, 307])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--continue-plies", type=int, default=0,
                    help="0 = root turn only; otherwise total searched plies, at least the root turn")
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if min(args.sims) <= 0 or args.continue_plies < 0 or args.temperature < 0:
        ap.error("positive simulations; nonnegative continuation length and temperature")
    run(args)


if __name__ == "__main__":
    main()
