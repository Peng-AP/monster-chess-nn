"""Promotion-specific evaluation for runner/defender positions."""
import chess

from monster_chess import MonsterChessGame


def _white_already_promoted(board):
    return any(board.pieces(piece_type, chess.WHITE) for piece_type in (
        chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN,
    ))


def _black_already_promoted(board):
    standard_max = {
        chess.QUEEN: 1,
        chess.ROOK: 2,
        chess.BISHOP: 2,
        chess.KNIGHT: 2,
    }
    return any(len(board.pieces(piece_type, chess.BLACK)) > maximum
               for piece_type, maximum in standard_max.items())


def _action_promotes(action):
    if isinstance(action, tuple):
        return any(move.promotion is not None for move in action)
    return action.promotion is not None


def play_probe_game(white_engine, black_engine, start_fen, runner_color,
                    defender_color, max_plies=600):
    """Play one measured game and report promotion, king, and outcome facts."""
    if runner_color not in ("white", "black"):
        raise ValueError("runner_color must be white or black")
    if defender_color not in ("white", "black") or defender_color == runner_color:
        raise ValueError("defender_color must be the opposite color")

    game = MonsterChessGame(fen=start_fen)
    runner_is_white = runner_color == "white"
    runner_promoted = (
        _white_already_promoted(game.board)
        if runner_is_white else _black_already_promoted(game.board)
    )
    plies = 0

    while not game.is_terminal() and plies < max_plies:
        mover_is_white = game.is_white_turn
        engine = white_engine if mover_is_white else black_engine
        action, _probs, _value = engine.get_best_action(game, temperature=0.0)
        if action is None:
            break
        if mover_is_white == runner_is_white and _action_promotes(action):
            runner_promoted = True
        apply_fn = getattr(game, "apply_search_action", None)
        if isinstance(action, tuple) or apply_fn is None:
            game.apply_action(action)
        else:
            apply_fn(action)
        plies += 1

    result_white = game.get_result()
    defender_is_white = defender_color == "white"
    defender_result = result_white if defender_is_white else -result_white
    defender_king = game.board.king(chess.WHITE if defender_is_white else chess.BLACK)
    return {
        "result_white": result_white,
        "defender_result": defender_result,
        "runner_promoted": bool(runner_promoted),
        "promotion_prevented": not runner_promoted,
        "defender_king_survived": defender_king is not None,
        "plies": plies,
    }


def summarize_probe_results(results):
    """Aggregate independent promotion, king-safety, and game-score metrics."""
    n = len(results)
    if not n:
        return {
            "games": 0,
            "prevention_rate": None,
            "king_survival_rate": None,
            "defender_score": None,
            "mean_plies": None,
        }
    prevented = sum(bool(r["promotion_prevented"]) for r in results)
    survived = sum(bool(r["defender_king_survived"]) for r in results)
    wins = sum(r["defender_result"] > 0 for r in results)
    draws = sum(r["defender_result"] == 0 for r in results)
    return {
        "games": n,
        "prevention_rate": prevented / n,
        "king_survival_rate": survived / n,
        "defender_score": (wins + 0.5 * draws) / n,
        "mean_plies": sum(r["plies"] for r in results) / n,
    }


def compare_probe_reports(candidate, incumbent, max_prevention_drop=0.0,
                          max_king_survival_drop=0.0, max_score_drop=0.05):
    """Compare a challenger with fixed no-regression tolerances per metric."""
    tolerances = {
        "prevention_rate": float(max_prevention_drop),
        "king_survival_rate": float(max_king_survival_drop),
        "defender_score": float(max_score_drop),
    }
    deltas = {}
    failures = {}
    for metric, max_drop in tolerances.items():
        delta = float(candidate[metric]) - float(incumbent[metric])
        deltas[metric] = delta
        if delta < -max_drop:
            failures[metric] = {
                "delta": delta,
                "max_drop": max_drop,
            }
    return {
        "passed": not failures,
        "deltas": deltas,
        "failures": failures,
    }
