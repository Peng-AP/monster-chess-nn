"""Evaluate model performance on recorded human-vs-model games.

Human game files are expected in data/raw/human_games/*.jsonl, written by
play.py. Each record is an AI-turn position and includes:
  - fen
  - current_player ("white" or "black"): side controlled by AI in that game
  - game_result (scalar from White perspective; sign indicates winner)

This script reports game-level win/loss/draw rates by AI side and (optionally)
position-level value diagnostics from a given model.
"""
import argparse
import glob
import json
import os
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)


def _result_bucket(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _new_side_stats():
    return {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "score": 0.0,
        "win_rate": 0.0,
        "loss_rate": 0.0,
        "draw_rate": 0.0,
    }


def _finalize_side_stats(stats):
    n = stats["games"]
    if n <= 0:
        return stats
    stats["win_rate"] = stats["wins"] / n
    stats["loss_rate"] = stats["losses"] / n
    stats["draw_rate"] = stats["draws"] / n
    stats["score"] = stats["score"] / n
    return stats


def _update_game_stats(stats, ai_side, game_result):
    """Update side stats using game_result from White perspective."""
    rb = _result_bucket(game_result)
    stats[ai_side]["games"] += 1
    if rb == 0:
        stats[ai_side]["draws"] += 1
        stats[ai_side]["score"] += 0.5
        return

    ai_won = (rb > 0 and ai_side == "white") or (rb < 0 and ai_side == "black")
    if ai_won:
        stats[ai_side]["wins"] += 1
        stats[ai_side]["score"] += 1.0
    else:
        stats[ai_side]["losses"] += 1


def _new_diag_stats():
    return {
        "positions": 0,
        "mae_vs_game_result": None,
        "sign_accuracy_non_draw": None,
    }


def _finalize_diag_stats(stats):
    out = {}
    for side, vals in stats.items():
        n = vals["positions"]
        if n <= 0:
            out[side] = _new_diag_stats()
            continue
        mae = vals["mae_sum"] / n
        if vals["sign_total"] > 0:
            sign_acc = vals["sign_correct"] / vals["sign_total"]
        else:
            sign_acc = None
        out[side] = {
            "positions": n,
            "mae_vs_game_result": mae,
            "sign_accuracy_non_draw": sign_acc,
        }
    return out


def evaluate_human_games(human_dir, model_path=None, include_value_diagnostics=True, max_games=None):
    files = sorted(glob.glob(os.path.join(human_dir, "*.jsonl")))
    if max_games is not None and max_games > 0:
        files = files[-max_games:]

    game_stats = {
        "white": _new_side_stats(),
        "black": _new_side_stats(),
    }
    diag_acc = {
        "white": {"positions": 0, "mae_sum": 0.0, "sign_total": 0, "sign_correct": 0},
        "black": {"positions": 0, "mae_sum": 0.0, "sign_total": 0, "sign_correct": 0},
    }

    skipped_empty = 0
    skipped_unknown_side = 0
    parsed_games = 0

    evaluator = None
    if include_value_diagnostics and model_path and os.path.exists(model_path):
        from evaluation import NNEvaluator
        evaluator = NNEvaluator(model_path)

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if not records:
            skipped_empty += 1
            continue

        ai_side = records[0].get("current_player")
        if ai_side not in ("white", "black"):
            skipped_unknown_side += 1
            continue

        result = float(records[-1].get("game_result", 0.0))
        parsed_games += 1
        _update_game_stats(game_stats, ai_side, result)

        if evaluator is None:
            continue

        from monster_chess import MonsterChessGame

        for rec in records:
            side = rec.get("current_player")
            fen = rec.get("fen")
            if side not in ("white", "black") or not fen:
                continue

            game = MonsterChessGame(fen=fen)
            pred_white = float(evaluator(game))
            pred_side = pred_white if side == "white" else -pred_white
            target_side = result if side == "white" else -result

            acc = diag_acc[side]
            acc["positions"] += 1
            acc["mae_sum"] += abs(pred_side - target_side)

            tb = _result_bucket(target_side)
            if tb != 0:
                acc["sign_total"] += 1
                pb = _result_bucket(pred_side)
                if pb == tb:
                    acc["sign_correct"] += 1

    for side in ("white", "black"):
        _finalize_side_stats(game_stats[side])

    by_human_side = {
        "white": {
            "games": game_stats["black"]["games"],
            "wins": game_stats["black"]["losses"],
            "losses": game_stats["black"]["wins"],
            "draws": game_stats["black"]["draws"],
            "score": 1.0 - game_stats["black"]["score"] if game_stats["black"]["games"] > 0 else 0.0,
        },
        "black": {
            "games": game_stats["white"]["games"],
            "wins": game_stats["white"]["losses"],
            "losses": game_stats["white"]["wins"],
            "draws": game_stats["white"]["draws"],
            "score": 1.0 - game_stats["white"]["score"] if game_stats["white"]["games"] > 0 else 0.0,
        },
    }

    for side in ("white", "black"):
        n = by_human_side[side]["games"]
        if n > 0:
            by_human_side[side]["win_rate"] = by_human_side[side]["wins"] / n
            by_human_side[side]["loss_rate"] = by_human_side[side]["losses"] / n
            by_human_side[side]["draw_rate"] = by_human_side[side]["draws"] / n
        else:
            by_human_side[side]["win_rate"] = 0.0
            by_human_side[side]["loss_rate"] = 0.0
            by_human_side[side]["draw_rate"] = 0.0

    total_games = game_stats["white"]["games"] + game_stats["black"]["games"]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "human_dir": human_dir,
        "model_path": model_path if evaluator is not None else None,
        "files_scanned": len(files),
        "games_parsed": parsed_games,
        "skipped_empty": skipped_empty,
        "skipped_unknown_side": skipped_unknown_side,
        "total_games": total_games,
        "by_ai_side": game_stats,
        "by_human_side": by_human_side,
    }

    if evaluator is not None:
        summary["value_diagnostics"] = _finalize_diag_stats(diag_acc)
    return summary


def _fmt_pct(x):
    return f"{100.0 * x:.1f}%"


def main():
    default_human_dir = os.path.join(PROJECT_ROOT, "data", "raw", "human_games")
    default_model_path = os.path.join(PROJECT_ROOT, "models", "best_value_net.pt")

    parser = argparse.ArgumentParser(description="Evaluate human-vs-model game outcomes by side")
    parser.add_argument("--human-dir", type=str, default=default_human_dir,
                        help="Directory containing human game .jsonl files")
    parser.add_argument("--model", type=str, default=default_model_path,
                        help="Model path for optional value diagnostics")
    parser.add_argument("--no-value-diagnostics", action="store_true",
                        help="Skip position-level diagnostics")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Use only the most recent N human games")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to save full JSON summary")
    args = parser.parse_args()

    include_diag = not args.no_value_diagnostics
    model_path = args.model if include_diag else None

    summary = evaluate_human_games(
        human_dir=args.human_dir,
        model_path=model_path,
        include_value_diagnostics=include_diag,
        max_games=args.max_games,
    )

    print("\nHuman-vs-model summary")
    print(f"  Directory: {summary['human_dir']}")
    print(f"  Files:     {summary['files_scanned']}")
    print(f"  Parsed:    {summary['games_parsed']} games")
    if summary["skipped_empty"] > 0:
        print(f"  Skipped empty: {summary['skipped_empty']}")
    if summary["skipped_unknown_side"] > 0:
        print(f"  Skipped unknown side: {summary['skipped_unknown_side']}")

    ai_white = summary["by_ai_side"]["white"]
    ai_black = summary["by_ai_side"]["black"]
    print("\nAI side performance")
    print(
        f"  AI as White: games={ai_white['games']}  W/L/D={ai_white['wins']}/{ai_white['losses']}/{ai_white['draws']}  "
        f"score={ai_white['score']:.3f}  win={_fmt_pct(ai_white['win_rate'])}"
    )
    print(
        f"  AI as Black: games={ai_black['games']}  W/L/D={ai_black['wins']}/{ai_black['losses']}/{ai_black['draws']}  "
        f"score={ai_black['score']:.3f}  win={_fmt_pct(ai_black['win_rate'])}"
    )

    human_black = summary["by_human_side"]["black"]
    print("\nHuman practical signal")
    print(
        f"  Human as Black vs AI-White: games={human_black['games']}  "
        f"W/L/D={human_black['wins']}/{human_black['losses']}/{human_black['draws']}  "
        f"score={human_black['score']:.3f}"
    )

    if "value_diagnostics" in summary:
        diag_black = summary["value_diagnostics"]["black"]
        diag_white = summary["value_diagnostics"]["white"]
        print("\nValue diagnostics (by current_player side)")
        print(
            f"  White positions: n={diag_white['positions']}  "
            f"mae={diag_white['mae_vs_game_result'] if diag_white['mae_vs_game_result'] is not None else 'n/a'}  "
            f"sign_acc={diag_white['sign_accuracy_non_draw'] if diag_white['sign_accuracy_non_draw'] is not None else 'n/a'}"
        )
        print(
            f"  Black positions: n={diag_black['positions']}  "
            f"mae={diag_black['mae_vs_game_result'] if diag_black['mae_vs_game_result'] is not None else 'n/a'}  "
            f"sign_acc={diag_black['sign_accuracy_non_draw'] if diag_black['sign_accuracy_non_draw'] is not None else 'n/a'}"
        )

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved JSON: {args.json_out}")


if __name__ == "__main__":
    main()
