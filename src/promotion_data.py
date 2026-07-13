"""Prepare White-runner games for outcome contrast without bad policy imitation.

All records retain their value/outcome labels. Black policy targets are enabled
only for games where Black wins without White ever promoting; failed defenses
remain valuable negative examples for the value head but are not move teachers.
"""
import argparse
import json
from pathlib import Path

import chess


WHITE_PROMOTED_TYPES = (
    chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN,
)


def _white_has_promoted_piece(fen):
    board = chess.Board(fen)
    return any(board.pieces(piece_type, chess.WHITE)
               for piece_type in WHITE_PROMOTED_TYPES)


def is_successful_white_runner_prevention(records):
    """True only when Black wins and White never has a promoted piece."""
    if not records:
        return False
    result = float(records[-1].get("game_result", 0.0))
    white_promoted = any(
        rec.get("fen") and _white_has_promoted_piece(rec["fen"])
        for rec in records
    )
    return result < 0 and not white_promoted


def prepare_white_runner_games(input_dir, output_dir, expected_start_source=None):
    """Write a training-ready copy and return preparation counts.

    Interface invariant: ``output_dir`` must be absent or empty. Every output
    record gets an explicit ``policy_weight``. Only Black records in failed
    defenses receive weight 0; all value fields and White policy targets stay
    unchanged.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input_dir and output_dir must differ")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "games": 0,
        "successful_preventions": 0,
        "failed_games": 0,
        "masked_black_positions": 0,
        "value_records_kept": 0,
    }

    for path in sorted(input_dir.glob("game_*.jsonl")):
        records = [json.loads(line) for line in
                   path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue
        seen_sources = {rec.get("start_source") for rec in records
                        if rec.get("start_source") is not None}
        if expected_start_source is not None and seen_sources != {expected_start_source}:
            raise ValueError(
                f"{path.name}: expected start_source={expected_start_source!r}, "
                f"found {sorted(seen_sources)!r}"
            )
        successful_prevention = is_successful_white_runner_prevention(records)

        summary["games"] += 1
        if successful_prevention:
            summary["successful_preventions"] += 1
        else:
            summary["failed_games"] += 1

        prepared = []
        for rec in records:
            out = dict(rec)
            out["policy_weight"] = 1.0
            if out.get("current_player") == "black" and not successful_prevention:
                out["policy_weight"] = 0.0
                out["policy_mask_reason"] = "failed_white_promotion_prevention"
                summary["masked_black_positions"] += 1
            prepared.append(out)
            summary["value_records_kept"] += 1

        with open(output_dir / path.name, "w", encoding="utf-8") as f:
            for rec in prepared:
                f.write(json.dumps(rec) + "\n")

    with open(output_dir / "promotion_data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prepare White-runner games for promotion-aware training")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--expected-start-source", default=None)
    args = parser.parse_args()
    summary = prepare_white_runner_games(
        args.input_dir, args.output_dir,
        expected_start_source=args.expected_start_source,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
