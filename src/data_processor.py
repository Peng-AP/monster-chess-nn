import chess
import json
import os
import re

import numpy as np
from tqdm import tqdm

from config import (
    TENSOR_SHAPE, TURN_LAYER,
    MOVE_COUNT_LAYER, PAWN_ADVANCEMENT_LAYER,
    POLICY_SIZE,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    HUMAN_DATA_WEIGHT, HUMANSEED_DATA_WEIGHT, BLACKFOCUS_DATA_WEIGHT,
    DATA_RETENTION_MAX_GENERATION_AGE, DATA_RETENTION_MIN_NONHUMAN_PLIES,
    DATA_RETENTION_MIN_HUMANSEED_POLICY_ENTROPY,
    SLIDING_WINDOW, POSITION_BUDGET, POSITION_BUDGET_MAX, PROCESSED_POSITION_CAP, RANDOM_SEED,
    SOURCE_QUOTA_ENABLED, SOURCE_QUOTA_SELFPLAY, SOURCE_QUOTA_HUMAN,
    SOURCE_QUOTA_BLACKFOCUS, SOURCE_QUOTA_HUMANSEED,
    SELFPLAY_TARGET_MCTS_LAMBDA, HUMAN_TARGET_MCTS_LAMBDA,
    BLACKFOCUS_TARGET_MCTS_LAMBDA, HUMANSEED_TARGET_MCTS_LAMBDA,
)

SOURCE_ORDER = ("selfplay", "human", "blackfocus", "humanseed")

# Piece -> layer index
PIECE_TO_LAYER = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def fen_to_tensor(fen, is_white_turn=True):
    """Convert a FEN string to an (8, 8, 15) tensor.

    Layers:
      0-11: piece positions (binary)
      12:   turn indicator (+1 White, -1 Black)
      13:   move count within turn (0 or 1 for White's double-move)
      14:   White pawn advancement gradient (0.0 at rank 2, 1.0 at rank 8)
    """
    board = chess.Board(fen)
    tensor = np.zeros(TENSOR_SHAPE, dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            layer = PIECE_TO_LAYER[(piece.piece_type, piece.color)]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[rank, file, layer] = 1.0

    # Turn indicator
    tensor[:, :, TURN_LAYER] = 1.0 if is_white_turn else -1.0

    # Move count layer (always 0 at position level — the MCTS treats
    # double-moves atomically, so we record 0 here; this layer is
    # reserved for future per-half-move encoding)
    tensor[:, :, MOVE_COUNT_LAYER] = 0.0

    # White pawn advancement gradient
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        tensor[rank, file, PAWN_ADVANCEMENT_LAYER] = (rank - 1) / 6.0

    return tensor


def mirror_tensor(tensor):
    """Horizontally mirror a position tensor (flip files a<->h).

    Monster Chess is symmetric across the file axis, so mirroring
    produces an equally valid position with the same evaluation.
    This doubles training data for free.
    """
    # Flip along the file axis (axis 1): file 0<->7, 1<->6, etc.
    return tensor[:, ::-1, :].copy()


# ------------------------------------------------------------------
# Policy encoding: flat from_sq * 64 + to_sq  (4096 indices)
# ------------------------------------------------------------------

def move_to_index(move):
    """Convert a chess.Move to a flat policy index."""
    return move.from_square * 64 + move.to_square


def mirror_move_index(idx):
    """Mirror a flat policy index across the file axis (a<->h)."""
    from_sq = idx // 64
    to_sq = idx % 64
    from_file, from_rank = from_sq % 8, from_sq // 8
    to_file, to_rank = to_sq % 8, to_sq // 8
    new_from = from_rank * 8 + (7 - from_file)
    new_to = to_rank * 8 + (7 - to_file)
    return new_from * 64 + new_to


def policy_dict_to_target(policy_dict, is_white):
    """Convert an MCTS action_probs dict to a dense policy target vector.

    For Black: each key is a UCI move string -> index directly.
    For White: each key is "m1_uci,m2_uci". We marginalize over m2 to
    get P(m1), since the policy head predicts single moves and m2 is
    evaluated from the post-m1 board state during MCTS.
    """
    target = np.zeros(POLICY_SIZE, dtype=np.float32)
    if policy_dict is None:
        return target  # uniform-ish fallback (all zeros, masked later)

    for action_str, prob in policy_dict.items():
        if is_white:
            m1_uci = action_str.split(",")[0]
            move = chess.Move.from_uci(m1_uci)
        else:
            move = chess.Move.from_uci(action_str)
        target[move_to_index(move)] += prob

    # Renormalize (White's m1 marginal should already sum to ~1)
    total = target.sum()
    if total > 0:
        target /= total
    return target


def mirror_policy(policy_vec):
    """Mirror a dense policy vector across the file axis."""
    mirrored = np.zeros_like(policy_vec)
    for idx in range(POLICY_SIZE):
        if policy_vec[idx] > 0:
            mirrored[mirror_move_index(idx)] = policy_vec[idx]
    return mirrored


def _build_source_quota_ratios(quota_selfplay, quota_human, quota_blackfocus, quota_humanseed):
    return {
        "selfplay": float(quota_selfplay),
        "human": float(quota_human),
        "blackfocus": float(quota_blackfocus),
        "humanseed": float(quota_humanseed),
    }


def _build_source_target_lambdas(selfplay_target_mcts_lambda, human_target_mcts_lambda,
                                 blackfocus_target_mcts_lambda, humanseed_target_mcts_lambda):
    return {
        "selfplay": float(selfplay_target_mcts_lambda),
        "human": float(human_target_mcts_lambda),
        "blackfocus": float(blackfocus_target_mcts_lambda),
        "humanseed": float(humanseed_target_mcts_lambda),
    }


def _validate_source_settings(use_source_quotas, source_quota_ratios, source_target_lambdas):
    for key, value in source_quota_ratios.items():
        if value < 0:
            raise ValueError(f"--quota-{key} must be >= 0")
    if use_source_quotas and sum(source_quota_ratios.values()) <= 0:
        raise ValueError("Source quotas enabled but all source ratios are zero")
    for key, value in source_target_lambdas.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--{key}-target-mcts-lambda must be in [0, 1]")


def _source_kind(game):
    """Normalize game source to one of SOURCE_ORDER."""
    if game.get("is_human"):
        return "human"
    if game.get("is_humanseed"):
        return "humanseed"
    if game.get("is_blackfocus"):
        return "blackfocus"
    return "selfplay"


def _interleave_games_by_source(games, seed):
    """Round-robin games by source to avoid long contiguous source runs."""
    rng = np.random.default_rng(seed)
    by_source = {}
    for g in games:
        src = _source_kind(g)
        by_source.setdefault(src, []).append(g)
    for src_games in by_source.values():
        rng.shuffle(src_games)

    source_cycle = [s for s in SOURCE_ORDER if s in by_source]
    # Preserve any future source buckets in deterministic order.
    source_cycle.extend(sorted(s for s in by_source if s not in SOURCE_ORDER))
    if not source_cycle:
        return []

    out = []
    idx = {s: 0 for s in source_cycle}
    while True:
        added_any = False
        for s in source_cycle:
            src_games = by_source[s]
            i = idx[s]
            if i < len(src_games):
                out.append(src_games[i])
                idx[s] = i + 1
                added_any = True
        if not added_any:
            break
    return out


def _allocate_source_quotas(train_cap, ratios, train_games):
    """Allocate integer source quotas that sum to train_cap."""
    present = {_source_kind(g) for g in train_games}
    active = [s for s in SOURCE_ORDER if s in present and ratios.get(s, 0.0) > 0.0]
    if train_cap is None or train_cap <= 0 or not active:
        return None

    denom = sum(float(ratios[s]) for s in active)
    if denom <= 0:
        return None

    raw = {s: train_cap * (float(ratios[s]) / denom) for s in active}
    quota = {s: int(np.floor(raw[s])) for s in active}
    remaining = int(train_cap - sum(quota.values()))
    if remaining > 0:
        order = sorted(active, key=lambda s: (raw[s] - quota[s]), reverse=True)
        for s in order:
            if remaining <= 0:
                break
            quota[s] += 1
            remaining -= 1

    for s in SOURCE_ORDER:
        quota.setdefault(s, 0)
    return quota


def _collect_generation_dirs(subdirs):
    """Map generation number -> list of matching nn_gen* directory names."""
    gen_pattern = re.compile(r'^nn_gen(\d+)(?:_.*)?$')
    gen_to_dirs = {}
    for d in subdirs:
        m = gen_pattern.match(d)
        if not m:
            continue
        gen = int(m.group(1))
        gen_to_dirs.setdefault(gen, []).append(d)
    return gen_to_dirs


def _count_positions_in_dir(raw_dir, dirname):
    """Count recorded positions (JSONL lines) in one raw data subdirectory."""
    total = 0
    dpath = os.path.join(raw_dir, dirname)
    for fname in os.listdir(dpath):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(dpath, fname)
        with open(path, "r", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def _generation_from_game_id(game_id):
    """Extract nn_gen number from a relative game path, else None."""
    top = str(game_id).replace("\\", "/").split("/", 1)[0]
    m = re.match(r'^nn_gen(\d+)(?:_.*)?$', top)
    if not m:
        return None
    return int(m.group(1))


def _sparse_policy_entropy(policy_dict):
    """Shannon entropy (nats) for sparse policy dicts."""
    if not isinstance(policy_dict, dict) or not policy_dict:
        return 0.0
    probs = np.array([float(v) for v in policy_dict.values() if float(v) > 0.0], dtype=np.float64)
    if probs.size == 0:
        return 0.0
    probs_sum = float(probs.sum())
    if probs_sum <= 0.0:
        return 0.0
    probs = probs / probs_sum
    return float(-np.sum(probs * np.log(probs)))


def _mean_record_policy_entropy(records):
    if not records:
        return 0.0
    vals = [_sparse_policy_entropy(rec.get("policy")) for rec in records]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _apply_game_retention_policy(games, max_generation_age=0, min_nonhuman_plies=0,
                                 min_humanseed_policy_entropy=0.0):
    """Apply bounded retention filters and return kept games + summary."""
    max_generation_age = int(max_generation_age or 0)
    min_nonhuman_plies = int(min_nonhuman_plies or 0)
    min_humanseed_policy_entropy = float(min_humanseed_policy_entropy or 0.0)

    generations = [
        int(g["generation"]) for g in games
        if g.get("generation") is not None
    ]
    latest_generation = max(generations) if generations else None

    kept = []
    dropped = {
        "age": {"games": 0, "positions": 0, "by_source": {}},
        "short_nonhuman": {"games": 0, "positions": 0, "by_source": {}},
        "low_entropy_humanseed": {"games": 0, "positions": 0, "by_source": {}},
    }
    for game in games:
        source_kind = game.get("source_kind") or _source_kind(game)
        positions = int(len(game.get("records", [])))
        generation = game.get("generation")

        if (
            latest_generation is not None
            and generation is not None
            and max_generation_age > 0
            and (latest_generation - int(generation)) > max_generation_age
        ):
            dropped["age"]["games"] += 1
            dropped["age"]["positions"] += positions
            dropped["age"]["by_source"][source_kind] = (
                int(dropped["age"]["by_source"].get(source_kind, 0)) + 1
            )
            continue

        if source_kind != "human" and min_nonhuman_plies > 0 and positions < min_nonhuman_plies:
            dropped["short_nonhuman"]["games"] += 1
            dropped["short_nonhuman"]["positions"] += positions
            dropped["short_nonhuman"]["by_source"][source_kind] = (
                int(dropped["short_nonhuman"]["by_source"].get(source_kind, 0)) + 1
            )
            continue
        game_entropy = float(game.get("mean_policy_entropy", 0.0))
        if (
            source_kind == "humanseed"
            and min_humanseed_policy_entropy > 0.0
            and game_entropy < min_humanseed_policy_entropy
        ):
            dropped["low_entropy_humanseed"]["games"] += 1
            dropped["low_entropy_humanseed"]["positions"] += positions
            dropped["low_entropy_humanseed"]["by_source"][source_kind] = (
                int(dropped["low_entropy_humanseed"]["by_source"].get(source_kind, 0)) + 1
            )
            continue

        kept.append(game)

    return kept, {
        "max_generation_age": max_generation_age,
        "min_nonhuman_plies": min_nonhuman_plies,
        "min_humanseed_policy_entropy": min_humanseed_policy_entropy,
        "latest_generation": latest_generation,
        "input_games": int(len(games)),
        "input_positions": int(sum(len(g.get("records", [])) for g in games)),
        "kept_games": int(len(kept)),
        "kept_positions": int(sum(len(g.get("records", [])) for g in kept)),
        "dropped": dropped,
    }


def _filter_dirs(raw_dir, keep_generations=None, position_budget=None,
                 position_budget_max=None, include_human=True):
    """Decide which subdirectories to include based on sliding window.

    Always includes: curriculum_bootstrap/ and optionally human_games/
    Includes either:
      - last `keep_generations` nn_gen* pairs (normal + curriculum), or
      - enough recent generations to hit `position_budget` raw positions.
    Excludes everything else (normal/ heuristic bootstrap, older generations).
    """
    if keep_generations is not None and position_budget is not None:
        raise ValueError("Use either keep_generations or position_budget, not both.")
    if position_budget_max is not None and position_budget is None:
        raise ValueError("position_budget_max requires position_budget.")
    if (
        position_budget is not None
        and position_budget_max is not None
        and position_budget_max < position_budget
    ):
        raise ValueError("position_budget_max must be >= position_budget.")
    subdirs = [d for d in os.listdir(raw_dir)
               if os.path.isdir(os.path.join(raw_dir, d))]

    gen_to_dirs = _collect_generation_dirs(subdirs)
    gen_numbers = sorted(gen_to_dirs.keys())
    kept_gens = []
    position_total = 0
    generation_position_counts = {}
    if keep_generations is not None:
        kept_gens = gen_numbers[-keep_generations:] if gen_numbers else []
    elif position_budget is not None:
        for gen in reversed(gen_numbers):
            gen_dirs = gen_to_dirs.get(gen, [])
            gen_pos = sum(_count_positions_in_dir(raw_dir, d) for d in gen_dirs)
            generation_position_counts[gen] = gen_pos
            kept_gens.append(gen)
            position_total += gen_pos
            if position_total >= position_budget:
                if position_budget_max is not None:
                    # Trim oldest generations while preserving the lower bound.
                    while len(kept_gens) > 1 and position_total > position_budget_max:
                        oldest_gen = kept_gens[-1]
                        oldest_pos = generation_position_counts.get(oldest_gen, 0)
                        if position_total - oldest_pos < position_budget:
                            break
                        kept_gens.pop()
                        position_total -= oldest_pos
                break
        kept_gens.sort()

    kept_gen_set = set(kept_gens)
    kept_gen_dirs = set()
    for d in subdirs:
        m = re.match(r'^nn_gen(\d+)(?:_.*)?$', d)
        if m and int(m.group(1)) in kept_gen_set:
            kept_gen_dirs.add(d)

    # Always-include dirs
    always_include = {"curriculum_bootstrap"}
    if include_human:
        always_include.add("human_games")

    include = []
    exclude = []
    for d in sorted(subdirs):
        if d in always_include or d in kept_gen_dirs:
            include.append(d)
        else:
            exclude.append(d)

    summary = {
        "kept_generations": kept_gens,
        "position_budget": position_budget,
        "position_budget_max": position_budget_max,
        "estimated_positions": position_total,
        "generation_position_counts": generation_position_counts,
    }
    return include, exclude, summary


def load_all_games(
    raw_dir,
    keep_generations=None,
    position_budget=None,
    position_budget_max=None,
    include_human=True,
    min_blackfocus_plies=0,
    blackfocus_result_filter="any",
    max_generation_age=DATA_RETENTION_MAX_GENERATION_AGE,
    min_nonhuman_plies=DATA_RETENTION_MIN_NONHUMAN_PLIES,
    min_humanseed_policy_entropy=DATA_RETENTION_MIN_HUMANSEED_POLICY_ENTROPY,
    return_summary=False,
):
    """Load game files as game-level units.

    Returns a list of dicts (or list + retention summary when return_summary=True):
      {
        "game_id": relative path,
        "records": [position records...],
        "is_human": bool,
        "result_bucket": -1/0/1 (sign of game_result)
      }

    This keeps game boundaries intact so train/val/test can be split
    at the game level (no leakage across splits).
    """
    paths = []

    if keep_generations is not None or position_budget is not None:
        include, exclude, summary = _filter_dirs(
            raw_dir,
            keep_generations=keep_generations,
            position_budget=position_budget,
            position_budget_max=position_budget_max,
            include_human=include_human,
        )
        if position_budget is not None:
            budget_str = f"{position_budget}"
            if position_budget_max is not None:
                budget_str = f"{position_budget}..{position_budget_max}"
            print(
                f"Position budget window: target={budget_str}, "
                f"estimated={summary['estimated_positions']} raw positions"
            )
            print(f"  Generations kept: {summary['kept_generations']}")
        else:
            print(f"Sliding window: keeping last {keep_generations} generations")
        print(f"  Human games: {'included' if include_human else 'excluded'}")
        print(f"  Include: {', '.join(include)}")
        if exclude:
            print(f"  Exclude: {', '.join(exclude)}")
        for d in include:
            dpath = os.path.join(raw_dir, d)
            for fname in sorted(os.listdir(dpath)):
                if fname.endswith(".jsonl"):
                    paths.append(os.path.join(dpath, fname))
    else:
        for dirpath, _dirnames, filenames in os.walk(raw_dir):
            rel_dir = os.path.relpath(dirpath, raw_dir).replace("\\", "/")
            if (not include_human) and (rel_dir == "human_games" or rel_dir.startswith("human_games/")):
                continue
            for fname in sorted(filenames):
                if fname.endswith(".jsonl"):
                    paths.append(os.path.join(dirpath, fname))

    if not paths:
        print(f"No .jsonl files found in {raw_dir}")
        empty_summary = {
            "max_generation_age": int(max_generation_age or 0),
            "min_nonhuman_plies": int(min_nonhuman_plies or 0),
            "min_humanseed_policy_entropy": float(min_humanseed_policy_entropy or 0.0),
            "latest_generation": None,
            "input_games": 0,
            "input_positions": 0,
            "kept_games": 0,
            "kept_positions": 0,
            "dropped": {
                "age": {"games": 0, "positions": 0, "by_source": {}},
                "short_nonhuman": {"games": 0, "positions": 0, "by_source": {}},
                "low_entropy_humanseed": {"games": 0, "positions": 0, "by_source": {}},
            },
        }
        return ([], empty_summary) if return_summary else []

    games = []
    skipped_blackfocus_short = 0
    skipped_blackfocus_short_positions = 0
    skipped_blackfocus_result = 0
    skipped_blackfocus_result_positions = 0
    for path in tqdm(paths, desc="Loading games"):
        rel = os.path.relpath(path, raw_dir).replace("\\", "/")
        is_human = "human_games/" in rel or rel.startswith("human_games")
        is_humanseed = "_humanseed/" in rel or rel.endswith("_humanseed")
        generation = _generation_from_game_id(rel)
        with open(path, "r") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]
        if not records:
            continue
        is_blackfocus = "_blackfocus/" in rel or rel.endswith("_blackfocus")
        if (not is_human) and is_blackfocus and len(records) < min_blackfocus_plies:
            skipped_blackfocus_short += 1
            skipped_blackfocus_short_positions += len(records)
            continue
        result = records[-1].get("game_result", 0)
        result_bucket = _result_bucket(result)
        if (not is_human) and is_blackfocus:
            if blackfocus_result_filter == "nonloss" and result_bucket > 0:
                skipped_blackfocus_result += 1
                skipped_blackfocus_result_positions += len(records)
                continue
            if blackfocus_result_filter == "win" and result_bucket >= 0:
                skipped_blackfocus_result += 1
                skipped_blackfocus_result_positions += len(records)
                continue
        games.append({
            "game_id": rel,
            "records": records,
            "is_human": is_human,
            "is_humanseed": is_humanseed,
            "is_blackfocus": is_blackfocus,
            "result_bucket": result_bucket,
            "generation": generation,
            "mean_policy_entropy": _mean_record_policy_entropy(records),
            "source_kind": (
                "human" if is_human
                else "humanseed" if is_humanseed
                else "blackfocus" if is_blackfocus
                else "selfplay"
            ),
        })

    if min_blackfocus_plies > 0:
        print(
            f"  Black-focus short-game filter: min_plies={min_blackfocus_plies}, "
            f"skipped_games={skipped_blackfocus_short}, "
            f"skipped_positions={skipped_blackfocus_short_positions}"
        )
    if blackfocus_result_filter != "any":
        print(
            f"  Black-focus result filter: mode={blackfocus_result_filter}, "
            f"skipped_games={skipped_blackfocus_result}, "
            f"skipped_positions={skipped_blackfocus_result_positions}"
        )

    games, retention_summary = _apply_game_retention_policy(
        games,
        max_generation_age=max_generation_age,
        min_nonhuman_plies=min_nonhuman_plies,
        min_humanseed_policy_entropy=min_humanseed_policy_entropy,
    )
    dropped_age = int(retention_summary["dropped"]["age"]["games"])
    dropped_short = int(retention_summary["dropped"]["short_nonhuman"]["games"])
    dropped_low_entropy = int(retention_summary["dropped"]["low_entropy_humanseed"]["games"])
    if dropped_age or dropped_short or dropped_low_entropy:
        print(
            "  Retention policy: "
            f"max_generation_age={retention_summary['max_generation_age']}, "
            f"min_nonhuman_plies={retention_summary['min_nonhuman_plies']}, "
            f"min_humanseed_policy_entropy={retention_summary['min_humanseed_policy_entropy']:.3f}, "
            f"dropped_age_games={dropped_age}, "
            f"dropped_short_nonhuman_games={dropped_short}, "
            f"dropped_low_entropy_humanseed_games={dropped_low_entropy}"
        )

    return (games, retention_summary) if return_summary else games


def _result_bucket(result):
    """Bucket scalar game results to {-1, 0, +1} by sign."""
    if result > 0:
        return 1
    if result < 0:
        return -1
    return 0


def _policy_entropy(policy_vec):
    """Shannon entropy (nats) of a dense policy vector."""
    probs = policy_vec[policy_vec > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _init_source_stats():
    stats = {}
    for src in SOURCE_ORDER:
        stats[src] = {
            "count": 0,
            "mcts_value_sum": 0.0,
            "mcts_value_sumsq": 0.0,
            "mcts_value_min": None,
            "mcts_value_max": None,
            "game_result_sum": 0.0,
            "target_lambda_sum": 0.0,
            "policy_entropy_sum": 0.0,
            "policy_entropy_sumsq": 0.0,
            "policy_entropy_min": None,
            "policy_entropy_max": None,
        }
    return stats


def _ensure_source_stats(stats, source_kind):
    if source_kind not in stats:
        stats[source_kind] = {
            "count": 0,
            "mcts_value_sum": 0.0,
            "mcts_value_sumsq": 0.0,
            "mcts_value_min": None,
            "mcts_value_max": None,
            "game_result_sum": 0.0,
            "target_lambda_sum": 0.0,
            "policy_entropy_sum": 0.0,
            "policy_entropy_sumsq": 0.0,
            "policy_entropy_min": None,
            "policy_entropy_max": None,
        }
    return stats[source_kind]


def _record_source_stat(stats, source_kind, val, game_result, target_lambda, policy_entropy):
    src = _ensure_source_stats(stats, source_kind)
    src["count"] += 1
    src["mcts_value_sum"] += float(val)
    src["mcts_value_sumsq"] += float(val) * float(val)
    src["game_result_sum"] += float(game_result)
    src["target_lambda_sum"] += float(target_lambda)
    src["policy_entropy_sum"] += float(policy_entropy)
    src["policy_entropy_sumsq"] += float(policy_entropy) * float(policy_entropy)
    src["mcts_value_min"] = float(val) if src["mcts_value_min"] is None else min(src["mcts_value_min"], float(val))
    src["mcts_value_max"] = float(val) if src["mcts_value_max"] is None else max(src["mcts_value_max"], float(val))
    src["policy_entropy_min"] = (
        float(policy_entropy)
        if src["policy_entropy_min"] is None
        else min(src["policy_entropy_min"], float(policy_entropy))
    )
    src["policy_entropy_max"] = (
        float(policy_entropy)
        if src["policy_entropy_max"] is None
        else max(src["policy_entropy_max"], float(policy_entropy))
    )


def _finalize_source_stats(source_stats):
    out = {}
    for source_kind, src in source_stats.items():
        count = int(src.get("count", 0))
        if count <= 0:
            out[source_kind] = {"count": 0}
            continue
        mcts_mean = float(src["mcts_value_sum"] / count)
        mcts_var = max(0.0, float(src["mcts_value_sumsq"] / count) - (mcts_mean * mcts_mean))
        ent_mean = float(src["policy_entropy_sum"] / count)
        ent_var = max(0.0, float(src["policy_entropy_sumsq"] / count) - (ent_mean * ent_mean))
        out[source_kind] = {
            "count": count,
            "mcts_value": {
                "mean": mcts_mean,
                "std": float(np.sqrt(mcts_var)),
                "min": float(src["mcts_value_min"]),
                "max": float(src["mcts_value_max"]),
            },
            "game_result_mean": float(src["game_result_sum"] / count),
            "target_lambda_mean": float(src["target_lambda_sum"] / count),
            "policy_entropy": {
                "mean": ent_mean,
                "std": float(np.sqrt(ent_var)),
                "min": float(src["policy_entropy_min"]),
                "max": float(src["policy_entropy_max"]),
            },
        }
    return out


def _build_processing_warnings(train_source_counts, train_source_stats, train_source_quotas):
    warnings = []
    total_train = int(sum(int(v) for v in train_source_counts.values()))
    present_sources = [s for s, c in train_source_counts.items() if int(c) > 0]
    if total_train <= 0:
        warnings.append("Train split contains zero positions after processing")
        return warnings

    if len(present_sources) < 2:
        warnings.append(
            f"Low source diversity in train split ({len(present_sources)} active source)"
        )

    for source_kind, count_raw in train_source_counts.items():
        count = int(count_raw)
        if count <= 0:
            continue
        frac = count / total_train
        if frac < 0.02:
            warnings.append(
                f"Train source '{source_kind}' is underrepresented ({count}/{total_train}, {100.0 * frac:.2f}%)"
            )

    if isinstance(train_source_quotas, dict):
        for source_kind, quota_raw in train_source_quotas.items():
            quota = int(quota_raw)
            if quota <= 0:
                continue
            got = int(train_source_counts.get(source_kind, 0))
            if got < max(50, int(0.5 * quota)):
                warnings.append(
                    f"Train source '{source_kind}' underfilled vs quota ({got}/{quota})"
                )

    for source_kind, stats in train_source_stats.items():
        count = int(stats.get("count", 0))
        if count < 200:
            continue
        mcts_mean = float(stats.get("mcts_value", {}).get("mean", 0.0))
        ent_mean = float(stats.get("policy_entropy", {}).get("mean", 0.0))
        if abs(mcts_mean) > 0.75:
            warnings.append(
                f"Train source '{source_kind}' has high |mcts_value mean| ({mcts_mean:.3f}); labels may be skewed"
            )
        if ent_mean < 0.35:
            warnings.append(
                f"Train source '{source_kind}' has low policy entropy mean ({ent_mean:.3f}); move targets may be collapsed"
            )

    return warnings


def _split_games_by_result(games, seed):
    """Stratified game-level split (80/10/10) by result bucket."""
    rng = np.random.default_rng(seed)
    by_bucket = {-1: [], 0: [], 1: []}
    for game in games:
        by_bucket[game["result_bucket"]].append(game)

    split = {"train": [], "val": [], "test": []}
    for bucket in (-1, 0, 1):
        group = by_bucket[bucket]
        if not group:
            continue
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        split["train"].extend(group[:n_train])
        split["val"].extend(group[n_train:n_train + n_val])
        split["test"].extend(group[n_train + n_val:])

    for key in ("train", "val", "test"):
        rng.shuffle(split[key])

    # Small-dataset fallback: keep splits non-empty when possible.
    if len(games) >= 3 and len(split["val"]) == 0:
        donor = "test" if len(split["test"]) > 1 else "train"
        if split[donor]:
            split["val"].append(split[donor].pop())
    if len(games) >= 2 and len(split["test"]) == 0:
        donor = "val" if len(split["val"]) > 1 else "train"
        if split[donor]:
            split["test"].append(split[donor].pop())
    if len(split["train"]) == 0:
        donor = "test" if split["test"] else "val"
        if split[donor]:
            split["train"].append(split[donor].pop())

    return split


def _convert_games_to_arrays(games, augment, human_repeat,
                             blackfocus_repeat=1, humanseed_repeat=1,
                             max_positions=None, source_quotas=None,
                             source_target_lambdas=None,
                             dedupe_positions=False):
    """Convert game records to tensors for one split."""
    if source_target_lambdas is None:
        source_target_lambdas = _build_source_target_lambdas(
            selfplay_target_mcts_lambda=SELFPLAY_TARGET_MCTS_LAMBDA,
            human_target_mcts_lambda=HUMAN_TARGET_MCTS_LAMBDA,
            blackfocus_target_mcts_lambda=BLACKFOCUS_TARGET_MCTS_LAMBDA,
            humanseed_target_mcts_lambda=HUMANSEED_TARGET_MCTS_LAMBDA,
        )
    tensors = []
    values = []
    game_results = []
    policy_targets = []
    target_lambdas = []
    split_game_ids = []
    source_counts = {s: 0 for s in SOURCE_ORDER}
    source_stats = _init_source_stats()
    capped = False
    seen_positions = set()

    for game in tqdm(games, desc="Converting", leave=False):
        if max_positions is not None and len(tensors) >= max_positions:
            capped = True
            break
        source_kind = game.get("source_kind") or _source_kind(game)
        if source_kind not in source_counts:
            source_counts[source_kind] = 0
        _ensure_source_stats(source_stats, source_kind)
        source_quota = None if source_quotas is None else source_quotas.get(source_kind)
        if source_quota is not None and source_counts[source_kind] >= source_quota:
            continue
        repeat = 1
        if game.get("is_human"):
            repeat = max(repeat, int(human_repeat))
        if game.get("is_blackfocus"):
            repeat = max(repeat, int(blackfocus_repeat))
        if game.get("is_humanseed"):
            repeat = max(repeat, int(humanseed_repeat))
        for _ in range(repeat):
            if max_positions is not None and len(tensors) >= max_positions:
                capped = True
                break
            if source_quota is not None and source_counts[source_kind] >= source_quota:
                break
            split_game_ids.append(game["game_id"])
            for rec in game["records"]:
                if max_positions is not None and len(tensors) >= max_positions:
                    capped = True
                    break
                if source_quota is not None and source_counts[source_kind] >= source_quota:
                    break
                is_white = rec["current_player"] == "white"
                pos_key = None
                dedupe_active = dedupe_positions and source_kind in ("human", "humanseed")
                if dedupe_active:
                    pos_key = f"{rec['fen']}|{rec['current_player']}"
                    if pos_key in seen_positions:
                        continue
                tensor = fen_to_tensor(rec["fen"], is_white_turn=is_white)
                # mcts_value from data_generation is already from the
                # side-to-move perspective for both White and Black.
                mv = rec["mcts_value"]
                val = mv
                gr = rec["game_result"]
                pol = policy_dict_to_target(rec["policy"], is_white)
                lam = float(source_target_lambdas.get(source_kind, 1.0))
                pol_entropy = _policy_entropy(pol)

                tensors.append(tensor)
                values.append(val)
                game_results.append(gr)
                policy_targets.append(pol)
                target_lambdas.append(lam)
                source_counts[source_kind] += 1
                _record_source_stat(
                    source_stats, source_kind, val, gr, lam, pol_entropy
                )
                if dedupe_active and pos_key is not None:
                    seen_positions.add(pos_key)

                can_augment = augment and (max_positions is None or len(tensors) < max_positions)
                if source_quota is not None and source_counts[source_kind] >= source_quota:
                    can_augment = False
                if can_augment:
                    tensors.append(mirror_tensor(tensor))
                    values.append(val)
                    game_results.append(gr)
                    policy_targets.append(mirror_policy(pol))
                    target_lambdas.append(lam)
                    source_counts[source_kind] += 1
                    _record_source_stat(
                        source_stats, source_kind, val, gr, lam, pol_entropy
                    )
                elif augment and max_positions is not None and len(tensors) >= max_positions:
                    capped = True
                    break
            if capped:
                break
        if capped:
            break

    if tensors:
        X = np.array(tensors, dtype=np.float32)
        y_value = np.array(values, dtype=np.float32)
        y_result = np.array(game_results, dtype=np.float32)
        y_policy = np.array(policy_targets, dtype=np.float32)
        y_target_lambda = np.array(target_lambdas, dtype=np.float32)
    else:
        X = np.zeros((0,) + TENSOR_SHAPE, dtype=np.float32)
        y_value = np.zeros((0,), dtype=np.float32)
        y_result = np.zeros((0,), dtype=np.float32)
        y_policy = np.zeros((0, POLICY_SIZE), dtype=np.float32)
        y_target_lambda = np.zeros((0,), dtype=np.float32)

    return (
        X, y_value, y_result, y_policy, y_target_lambda,
        split_game_ids, capped, source_counts, _finalize_source_stats(source_stats),
    )


def process_raw_data(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR,
                     augment=True, keep_generations=None, position_budget=None,
                     position_budget_max=None,
                     seed=RANDOM_SEED,
                     include_human=True,
                     min_blackfocus_plies=0,
                     blackfocus_result_filter="any",
                     max_generation_age=DATA_RETENTION_MAX_GENERATION_AGE,
                     min_nonhuman_plies=DATA_RETENTION_MIN_NONHUMAN_PLIES,
                     min_humanseed_policy_entropy=DATA_RETENTION_MIN_HUMANSEED_POLICY_ENTROPY,
                     human_repeat=HUMAN_DATA_WEIGHT,
                     humanseed_repeat=HUMANSEED_DATA_WEIGHT,
                     blackfocus_repeat=BLACKFOCUS_DATA_WEIGHT,
                     max_positions=PROCESSED_POSITION_CAP,
                     use_source_quotas=SOURCE_QUOTA_ENABLED,
                     quota_selfplay=SOURCE_QUOTA_SELFPLAY,
                     quota_human=SOURCE_QUOTA_HUMAN,
                     quota_blackfocus=SOURCE_QUOTA_BLACKFOCUS,
                     quota_humanseed=SOURCE_QUOTA_HUMANSEED,
                     human_target_mcts_lambda=HUMAN_TARGET_MCTS_LAMBDA,
                     humanseed_target_mcts_lambda=HUMANSEED_TARGET_MCTS_LAMBDA,
                     blackfocus_target_mcts_lambda=BLACKFOCUS_TARGET_MCTS_LAMBDA,
                     selfplay_target_mcts_lambda=SELFPLAY_TARGET_MCTS_LAMBDA):
    """Convert raw game records to training tensors and save.

    When augment=True (default), each position is also horizontally
    mirrored, doubling the dataset size.  The mirror preserves the
    evaluation (same value / game_result) since Monster Chess is
    file-symmetric.

    When keep_generations is set, only the last N NN generations are loaded.
    When position_budget is set, enough recent generations are loaded to hit
    at least that many raw positions.
    Retention controls can also drop very old generations and ultra-short
    non-human games before train/val/test splitting.
    """
    games, retention_summary = load_all_games(
        raw_dir,
        keep_generations=keep_generations,
        position_budget=position_budget,
        position_budget_max=position_budget_max,
        include_human=include_human,
        min_blackfocus_plies=min_blackfocus_plies,
        blackfocus_result_filter=blackfocus_result_filter,
        max_generation_age=max_generation_age,
        min_nonhuman_plies=min_nonhuman_plies,
        min_humanseed_policy_entropy=min_humanseed_policy_entropy,
        return_summary=True,
    )
    if not games:
        print("No data to process.")
        return
    if isinstance(retention_summary, dict):
        print(
            "Retention summary: "
            f"kept_games={int(retention_summary.get('kept_games', 0))}/"
            f"{int(retention_summary.get('input_games', 0))}, "
            f"kept_positions={int(retention_summary.get('kept_positions', 0))}/"
            f"{int(retention_summary.get('input_positions', 0))}"
        )
    if max_positions is not None and max_positions <= 0:
        max_positions = None
    source_quota_ratios = _build_source_quota_ratios(
        quota_selfplay=quota_selfplay,
        quota_human=quota_human,
        quota_blackfocus=quota_blackfocus,
        quota_humanseed=quota_humanseed,
    )
    source_target_lambdas = _build_source_target_lambdas(
        selfplay_target_mcts_lambda=selfplay_target_mcts_lambda,
        human_target_mcts_lambda=human_target_mcts_lambda,
        blackfocus_target_mcts_lambda=blackfocus_target_mcts_lambda,
        humanseed_target_mcts_lambda=humanseed_target_mcts_lambda,
    )

    split_games = _split_games_by_result(games, seed=seed)
    train_games = split_games["train"]
    val_games = split_games["val"]
    test_games = split_games["test"]

    train_ids = {g["game_id"] for g in train_games}
    val_ids = {g["game_id"] for g in val_games}
    test_ids = {g["game_id"] for g in test_games}
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError("Game-level split overlap detected")
    print("Game split integrity: PASS (no overlap across train/val/test game IDs)")
    print(f"  Games: train={len(train_games)}, val={len(val_games)}, test={len(test_games)}")
    print(f"  Processing positions (augment={augment})...")
    print(
        "  Train repetition weights: "
        f"human={human_repeat}, "
        f"humanseed={humanseed_repeat}, "
        f"blackfocus={blackfocus_repeat}"
    )
    print(
        "  Source-aware value lambdas (mcts weight): "
        f"selfplay={source_target_lambdas['selfplay']:.2f}, "
        f"human={source_target_lambdas['human']:.2f}, "
        f"blackfocus={source_target_lambdas['blackfocus']:.2f}, "
        f"humanseed={source_target_lambdas['humanseed']:.2f}"
    )
    if max_positions is not None:
        print(f"  Processed position cap: total<={max_positions}")

    # Validation/test are never weighted.
    (
        X_val, yv_val, yr_val, yp_val, yl_val,
        val_game_ids, _, val_source_counts, val_source_stats,
    ) = _convert_games_to_arrays(
        val_games,
        augment=augment,
        human_repeat=1,
        blackfocus_repeat=1,
        humanseed_repeat=1,
        source_target_lambdas=source_target_lambdas,
    )
    (
        X_test, yv_test, yr_test, yp_test, yl_test,
        test_game_ids, _, test_source_counts, test_source_stats,
    ) = _convert_games_to_arrays(
        test_games,
        augment=augment,
        human_repeat=1,
        blackfocus_repeat=1,
        humanseed_repeat=1,
        source_target_lambdas=source_target_lambdas,
    )
    eval_positions = len(X_val) + len(X_test)
    train_position_cap = None
    if max_positions is not None:
        if eval_positions >= max_positions:
            raise ValueError(
                "Validation+test positions already exceed --max-positions "
                f"({eval_positions} >= {max_positions}). Increase the cap."
            )
        train_position_cap = max_positions - eval_positions
        print(
            f"  Train position cap after val/test reservation: "
            f"{train_position_cap} (val+test={eval_positions})"
        )

    train_source_quotas = None
    if use_source_quotas and train_position_cap is not None:
        train_source_quotas = _allocate_source_quotas(
            train_position_cap,
            source_quota_ratios,
            train_games,
        )
        if train_source_quotas:
            print(
                "  Train source quotas: "
                + ", ".join(f"{k}={train_source_quotas.get(k, 0)}" for k in SOURCE_ORDER)
            )
        else:
            print("  Train source quotas: disabled (no active sources/ratios)")
    elif use_source_quotas and train_position_cap is None:
        print("  Train source quotas: skipped (no max position cap)")
    else:
        print("  Train source quotas: disabled")

    train_games_ordered = train_games
    if use_source_quotas:
        train_games_ordered = _interleave_games_by_source(train_games, seed=seed + 17)

    # Upweight targeted game sources only in TRAIN split to avoid validation/test skew.
    (
        X_train, yv_train, yr_train, yp_train, yl_train,
        train_game_ids, train_capped, train_source_counts, train_source_stats,
    ) = _convert_games_to_arrays(
        train_games_ordered,
        augment=augment,
        human_repeat=human_repeat,
        blackfocus_repeat=blackfocus_repeat,
        humanseed_repeat=humanseed_repeat,
        max_positions=train_position_cap,
        source_quotas=train_source_quotas,
        source_target_lambdas=source_target_lambdas,
        dedupe_positions=bool(use_source_quotas),
    )

    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y_value = np.concatenate([yv_train, yv_val, yv_test], axis=0)
    y_result = np.concatenate([yr_train, yr_val, yr_test], axis=0)
    y_policy = np.concatenate([yp_train, yp_val, yp_test], axis=0)
    y_target_lambda = np.concatenate([yl_train, yl_val, yl_test], axis=0)
    total_positions = len(X)
    if max_positions is not None and total_positions > max_positions:
        raise RuntimeError(
            f"Processed position cap violated ({total_positions} > {max_positions})"
        )
    if train_capped:
        print(f"  Train split capped at {len(X_train)} positions to honor max total size")
    processing_warnings = _build_processing_warnings(
        train_source_counts=train_source_counts,
        train_source_stats=train_source_stats,
        train_source_quotas=train_source_quotas,
    )
    if processing_warnings:
        for msg in processing_warnings:
            print(f"  WARNING: {msg}")

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "positions.npy"), X)
    np.save(os.path.join(output_dir, "mcts_values.npy"), y_value)
    np.save(os.path.join(output_dir, "game_results.npy"), y_result)
    np.save(os.path.join(output_dir, "policies.npy"), y_policy)
    np.save(os.path.join(output_dir, "target_lambdas.npy"), y_target_lambda)

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    splits = {
        "train": np.arange(0, n_train, dtype=np.int64),
        "val": np.arange(n_train, n_train + n_val, dtype=np.int64),
        "test": np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64),
    }
    np.savez(os.path.join(output_dir, "splits.npz"), **splits)

    split_game_ids = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
        "train_weighted_instances": len(train_game_ids),
        "val_weighted_instances": len(val_game_ids),
        "test_weighted_instances": len(test_game_ids),
        "train_source_position_counts": train_source_counts,
        "val_source_position_counts": val_source_counts,
        "test_source_position_counts": test_source_counts,
        "train_source_quotas": train_source_quotas,
        "source_quota_ratios": source_quota_ratios,
        "source_target_lambdas": source_target_lambdas,
        "retention": retention_summary,
        "max_positions": max_positions,
        "train_split_capped": bool(train_capped),
        "total_positions": int(total_positions),
    }
    with open(os.path.join(output_dir, "split_game_ids.json"), "w") as f:
        json.dump(split_game_ids, f, indent=2)
    processing_summary = {
        "positions_shape": list(X.shape),
        "split_sizes": {
            "train": int(n_train),
            "val": int(n_val),
            "test": int(n_test),
        },
        "source_counts": {
            "train": train_source_counts,
            "val": val_source_counts,
            "test": test_source_counts,
        },
        "source_stats": {
            "train": train_source_stats,
            "val": val_source_stats,
            "test": test_source_stats,
        },
        "source_quota_ratios": source_quota_ratios,
        "source_quotas": train_source_quotas,
        "source_target_lambdas": source_target_lambdas,
        "retention": retention_summary,
        "train_split_capped": bool(train_capped),
        "total_positions": int(total_positions),
        "weights": {
            "human": int(human_repeat),
            "humanseed": int(humanseed_repeat),
            "blackfocus": int(blackfocus_repeat),
        },
        "augment": bool(augment),
        "warnings": processing_warnings,
    }
    with open(os.path.join(output_dir, "processing_summary.json"), "w") as f:
        json.dump(processing_summary, f, indent=2)

    print(f"\nSaved to {output_dir}:")
    print(f"  positions.npy:    {X.shape}")
    print(f"  mcts_values.npy:  {y_value.shape}")
    print(f"  game_results.npy: {y_result.shape}")
    print(f"  policies.npy:     {y_policy.shape}")
    print(f"  target_lambdas.npy: {y_target_lambda.shape}")
    print(f"  splits.npz:       train={n_train}, val={n_val}, test={n_test}")
    print("  split_game_ids.json: game-level split membership saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process raw game data into training tensors")
    parser.add_argument("--raw-dir", type=str, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DATA_DIR)
    parser.add_argument("--no-augment", action="store_true", help="Disable mirror augmentation")
    parser.add_argument("--keep-generations", type=int, default=None,
                        help=f"Sliding window: keep last N generations (default: all)")
    parser.add_argument("--position-budget", type=int, default=None,
                        help=f"Position budget window: include enough recent generations to hit N raw positions (default: {POSITION_BUDGET})")
    parser.add_argument("--position-budget-max", type=int, default=None,
                        help=f"Optional max-cap for position budget window (default: {POSITION_BUDGET_MAX})")
    parser.add_argument("--max-positions", type=int, default=PROCESSED_POSITION_CAP,
                        help=f"Hard cap on processed train+val+test positions (default: {PROCESSED_POSITION_CAP}, <=0 disables)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for deterministic game-level splitting (default: {RANDOM_SEED})")
    parser.add_argument("--exclude-human-games", action="store_true",
                        help="Exclude data/raw/human_games from processing")
    parser.add_argument("--min-blackfocus-plies", type=int, default=0,
                        help="Drop non-human _blackfocus games shorter than this many plies")
    parser.add_argument("--blackfocus-result-filter", type=str, default="any",
                        choices=["any", "nonloss", "win"],
                        help="Keep _blackfocus games with any result, Black non-loss, or Black win only")
    parser.add_argument("--max-generation-age", type=int, default=DATA_RETENTION_MAX_GENERATION_AGE,
                        help=f"Drop nn_gen* games older than this many generations behind latest (default: {DATA_RETENTION_MAX_GENERATION_AGE}, <=0 disables)")
    parser.add_argument("--min-nonhuman-plies", type=int, default=DATA_RETENTION_MIN_NONHUMAN_PLIES,
                        help=f"Drop non-human games shorter than this many plies (default: {DATA_RETENTION_MIN_NONHUMAN_PLIES}, <=0 disables)")
    parser.add_argument("--min-humanseed-policy-entropy", type=float, default=DATA_RETENTION_MIN_HUMANSEED_POLICY_ENTROPY,
                        help=f"Drop human-seed games with mean policy entropy below this threshold (default: {DATA_RETENTION_MIN_HUMANSEED_POLICY_ENTROPY}, <=0 disables)")
    parser.add_argument("--human-data-weight", type=int, default=HUMAN_DATA_WEIGHT,
                        help=f"Train repetition weight for human_games (default: {HUMAN_DATA_WEIGHT})")
    parser.add_argument("--humanseed-data-weight", type=int, default=HUMANSEED_DATA_WEIGHT,
                        help=f"Train repetition weight for _humanseed streams (default: {HUMANSEED_DATA_WEIGHT})")
    parser.add_argument("--blackfocus-data-weight", type=int, default=BLACKFOCUS_DATA_WEIGHT,
                        help=f"Train repetition weight for _blackfocus streams (default: {BLACKFOCUS_DATA_WEIGHT})")
    parser.add_argument("--use-source-quotas", action=argparse.BooleanOptionalAction,
                        default=SOURCE_QUOTA_ENABLED,
                        help="Cap train positions by source ratio within --max-positions")
    parser.add_argument("--quota-selfplay", type=float, default=SOURCE_QUOTA_SELFPLAY,
                        help=f"Source quota ratio for selfplay/curriculum streams (default: {SOURCE_QUOTA_SELFPLAY})")
    parser.add_argument("--quota-human", type=float, default=SOURCE_QUOTA_HUMAN,
                        help=f"Source quota ratio for human_games stream (default: {SOURCE_QUOTA_HUMAN})")
    parser.add_argument("--quota-blackfocus", type=float, default=SOURCE_QUOTA_BLACKFOCUS,
                        help=f"Source quota ratio for _blackfocus streams (default: {SOURCE_QUOTA_BLACKFOCUS})")
    parser.add_argument("--quota-humanseed", type=float, default=SOURCE_QUOTA_HUMANSEED,
                        help=f"Source quota ratio for _humanseed streams (default: {SOURCE_QUOTA_HUMANSEED})")
    parser.add_argument("--human-target-mcts-lambda", type=float, default=HUMAN_TARGET_MCTS_LAMBDA,
                        help=f"Human source mcts target weight lambda (default: {HUMAN_TARGET_MCTS_LAMBDA})")
    parser.add_argument("--humanseed-target-mcts-lambda", type=float, default=HUMANSEED_TARGET_MCTS_LAMBDA,
                        help=f"Human-seed source mcts target weight lambda (default: {HUMANSEED_TARGET_MCTS_LAMBDA})")
    parser.add_argument("--blackfocus-target-mcts-lambda", type=float, default=BLACKFOCUS_TARGET_MCTS_LAMBDA,
                        help=f"Black-focus source mcts target weight lambda (default: {BLACKFOCUS_TARGET_MCTS_LAMBDA})")
    parser.add_argument("--selfplay-target-mcts-lambda", type=float, default=SELFPLAY_TARGET_MCTS_LAMBDA,
                        help=f"Self-play source mcts target weight lambda (default: {SELFPLAY_TARGET_MCTS_LAMBDA})")
    args = parser.parse_args()
    if args.keep_generations is not None and args.position_budget is not None:
        raise ValueError("Specify only one of --keep-generations or --position-budget")
    if args.position_budget_max is not None and args.position_budget is None:
        raise ValueError("--position-budget-max requires --position-budget")
    if (
        args.position_budget is not None
        and args.position_budget_max is not None
        and args.position_budget_max < args.position_budget
    ):
        raise ValueError("--position-budget-max must be >= --position-budget")
    if args.min_blackfocus_plies < 0:
        raise ValueError("--min-blackfocus-plies must be >= 0")
    if args.max_generation_age is not None and args.max_generation_age < 0:
        raise ValueError("--max-generation-age must be >= 0")
    if args.min_nonhuman_plies < 0:
        raise ValueError("--min-nonhuman-plies must be >= 0")
    if args.min_humanseed_policy_entropy < 0.0:
        raise ValueError("--min-humanseed-policy-entropy must be >= 0")
    if args.human_data_weight < 1:
        raise ValueError("--human-data-weight must be >= 1")
    if args.humanseed_data_weight < 1:
        raise ValueError("--humanseed-data-weight must be >= 1")
    if args.blackfocus_data_weight < 1:
        raise ValueError("--blackfocus-data-weight must be >= 1")
    _validate_source_settings(
        use_source_quotas=args.use_source_quotas,
        source_quota_ratios=_build_source_quota_ratios(
            quota_selfplay=args.quota_selfplay,
            quota_human=args.quota_human,
            quota_blackfocus=args.quota_blackfocus,
            quota_humanseed=args.quota_humanseed,
        ),
        source_target_lambdas=_build_source_target_lambdas(
            selfplay_target_mcts_lambda=args.selfplay_target_mcts_lambda,
            human_target_mcts_lambda=args.human_target_mcts_lambda,
            blackfocus_target_mcts_lambda=args.blackfocus_target_mcts_lambda,
            humanseed_target_mcts_lambda=args.humanseed_target_mcts_lambda,
        ),
    )
    if args.max_positions is not None and args.max_positions <= 0:
        args.max_positions = None

    process_raw_data(raw_dir=args.raw_dir, output_dir=args.output_dir,
                     augment=not args.no_augment,
                     keep_generations=args.keep_generations,
                     position_budget=args.position_budget,
                     position_budget_max=args.position_budget_max,
                     seed=args.seed,
                     include_human=not args.exclude_human_games,
                     min_blackfocus_plies=args.min_blackfocus_plies,
                     blackfocus_result_filter=args.blackfocus_result_filter,
                     max_generation_age=args.max_generation_age,
                     min_nonhuman_plies=args.min_nonhuman_plies,
                     min_humanseed_policy_entropy=args.min_humanseed_policy_entropy,
                     human_repeat=args.human_data_weight,
                     humanseed_repeat=args.humanseed_data_weight,
                     blackfocus_repeat=args.blackfocus_data_weight,
                     max_positions=args.max_positions,
                     use_source_quotas=args.use_source_quotas,
                     quota_selfplay=args.quota_selfplay,
                     quota_human=args.quota_human,
                     quota_blackfocus=args.quota_blackfocus,
                     quota_humanseed=args.quota_humanseed,
                     human_target_mcts_lambda=args.human_target_mcts_lambda,
                     humanseed_target_mcts_lambda=args.humanseed_target_mcts_lambda,
                     blackfocus_target_mcts_lambda=args.blackfocus_target_mcts_lambda,
                     selfplay_target_mcts_lambda=args.selfplay_target_mcts_lambda)
